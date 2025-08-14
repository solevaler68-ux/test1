import os
import time
import logging
import re
from typing import Any, Dict, Optional

import telebot
from dotenv import load_dotenv

from yandex_cloud_ml_sdk import YCloudML


load_dotenv()

logger = logging.getLogger(__name__)


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _strip_markdown(text: str) -> str:
    """Remove Markdown/MarkdownV2/HTML formatting and LaTeX commands, return plain text.
    
    This aims to be robust for most AI-generated outputs.
    """
    if not text:
        return ""

    result = text

    # Remove HTML tags
    result = re.sub(r"<[^>]+>", "", result)

    # Remove standalone square bracket wrapper lines and unwrap bracketed content
    result = re.sub(r"^\s*\[\s*$", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\]\s*$", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\[\s*(.*?)\s*\]\s*$", r"\1", result, flags=re.MULTILINE)

    # Remove LaTeX inline/block math delimiters but keep content
    result = re.sub(r"\$\$([\s\S]*?)\$\$", r"\1", result)
    result = re.sub(r"\$([^$\n]*?)\$", r"\1", result)

    # Simplify common LaTeX commands
    for _ in range(3):
        result = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", lambda m: f"{m.group(1)}/{m.group(2)}", result)
    result = re.sub(r"\\left|\\right", "", result)
    replacements = {
        r"\\pm": "±",
        r"\\cdot": "*",
        r"\\times": "×",
        r"\\neq": "≠",
        r"\\leq": "≤",
        r"\\geq": "≥",
        r"\\approx": "≈",
        r"\\ldots": "...",
        r"\\dots": "...",
        r"\\infty": "∞",
        r"\\sqrt": "√",
        r"\\frac": "/",
    }
    for pat, repl in replacements.items():
        result = re.sub(pat, repl, result)

    # Fenced code blocks ```lang\n...```
    result = re.sub(r"```[a-zA-Z0-9]*\n?([\s\S]*?)```", r"\1", result)

    # Inline code `code`
    result = re.sub(r"`([^`]*)`", r"\1", result)

    # Images ![alt](url) -> alt
    result = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", result)

    # Links [text](url) -> text
    result = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", result)

    # Bold/italic/strike
    result = re.sub(r"(\*\*|__)(.*?)\1", r"\2", result, flags=re.DOTALL)
    result = re.sub(r"(\*|_)(.*?)\1", r"\2", result, flags=re.DOTALL)
    result = re.sub(r"~~(.*?)~~", r"\1", result, flags=re.DOTALL)

    # Headers, blockquotes
    result = re.sub(r"^#{1,6}\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r"^>\s?", "", result, flags=re.MULTILINE)

    # Lists markers
    result = re.sub(r"^\s*[-*+]\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\d+\.\s+", "", result, flags=re.MULTILINE)

    # Remove remaining backslashes (markdown escapes) and LaTeX braces
    result = re.sub(r"\\([_\*\[\]\(\)~`>#+\-=|{}.!])", r"\1", result)
    result = re.sub(r"[{}]", "", result)

    # Collapse extra spaces introduced by removals
    result = re.sub(r"\s+\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _extract_text(result: Any) -> str:
    """Best-effort extraction of text from YandexGPT result of deferred completion.

    Supports several common shapes; falls back to str(result).
    """
    try:
        # Handle GPTModelResult objects with .alternatives attribute
        if hasattr(result, "alternatives"):
            alternatives = getattr(result, "alternatives")
            if alternatives and len(alternatives) > 0:
                alt0 = alternatives[0]
                if hasattr(alt0, "text"):
                    return str(getattr(alt0, "text"))
                # Fallback: try to access as tuple/dict
                if isinstance(alt0, (tuple, list)) and len(alt0) > 0:
                    if hasattr(alt0[0], "text"):
                        return str(getattr(alt0[0], "text"))
        # Handle dict responses
        if isinstance(result, dict):
            alts = result.get("alternatives") or result.get("choices")
            if isinstance(alts, list) and alts:
                alt0 = alts[0]
                if isinstance(alt0, dict):
                    text = alt0.get("text") or alt0.get("content")
                    if text:
                        return str(text)
        # Handle other object types with alternatives
        if hasattr(result, "alternatives"):
            alternatives = getattr(result, "alternatives")
            if isinstance(alternatives, (list, tuple)) and alternatives:
                alt0 = alternatives[0]
                if hasattr(alt0, "text"):
                    return str(getattr(alt0, "text"))
                if isinstance(alt0, dict):
                    text = alt0.get("text") or alt0.get("content")
                    if text:
                        return str(text)
    except Exception as e:
        logger.warning(f"Failed to extract text from result: {e}")
    return str(result)


def build_yandex_client() -> YCloudML:
    folder_id = os.getenv("YANDEX_CLOUD_FOLDER")
    api_key = os.getenv("YANDEX_CLOUD_API_KEY")
    if not folder_id or not api_key:
        raise RuntimeError("YANDEX_CLOUD_FOLDER and YANDEX_CLOUD_API_KEY must be set in .env")
    return YCloudML(folder_id=folder_id, auth=api_key)


def generate_reply(sdk: YCloudML, user_text: str) -> str:
    model_name = os.getenv("YANDEX_GPT_MODEL", "yandexgpt")
    temperature = float(os.getenv("YANDEX_GPT_TEMPERATURE", "0.5"))

    messages = [
        {"role": "system", "text": "Ты вежливый и умный Телеграмм помощник, помогай кратко и понятно"},
        {"role": "user", "text": user_text},
    ]

    model = sdk.models.completions(model_name)
    operation = model.configure(temperature=temperature).run_deferred(messages)

    # Poll until completed
    status = operation.get_status()
    while getattr(status, "is_running", False):
        time.sleep(2)
        status = operation.get_status()

    result = operation.get_result()
    answer = _extract_text(result)
    return answer.strip()


def main() -> None:
    # Logging
    log_level = os.getenv("YANDEX_BOT_LOG_LEVEL", os.getenv("GIGABOT_LOG_LEVEL", "INFO"))
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment/.env")

    # Disable parse mode to avoid formatting issues
    bot = telebot.TeleBot(tg_token, parse_mode=None)

    # Init Yandex client once
    sdk = build_yandex_client()

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: telebot.types.Message) -> None:
        bot.reply_to(message, "Привет! Я помощник на базе YandexGPT. Напишите вопрос.")

    @bot.message_handler(content_types=["text"])
    def handle_text(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        user_text = message.text or ""
        try:
            reply = generate_reply(sdk, user_text)
            # Clean Markdown/LaTeX formatting before sending
            clean_reply = _strip_markdown(reply)
            bot.send_message(chat_id, clean_reply, parse_mode=None)
        except Exception:
            logger.exception("Failed to get response from YandexGPT")
            bot.send_message(chat_id, "Произошла ошибка при обращении к YandexGPT. Попробуйте позже.")

    logger.info("Yandex bot is running")
    bot.infinity_polling()


if __name__ == "__main__":
    main()


