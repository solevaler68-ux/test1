from __future__ import annotations

import os
import time
import logging
import re
from typing import Dict, Optional

import telebot
from dotenv import load_dotenv

from get_token import get_gigachat_token
from gigachat import GigaChat
from yandex_cloud_ml_sdk import YCloudML


# Load .env if present
load_dotenv()

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = "Ты вежливый и профессиональный личный помощник, работающий в Telegram."


def _strip_markdown(text: str) -> str:
    """Remove Markdown/MarkdownV2/HTML/LaTeX formatting; return plain text."""
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


def build_gigachat_client() -> Optional[GigaChat]:
    """Create a configured GigaChat client using environment variables.

    Returns None if credentials are not found to allow partial functionality.
    """
    try:
        # Try to prefetch token to speed up the first request
        token_obj = get_gigachat_token()
    except Exception:
        logger.warning("GigaChat credentials are not configured or token fetch failed; GigaChat will be unavailable")
        token_obj = None

    verify_ssl_certs: Optional[bool] = None
    env_verify = os.getenv("GIGACHAT_VERIFY_SSL")
    if env_verify is not None:
        verify_ssl_certs = env_verify.strip().lower() in {"1", "true", "yes", "on"}
    ca_bundle_file = os.getenv("GIGACHAT_CA_BUNDLE_FILE")

    credentials = (
        os.getenv("GIGACHAT_API_KEY")
        or os.getenv("GIGACHAT_CREDENTIALS")
        or os.getenv("GIGACHAT_AUTHORIZATION")
    )

    try:
        client = GigaChat(
            credentials=credentials,
            verify_ssl_certs=verify_ssl_certs,
            ca_bundle_file=ca_bundle_file,
            access_token=getattr(token_obj, "access_token", None),
        )
        return client
    except Exception:
        logger.exception("Failed to initialize GigaChat client")
        return None


def build_yandex_client() -> Optional[YCloudML]:
    folder_id = os.getenv("YANDEX_CLOUD_FOLDER")
    api_key = os.getenv("YANDEX_CLOUD_API_KEY")
    if not folder_id or not api_key:
        logger.warning("YANDEX_CLOUD_FOLDER or YANDEX_CLOUD_API_KEY not set; YandexGPT will be unavailable")
        return None
    try:
        return YCloudML(folder_id=folder_id, auth=api_key)
    except Exception:
        logger.exception("Failed to initialize Yandex Cloud ML SDK client")
        return None


def generate_with_gigachat(client: GigaChat, user_text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nПользователь: {user_text.strip()}"
    completion = client.chat(prompt)
    answer = None
    try:
        if completion and getattr(completion, "choices", None):
            answer = completion.choices[0].message.content
    except Exception:
        logger.warning("Unexpected GigaChat completion format")
    return (answer or "Извините, не удалось получить ответ.").strip()


def generate_with_yandex(sdk: YCloudML, user_text: str) -> str:
    model_name = os.getenv("YANDEX_GPT_MODEL", "yandexgpt")
    temperature = float(os.getenv("YANDEX_GPT_TEMPERATURE", "0.5"))

    messages = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": user_text},
    ]

    model = sdk.models.completions(model_name)
    operation = model.configure(temperature=temperature).run_deferred(messages)

    status = operation.get_status()
    while getattr(status, "is_running", False):
        time.sleep(2)
        status = operation.get_status()

    result = operation.get_result()

    # Try to extract text from various result shapes
    try:
        if hasattr(result, "alternatives"):
            alts = getattr(result, "alternatives")
            if alts:
                alt0 = alts[0]
                if hasattr(alt0, "text"):
                    return str(getattr(alt0, "text")).strip()
                if isinstance(alt0, dict):
                    text = alt0.get("text") or alt0.get("content")
                    if text:
                        return str(text).strip()
        if isinstance(result, dict):
            alts = result.get("alternatives") or result.get("choices")
            if isinstance(alts, list) and alts:
                alt0 = alts[0]
                if isinstance(alt0, dict):
                    text = alt0.get("text") or alt0.get("content")
                    if text:
                        return str(text).strip()
    except Exception:
        logger.warning("Failed to extract text from Yandex result; using string representation")

    return str(result).strip()


def main() -> None:
    # Logging setup
    log_level = os.getenv(
        "UNIBOT_LOG_LEVEL",
        os.getenv("YANDEX_BOT_LOG_LEVEL", os.getenv("GIGABOT_LOG_LEVEL", "INFO")),
    )
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment/.env")

    # Disable parse mode to avoid formatting issues; we clean output ourselves
    bot = telebot.TeleBot(tg_token, parse_mode=None)

    # Initialize clients (optional if creds missing)
    giga_client = build_gigachat_client()
    yandex_sdk = build_yandex_client()

    # Per-chat preferred model storage
    user_model: Dict[int, str] = {}

    def choose_default_model() -> Optional[str]:
        if giga_client is not None:
            return "gigachat"
        if yandex_sdk is not None:
            return "yandex"
        return None

    def is_available(model: str) -> bool:
        return (model == "gigachat" and giga_client is not None) or (
            model == "yandex" and yandex_sdk is not None
        )

    @bot.message_handler(commands=["start", "help"])  # type: ignore[misc]
    def handle_start(message: telebot.types.Message) -> None:
        default_model = choose_default_model()
        if default_model is None:
            bot.reply_to(
                message,
                "Привет! Я универсальный помощник. Настройте переменные окружения для GigaChat и/или YandexGPT и перезапустите бота.",
            )
            return
        user_model[message.chat.id] = default_model
        bot.reply_to(
            message,
            (
                "Привет! Я персональный помощник. Пишите вопрос.\n\n"
                "Доступные команды:\n"
                "/model — показать/изменить модель (например: /model gigachat или /model yandex)\n"
                "/gigachat — переключиться на GigaChat\n"
                "/yandex — переключиться на YandexGPT"
            ),
        )

    @bot.message_handler(commands=["model"])  # type: ignore[misc]
    def handle_model(message: telebot.types.Message) -> None:
        parts = (message.text or "").split(maxsplit=1)
        chat_id = message.chat.id
        if len(parts) == 2:
            choice = parts[1].strip().lower()
            alias = {"yandexgpt": "yandex", "y": "yandex", "g": "gigachat", "giga": "gigachat"}
            choice = alias.get(choice, choice)
            if choice in {"gigachat", "yandex"}:
                if is_available(choice):
                    user_model[chat_id] = choice
                    bot.reply_to(message, f"Модель переключена на: {choice}")
                else:
                    bot.reply_to(message, f"Модель '{choice}' недоступна. Проверьте токены и настройки.")
            else:
                bot.reply_to(message, "Неизвестная модель. Используйте 'gigachat' или 'yandex'.")
            return

        current = user_model.get(chat_id) or choose_default_model()
        availability = []
        availability.append("GigaChat: доступна" if giga_client else "GigaChat: недоступна")
        availability.append("YandexGPT: доступна" if yandex_sdk else "YandexGPT: недоступна")
        bot.reply_to(
            message,
            (
                f"Текущая модель: {current or '—'}\n" +
                "\n".join(availability) +
                "\n\nСменить: /model gigachat или /model yandex"
            ),
        )

    @bot.message_handler(commands=["gigachat"])  # type: ignore[misc]
    def handle_giga(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        if giga_client is None:
            bot.reply_to(message, "GigaChat недоступен. Проверьте переменные окружения.")
            return
        user_model[chat_id] = "gigachat"
        bot.reply_to(message, "Переключено на GigaChat.")

    @bot.message_handler(commands=["yandex", "yandexgpt"])  # type: ignore[misc]
    def handle_yandex(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        if yandex_sdk is None:
            bot.reply_to(message, "YandexGPT недоступен. Проверьте переменные окружения.")
            return
        user_model[chat_id] = "yandex"
        bot.reply_to(message, "Переключено на YandexGPT.")

    @bot.message_handler(content_types=["text"])  # type: ignore[misc]
    def handle_text(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        model = user_model.get(chat_id) or choose_default_model()
        if model is None:
            bot.send_message(chat_id, "Ни одна модель недоступна. Проверьте настройки и токены.")
            return

        user_text = message.text or ""
        try:
            if model == "gigachat":
                assert giga_client is not None
                reply = generate_with_gigachat(giga_client, user_text)
            else:
                assert yandex_sdk is not None
                reply = generate_with_yandex(yandex_sdk, user_text)
            clean_reply = _strip_markdown(reply)
            bot.send_message(chat_id, clean_reply, parse_mode=None)
        except Exception:
            logger.exception("Failed to generate reply using %s", model)
            bot.send_message(chat_id, "Произошла ошибка при обращении к модели. Попробуйте позже.")

    logger.info("Universal bot is running")
    bot.infinity_polling()


if __name__ == "__main__":
    main()


