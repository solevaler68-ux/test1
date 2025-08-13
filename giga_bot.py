import os
import logging
import re
from typing import Optional

import telebot
from dotenv import load_dotenv

from get_token import get_gigachat_token
from gigachat import GigaChat


load_dotenv()

logger = logging.getLogger(__name__)


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_gigachat_client() -> GigaChat:
    token_obj = get_gigachat_token()
    # Клиенту GigaChat передаём credentials (Authorization Data),
    # но библиотека поддерживает и получение токена по get_token() автоматически.
    # Для последовательности используем тот же механизм, что и в get_token.py
    credentials = None  # пусть клиент сам получит токен через credentials

    # SSL-настройки из .env (совпадают с get_token.py)
    verify_ssl_certs: Optional[bool] = None
    env_verify = os.getenv("GIGACHAT_VERIFY_SSL")
    if env_verify is not None:
        verify_ssl_certs = env_verify.strip().lower() in {"1", "true", "yes", "on"}
    ca_bundle_file = os.getenv("GIGACHAT_CA_BUNDLE_FILE")

    return GigaChat(
        credentials=credentials or os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGACHAT_AUTHORIZATION"),
        verify_ssl_certs=verify_ssl_certs,
        ca_bundle_file=ca_bundle_file,
        access_token=token_obj.access_token,  # ускоряет первый запрос
    )


def _strip_markdown(text: str) -> str:
    """Remove Markdown/MarkdownV2/HTML formatting and return plain text.

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


def main() -> None:
    # Логи
    log_level = os.getenv("GIGABOT_LOG_LEVEL", os.getenv("GIGACHAT_LOG_LEVEL", "INFO"))
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Telegram токен
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment/.env")

    # Disable formatting to avoid Telegram entity parsing errors
    bot = telebot.TeleBot(tg_token, parse_mode=None)

    # Инициализируем GigaChat-клиент один раз
    giga_client = build_gigachat_client()

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: telebot.types.Message) -> None:
        bot.reply_to(message, "Привет! Я бот-помощник на базе GigaChat. Просто напишите вопрос.")

    @bot.message_handler(content_types=["text"])
    def handle_text(message: telebot.types.Message) -> None:
        user_text = message.text or ""
        chat_id = message.chat.id
        try:
            # Вызов GigaChat
            completion = giga_client.chat(user_text)
            answer = completion.choices[0].message.content if completion and completion.choices else "Извините, не удалось получить ответ."
            plain_answer = _strip_markdown(answer)
            bot.send_message(chat_id, plain_answer, parse_mode=None)
        except Exception:
            logger.exception("Failed to get response from GigaChat")
            bot.send_message(chat_id, "Произошла ошибка при обращении к GigaChat. Попробуйте позже.")

    logger.info("Bot is running")
    bot.infinity_polling()


if __name__ == "__main__":
    main()


