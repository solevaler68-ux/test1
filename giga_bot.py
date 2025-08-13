import os
import logging
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


def main() -> None:
    # Логи
    log_level = os.getenv("GIGABOT_LOG_LEVEL", os.getenv("GIGACHAT_LOG_LEVEL", "INFO"))
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Telegram токен
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment/.env")

    bot = telebot.TeleBot(tg_token, parse_mode="HTML")

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
            bot.send_message(chat_id, answer)
        except Exception:
            logger.exception("Failed to get response from GigaChat")
            bot.send_message(chat_id, "Произошла ошибка при обращении к GigaChat. Попробуйте позже.")

    logger.info("Bot is running")
    bot.infinity_polling()


if __name__ == "__main__":
    main()


