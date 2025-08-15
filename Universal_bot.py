from __future__ import annotations

import os
import time
import logging
import re
from typing import Dict, Optional, Tuple, List, Any
import threading

import telebot
from dotenv import load_dotenv

from get_token import get_gigachat_token
from gigachat import GigaChat
from yandex_cloud_ml_sdk import YCloudML
import requests
import json


# Load .env if present
load_dotenv()

logger = logging.getLogger(__name__)

# File-based persistence
ALLOWED_USERS_FILE = "allowed_users.json"
USER_CONTEXTS_FILE = "user_contexts.json"

# Context window: keep last N pairs (user+assistant)
MAX_CONTEXT_LENGTH = 30


SYSTEM_PROMPT = "Ты вежливый и профессиональный личный помощник, работающий в Telegram."

# Model aliases normalization
def normalize_model_key(raw: str) -> str:
    key = (raw or "").strip().lower()
    alias = {
        "yandexgpt": "yandex",
        "y": "yandex",
        "g": "gigachat",
        "giga": "gigachat",
        "gemini": "gemini",
        "o4": "o4-mini",
        "o4mini": "o4-mini",
        "gpt5": "gpt-5",
    }
    return alias.get(key, key)


def get_model_display_name(model: str) -> str:
    names = {
        "gemini": "Gemini 2.5 Flash",
        "o4-mini": "o4-mini",
        "gpt-5": "GPT-5",
        "gigachat": "GigaChat",
        "yandex": "YandexGPT",
    }
    return names.get(model, model)


def get_model_emoji(model: str) -> str:
    # Original icons (from existing bot): only for 3 models we keep emojis
    icons = {
        "gemini": "🤖",
        "o4-mini": "🧠",
        "gpt-5": "🚀",
        "gigachat": "💹",
        "yandex": "✴️",
    }
    return icons.get(model, "")


def get_model_color_square(model: str) -> str:
    # Colored square used for visually separating pairs by model
    squares = {
        "gemini": "🟦",
        "o4-mini": "🟪",
        "gpt-5": "🟥",
        "gigachat": "🟩",
        "yandex": "🟧",
    }
    return squares.get(model, "⬜")


def escape_markdown(text: str) -> str:
    # Escape for Telegram Markdown (basic)
    return re.sub(r"([_\*\[\]\(\)`>#+\-=|{}.!])", r"\\\1", text or "")


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


def generate_with_gigachat(client: GigaChat, user_id: int, user_text: str) -> str:
    # Build conversational context as plain text transcript
    history_text = _context_as_transcript(user_id, model="gigachat")
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Контекст диалога (последние сообщения):\n{history_text}\n\n"
        f"Пользователь: {user_text.strip()}"
    )
    completion = client.chat(prompt)
    answer = None
    try:
        if completion and getattr(completion, "choices", None):
            answer = completion.choices[0].message.content
    except Exception:
        logger.warning("Unexpected GigaChat completion format")
    return (answer or "Извините, не удалось получить ответ.").strip()


def generate_with_yandex(sdk: YCloudML, user_id: int, user_text: str) -> str:
    model_name = os.getenv("YANDEX_GPT_MODEL", "yandexgpt")
    temperature = float(os.getenv("YANDEX_GPT_TEMPERATURE", "0.5"))

    # Convert global context to Yandex messages
    messages: List[Dict[str, str]] = [{"role": "system", "text": SYSTEM_PROMPT}]
    for msg in _get_context_for_model("yandex", user_id=user_id):
        if msg["role"] == "user":
            messages.append({"role": "user", "text": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "text": msg["content"]})
    messages.append({"role": "user", "text": user_text})

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


# ===== ProxyAPI models: Gemini, o4-mini, GPT-5 =====
def _get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("PROXYAPI_KEY")


def _get_context_messages(user_id: int, model: str, current_message: str, add_system: bool = True) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if add_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    if user_id in user_contexts and model in user_contexts[user_id]:
        messages.extend(user_contexts[user_id][model])
    messages.append({"role": "user", "content": current_message})
    return messages


def _context_as_transcript(user_id: Optional[int] = None, model: Optional[str] = None) -> str:
    """Return last messages as a readable transcript for plain-text models (GigaChat)."""
    messages = []
    if user_id is not None and user_id in user_contexts and model and model in user_contexts[user_id]:
        messages = user_contexts[user_id][model]
    lines: List[str] = []
    for m in messages:
        prefix = "Пользователь" if m["role"] == "user" else "Ассистент"
        lines.append(f"{prefix}: {m['content']}")
    return "\n".join(lines[-MAX_CONTEXT_LENGTH * 2 :])


def _get_context_for_model(model_key: str, user_id: Optional[int] = None) -> List[Dict[str, str]]:
    if user_id is not None and user_id in user_contexts and model_key in user_contexts[user_id]:
        return user_contexts[user_id][model_key][-MAX_CONTEXT_LENGTH * 2 :]
    return []


def render_context_history(user_id: int, model: str, limit_pairs: int = 10) -> str:
    context = user_contexts.get(user_id, {}).get(model, [])
    if not context:
        return "Контекст пуст"
    # Take last N pairs (2 messages per pair)
    messages = context[-2 * max(1, limit_pairs) :]
    lines: List[str] = []
    pair_index = 1
    i = 0
    def _squash_empty_lines(text: str) -> str:
        parts = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        return " ".join(parts)
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "user":
            user_text = _squash_empty_lines(msg.get('content', ''))
            lines.append(f"{pair_index}. 🙍🏻‍♂️: {user_text}")
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                assistant_text = _squash_empty_lines(messages[i + 1].get('content', ''))
                lines.append(f"{pair_index}. 💢: {assistant_text}")
                i += 2
            else:
                i += 1
            pair_index += 1
            lines.append("")  # blank line between pairs
        else:
            # orphan assistant message
            assistant_text = _squash_empty_lines(msg.get('content', ''))
            lines.append(f"{pair_index}. 💢: {assistant_text}")
            lines.append("")
            i += 1
    # We keep **bold** markers for visual emphasis, then strip conflicting md in content
    plain = "\n".join(lines)
    shown_pairs = min(limit_pairs, len(context) // 2)
    header = f"История для {get_model_display_name(model)} (последние {shown_pairs} пар):"
    return header + "\n\n" + plain


def split_message(text: str, max_length: int = 4096) -> List[str]:
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def generate_with_gemini(user_id: int, message: str) -> str:
    try:
        url = "https://api.proxyapi.ru/google/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": _get_openai_api_key()}

        # Build context in Gemini format
        contents: List[Dict[str, Any]] = []

        context_messages = _get_context_messages(user_id, "gemini", message, add_system=False)
        for msg in context_messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        data = {
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048, "topP": 0.8, "topK": 40},
        }

        response = requests.post(url, headers=headers, params=params, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        logger.exception("Gemini request failed")
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"


def generate_with_o4_mini(user_id: int, message: str) -> str:
    try:
        url = "https://api.proxyapi.ru/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {_get_openai_api_key()}", "Content-Type": "application/json"}
        messages = _get_context_messages(user_id, "o4-mini", message, add_system=True)
        data = {"model": "o4-mini", "messages": messages}
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        logger.exception("o4-mini request failed")
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"


def generate_with_gpt5(user_id: int, message: str) -> str:
    try:
        url = "https://api.proxyapi.ru/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {_get_openai_api_key()}", "Content-Type": "application/json"}
        messages = _get_context_messages(user_id, "gpt-5", message, add_system=True)
        data = {"model": "gpt-5", "messages": messages, "max_completion_tokens": 2048}
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        logger.exception("GPT-5 request failed")
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"


# ===== Authorization and persistence (reused from bot.py) =====
user_models: Dict[int, str] = {}
user_contexts: Dict[int, Dict[str, List[Dict[str, str]]]] = {}
thinking_messages: Dict[int, Optional[int]] = {}


def load_allowed_users() -> Tuple[Optional[int], set]:
    try:
        if os.path.exists(ALLOWED_USERS_FILE):
            with open(ALLOWED_USERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("admin_user_id"), set(data.get("allowed_users", []))
        else:
            save_allowed_users(None, [])
            return None, set()
    except Exception as e:
        logger.error(f"Ошибка при загрузке списка пользователей: {e}")
        return None, set()


def save_allowed_users(admin_id: Optional[int], allowed_users_list: List[int]) -> None:
    try:
        data = {
            "admin_user_id": admin_id,
            "allowed_users": list(allowed_users_list),
            "description": "Файл с разрешенными пользователями для бота",
            "instructions": {
                "admin_user_id": "ID администратора бота",
                "allowed_users": "Список ID пользователей, которым разрешен доступ к боту",
                "format": "Добавляйте ID пользователей как числа в массив allowed_users",
            },
        }
        with open(ALLOWED_USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении списка пользователей: {e}")


def is_user_allowed(user_id: int) -> Tuple[bool, bool]:
    admin_id, allowed_users = load_allowed_users()
    if admin_id is None:
        admin_id = user_id
        allowed_users.add(user_id)
        save_allowed_users(admin_id, list(allowed_users))
        return True, True
    if user_id == admin_id:
        return True, True
    if user_id in allowed_users:
        return True, False
    return False, False


def add_user(user_id: int) -> None:
    admin_id, allowed_users = load_allowed_users()
    allowed_users.add(user_id)
    save_allowed_users(admin_id, list(allowed_users))


def remove_user(user_id: int) -> None:
    admin_id, allowed_users = load_allowed_users()
    allowed_users.discard(user_id)
    save_allowed_users(admin_id, list(allowed_users))


def get_users_info() -> Tuple[Optional[int], List[int]]:
    admin_id, allowed_users = load_allowed_users()
    return admin_id, list(allowed_users)


def save_user_contexts() -> None:
    try:
        contexts_to_save: Dict[str, Any] = {str(uid): ctx for uid, ctx in user_contexts.items()}
        with open(USER_CONTEXTS_FILE, "w", encoding="utf-8") as f:
            json.dump(contexts_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении контекстов: {e}")


def load_user_contexts() -> None:
    global user_contexts
    try:
        if os.path.exists(USER_CONTEXTS_FILE):
            with open(USER_CONTEXTS_FILE, "r", encoding="utf-8") as f:
                contexts_data = json.load(f)
            # Migrate possible old schema (list) -> new schema (dict per model)
            migrated: Dict[int, Dict[str, List[Dict[str, str]]]] = {}
            for uid_str, ctx in contexts_data.items():
                uid = int(uid_str)
                if isinstance(ctx, list):
                    migrated[uid] = {"gemini": ctx}
                elif isinstance(ctx, dict):
                    migrated[uid] = ctx
                else:
                    migrated[uid] = {}
            user_contexts = migrated
            # Persist migration if changed
            try:
                save_user_contexts()
            except Exception:
                pass
        else:
            user_contexts = {}
    except Exception as e:
        logger.error(f"Ошибка при загрузке контекстов: {e}")
        user_contexts = {}


def add_to_context(user_id: int, model: str, user_message: str, ai_response: str) -> None:
    if user_id not in user_contexts:
        user_contexts[user_id] = {}
    if model not in user_contexts[user_id]:
        user_contexts[user_id][model] = []
    ctx = user_contexts[user_id][model]
    ctx.append({"role": "user", "content": user_message})
    ctx.append({"role": "assistant", "content": ai_response})
    max_messages = MAX_CONTEXT_LENGTH * 2
    if len(ctx) > max_messages:
        user_contexts[user_id][model] = ctx[-max_messages:]
    save_user_contexts()


def get_context_info(user_id: int, model: Optional[str] = None) -> str:
    if user_id not in user_contexts or not user_contexts[user_id]:
        return "Контекст пуст"
    if model:
        ctx = user_contexts[user_id].get(model, [])
        message_count = len(ctx)
        pair_count = message_count // 2
        return f"Модель: {model}. В контексте {pair_count} пар ({message_count} сообщений). Максимум: {MAX_CONTEXT_LENGTH} пар."
    parts: List[str] = []
    for m, ctx in user_contexts[user_id].items():
        message_count = len(ctx)
        pair_count = message_count // 2
        parts.append(f"- {m}: {pair_count} пар ({message_count})")
    if not parts:
        return "Контекст пуст"
    return "Контекст по моделям:\n" + "\n".join(parts)


def clear_context(user_id: int, model: Optional[str] = None) -> None:
    if user_id not in user_contexts:
        return
    if model:
        user_contexts[user_id].pop(model, None)
        if not user_contexts[user_id]:
            user_contexts.pop(user_id, None)
    else:
        user_contexts.pop(user_id, None)
    save_user_contexts()


def send_thinking_message(bot: telebot.TeleBot, chat_id: int, model_name: str, delay: int = 5) -> None:
    def delayed_send() -> None:
        time.sleep(delay)
        if chat_id in thinking_messages and thinking_messages[chat_id] is None:
            try:
                names = {
                    "gemini": "Gemini 2.5 Flash",
                    "o4-mini": "o4-mini",
                    "gpt-5": "GPT-5",
                    "gigachat": "GigaChat",
                    "yandex": "YandexGPT",
                }
                msg = bot.send_message(chat_id, f"Модель {names.get(model_name, model_name)} думает над ответом...")
                thinking_messages[chat_id] = msg.message_id
            except Exception:
                logger.exception("Ошибка при отправке сообщения о размышлении")

    thinking_messages[chat_id] = None
    t = threading.Thread(target=delayed_send)
    t.daemon = True
    t.start()


def clear_thinking_message(bot: telebot.TeleBot, chat_id: int) -> None:
    if chat_id in thinking_messages:
        message_id = thinking_messages[chat_id]
        if message_id is not None:
            try:
                bot.delete_message(chat_id, message_id)
            except Exception:
                logger.warning("Не удалось удалить сообщение о размышлении")
        del thinking_messages[chat_id]


def main() -> None:
    # Logging setup (file + console)
    log_level = os.getenv("UNIBOT_LOG_LEVEL", os.getenv("YANDEX_BOT_LOG_LEVEL", os.getenv("GIGABOT_LOG_LEVEL", "INFO")))
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not tg_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment/.env")

    # Disable parse mode to avoid formatting issues; we clean output ourselves
    bot = telebot.TeleBot(tg_token, parse_mode=None)

    # Initialize clients (optional if creds missing)
    giga_client = build_gigachat_client()
    yandex_sdk = build_yandex_client()

    # Load persisted contexts
    load_user_contexts()

    def choose_default_model() -> str:
        return "gemini"

    def is_available(model: str) -> bool:
        if model in {"gemini", "o4-mini", "gpt-5"}:
            return _get_openai_api_key() is not None
        if model == "gigachat":
            return giga_client is not None
        if model == "yandex":
            return yandex_sdk is not None
        return False

    def check_access_decorator(func):
        def wrapper(message):
            user_id = message.from_user.id
            allowed, is_admin = is_user_allowed(user_id)
            if not allowed:
                bot.reply_to(message, "Доступ запрещен. Обратитесь к администратору для получения доступа.")
                return
            message.is_admin = is_admin  # type: ignore[attr-defined]
            return func(message)
        return wrapper

    def _build_main_menu_kb() -> telebot.types.InlineKeyboardMarkup:
        kb = telebot.types.InlineKeyboardMarkup(row_width=1)
        kb.add(
            telebot.types.InlineKeyboardButton("Выбрать модель", callback_data="menu_model"),
            telebot.types.InlineKeyboardButton("Текущая модель", callback_data="menu_current"),
            telebot.types.InlineKeyboardButton("Контекст", callback_data="menu_context"),
        )
        return kb

    def _build_menu_button_kb() -> telebot.types.InlineKeyboardMarkup:
        kb = telebot.types.InlineKeyboardMarkup(row_width=1)
        kb.add(telebot.types.InlineKeyboardButton("Меню", callback_data="show_menu"))
        return kb

    def _send_menu_button(chat_id: int) -> None:
        try:
            bot.send_message(chat_id, "", reply_markup=_build_menu_button_kb(), parse_mode=None)
        except Exception:
            # Fallback text if empty not allowed
            bot.send_message(chat_id, "Открыть меню:", reply_markup=_build_menu_button_kb(), parse_mode=None)

    @bot.message_handler(commands=["start", "help"])  # type: ignore[misc]
    @check_access_decorator
    def handle_start(message: telebot.types.Message) -> None:
        user_models[message.chat.id] = choose_default_model()
        welcome = (
            "Привет! Я персональный помощник. Задайте вопрос.\n\n"
            "Выберите действие ниже."
        )
        bot.reply_to(message, _strip_markdown(welcome), reply_markup=_build_main_menu_kb(), parse_mode=None)

    def _create_model_keyboard() -> telebot.types.InlineKeyboardMarkup:
        kb = telebot.types.InlineKeyboardMarkup(row_width=2)
        models = ["gemini", "o4-mini", "gpt-5", "gigachat", "yandex"]
        buttons = []
        for m in models:
            icon = get_model_emoji(m)
            label = f"{icon + ' ' if icon else ''}{get_model_display_name(m)}"
            buttons.append(telebot.types.InlineKeyboardButton(label, callback_data=f"model_{m}"))
        kb.add(*buttons)
        return kb

    @bot.message_handler(commands=["model"])  # type: ignore[misc]
    @check_access_decorator
    def handle_model(message: telebot.types.Message) -> None:
        parts = (message.text or "").split(maxsplit=1)
        chat_id = message.chat.id
        if len(parts) == 2:
            choice = parts[1].strip().lower()
            alias = {"yandexgpt": "yandex", "y": "yandex", "g": "gigachat", "giga": "gigachat", "gemini": "gemini", "o4": "o4-mini"}
            choice = alias.get(choice, choice)
            if choice in {"gemini", "o4-mini", "gpt-5", "gigachat", "yandex"}:
                if is_available(choice):
                    user_models[chat_id] = choice
                    bot.reply_to(message, f"Модель переключена на: {choice}")
                else:
                    bot.reply_to(message, f"Модель '{choice}' недоступна. Проверьте токены и настройки.")
            else:
                bot.reply_to(message, "Неизвестная модель. Используйте: gemini, o4-mini, gpt-5, gigachat, yandex")
            return

        current = user_models.get(chat_id) or choose_default_model()
        availability = []
        availability.append("Gemini: доступна" if is_available("gemini") else "Gemini: недоступна")
        availability.append("o4-mini: доступна" if is_available("o4-mini") else "o4-mini: недоступна")
        availability.append("GPT-5: доступна" if is_available("gpt-5") else "GPT-5: недоступна")
        availability.append("GigaChat: доступна" if giga_client else "GigaChat: недоступна")
        availability.append("YandexGPT: доступна" if yandex_sdk else "YandexGPT: недоступна")
        bot.reply_to(
            message,
            (
                f"Текущая модель: {current or '—'}\n" +
                "\n".join(availability) +
                "\n\nСменить: /model <gemini|o4-mini|gpt-5|gigachat|yandex>"
            ),
            reply_markup=_create_model_keyboard(),
        )

    @bot.callback_query_handler(func=lambda call: call.data.startswith("model_"))  # type: ignore[misc]
    def handle_model_selection(call):
        allowed, _ = is_user_allowed(call.from_user.id)
        if not allowed:
            bot.answer_callback_query(call.id, "Доступ запрещен")
            return
        model = call.data.replace("model_", "")
        if not is_available(model):
            bot.answer_callback_query(call.id, "Модель недоступна")
            return
        user_models[call.from_user.id] = model
        bot.answer_callback_query(call.id, f"Модель переключена на {model}")
        try:
            bot.edit_message_text(f"Выбрана модель: {model}", call.message.chat.id, call.message.message_id)
        except Exception:
            pass

    @bot.callback_query_handler(func=lambda call: call.data == "show_menu")
    def handle_show_menu(call):
        try:
            bot.edit_message_text("Меню:", call.message.chat.id, call.message.message_id, reply_markup=_build_main_menu_kb(), parse_mode=None)
        except Exception:
            # If edit failed (e.g., too old), send a new one
            bot.send_message(call.message.chat.id, "Меню:", reply_markup=_build_main_menu_kb(), parse_mode=None)
        bot.answer_callback_query(call.id)

    @bot.callback_query_handler(func=lambda call: call.data == "menu_model")
    def handle_menu_model(call):
        # Reuse model view
        chat_id = call.message.chat.id
        current = user_models.get(chat_id) or choose_default_model()
        availability = []
        availability.append("Gemini: доступна" if is_available("gemini") else "Gemini: недоступна")
        availability.append("o4-mini: доступна" if is_available("o4-mini") else "o4-mini: недоступна")
        availability.append("GPT-5: доступна" if is_available("gpt-5") else "GPT-5: недоступна")
        availability.append("GigaChat: доступна" if giga_client else "GigaChat: недоступна")
        availability.append("YandexGPT: доступна" if yandex_sdk else "YandexGPT: недоступна")
        text = (
            f"Текущая модель: {current or '—'}\n" +
            "\n".join(availability)
        )
        bot.send_message(chat_id, _strip_markdown(text), reply_markup=_create_model_keyboard(), parse_mode=None)
        bot.answer_callback_query(call.id)

    @bot.callback_query_handler(func=lambda call: call.data == "menu_current")
    def handle_menu_current(call):
        chat_id = call.message.chat.id
        current = user_models.get(chat_id, choose_default_model())
        bot.send_message(chat_id, f"Текущая модель: {current}", parse_mode=None)
        bot.answer_callback_query(call.id)

    @bot.callback_query_handler(func=lambda call: call.data == "menu_context")
    def handle_menu_context(call):
        # Reuse context list UI
        chat_id = call.message.chat.id
        kb = telebot.types.InlineKeyboardMarkup(row_width=2)
        models = ["gemini", "o4-mini", "gpt-5", "gigachat", "yandex"]
        buttons = []
        for m in models:
            count_pairs = len(user_contexts.get(chat_id, {}).get(m, [])) // 2
            icon = get_model_emoji(m)
            display = get_model_display_name(m)
            label = f"{icon + ' ' if icon else ''}{display}: {count_pairs}"
            buttons.append(telebot.types.InlineKeyboardButton(label, callback_data=f"ctxshow_{m}_all"))
        kb.add(*buttons)
        current = user_models.get(chat_id, "gemini")
        header = (
            f"Просмотр контекста. Текущая модель: {get_model_emoji(current)} {get_model_display_name(current)}\n"
            "Выберите модель для показа."
        )
        bot.send_message(chat_id, _strip_markdown(header), reply_markup=kb, parse_mode=None)
        bot.answer_callback_query(call.id)

    @bot.message_handler(commands=["current"])  # type: ignore[misc]
    @check_access_decorator
    def handle_current(message: telebot.types.Message) -> None:
        current = user_models.get(message.chat.id, choose_default_model())
        bot.reply_to(message, f"Текущая модель: {current}")
        _send_menu_button(message.chat.id)

    @bot.message_handler(commands=["context"])  # type: ignore[misc]
    @check_access_decorator
    def handle_context(message: telebot.types.Message) -> None:
        # Build inline buttons showing models (five buttons, no color marks)
        chat_id = message.chat.id
        kb = telebot.types.InlineKeyboardMarkup(row_width=2)
        models = ["gemini", "o4-mini", "gpt-5", "gigachat", "yandex"]
        buttons = []
        for m in models:
            count_pairs = len(user_contexts.get(chat_id, {}).get(m, [])) // 2
            icon = get_model_emoji(m)
            display = get_model_display_name(m)
            label = f"{icon + ' ' if icon else ''}{display}: {count_pairs}"
            buttons.append(telebot.types.InlineKeyboardButton(label, callback_data=f"ctxshow_{m}_all"))
        kb.add(*buttons)

        current = user_models.get(chat_id, "gemini")
        header = (
            f"Просмотр контекста. Текущая модель: {get_model_emoji(current)} {get_model_display_name(current)}\n"
            "Выберите модель для показа (по умолчанию 10 пар)."
        )
        bot.reply_to(message, _strip_markdown(header), reply_markup=kb, parse_mode=None)

    @bot.callback_query_handler(func=lambda call: call.data.startswith("ctxshow_"))
    def handle_context_show(call):
        allowed, _ = is_user_allowed(call.from_user.id)
        if not allowed:
            bot.answer_callback_query(call.id, "Доступ запрещен")
            return
        # ctxshow_<model>_all
        try:
            _, model, _ = call.data.split("_", 2)
            limit_pairs = 10**9  # effectively all
        except Exception:
            model = "gemini"
            limit_pairs = 10
        chat_id = call.message.chat.id
        text = render_context_history(chat_id, model, limit_pairs)
        # Add a header with a neutral context indicator
        header = f"📚 История ({get_model_emoji(model)} {get_model_display_name(model)})\n"
        full_text = header + "\n" + text
        # Inline keyboard: clear context for this model (only if not empty)
        ctx_list = user_contexts.get(chat_id, {}).get(model, [])
        has_context = bool(ctx_list)
        kb = None
        if has_context:
            kb = telebot.types.InlineKeyboardMarkup(row_width=1)
            kb.add(telebot.types.InlineKeyboardButton(f"❌Очистить контекст {get_model_display_name(model)}?", callback_data=f"ctxclear_{model}"))
        # Render with basic Markdown; attach keyboard to first chunk
        chunks = split_message(full_text)
        for idx, chunk in enumerate(chunks):
            if idx == 0 and kb is not None:
                bot.send_message(chat_id, chunk, parse_mode='Markdown', reply_markup=kb)
            else:
                bot.send_message(chat_id, chunk, parse_mode='Markdown')
        bot.answer_callback_query(call.id)

    @bot.callback_query_handler(func=lambda call: call.data.startswith("ctxclear_"))
    def handle_context_clear(call):
        allowed, _ = is_user_allowed(call.from_user.id)
        if not allowed:
            bot.answer_callback_query(call.id, "Доступ запрещен")
            return
        try:
            _, model = call.data.split("_", 1)
        except Exception:
            model = "gemini"
        chat_id = call.message.chat.id
        clear_context(chat_id, model)
        bot.answer_callback_query(call.id, "Контекст очищен")
        bot.send_message(chat_id, f"Контекст модели {get_model_display_name(model)} очищен.")

    # Removed /clear command as per request

    @bot.message_handler(commands=["gigachat"])  # type: ignore[misc]
    @check_access_decorator
    def handle_giga(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        if giga_client is None:
            bot.reply_to(message, "GigaChat недоступен. Проверьте переменные окружения.")
            return
        user_models[chat_id] = "gigachat"
        bot.reply_to(message, "Переключено на GigaChat.")

    @bot.message_handler(commands=["yandex", "yandexgpt"])  # type: ignore[misc]
    @check_access_decorator
    def handle_yandex(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        if yandex_sdk is None:
            bot.reply_to(message, "YandexGPT недоступен. Проверьте переменные окружения.")
            return
        user_models[chat_id] = "yandex"
        bot.reply_to(message, "Переключено на YandexGPT.")

    @bot.message_handler(content_types=["text"])  # type: ignore[misc]
    @check_access_decorator
    def handle_text(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        # Skip commands
        if (message.text or "").startswith("/"):
            return
        model = user_models.get(chat_id) or choose_default_model()
        if model is None:
            bot.send_message(chat_id, "Ни одна модель недоступна. Проверьте настройки и токены.")
            return

        user_text = message.text or ""
        try:
            # Thinking indicator
            try:
                send_thinking_message(bot, chat_id, model)
            except Exception:
                pass

            if model == "gemini":
                reply = generate_with_gemini(chat_id, user_text)
            elif model == "o4-mini":
                reply = generate_with_o4_mini(chat_id, user_text)
            elif model == "gpt-5":
                reply = generate_with_gpt5(chat_id, user_text)
            elif model == "gigachat":
                assert giga_client is not None
                reply = generate_with_gigachat(giga_client, chat_id, user_text)
            else:  # yandex
                assert yandex_sdk is not None
                reply = generate_with_yandex(yandex_sdk, chat_id, user_text)

            # Update context for selected model
            add_to_context(chat_id, model, user_text, reply)

            # Clean markdown/HTML before sending, always plain text
            clean_reply = _strip_markdown(reply)
            # Clear thinking message
            try:
                clear_thinking_message(bot, chat_id)
            except Exception:
                pass
            for part in split_message(clean_reply):
                bot.send_message(chat_id, part, parse_mode=None)
        except Exception:
            logger.exception("Failed to generate reply using %s", model)
            bot.send_message(chat_id, "Произошла ошибка при обращении к модели. Попробуйте позже.")

    logger.info("Universal bot is running")
    bot.infinity_polling()


if __name__ == "__main__":
    main()


