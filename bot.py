import os
import telebot
import requests
import logging
import threading
import time
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize Telegram bot
bot = telebot.TeleBot(os.getenv('TELEGRAM_TOKEN'))

# User model preferences storage
user_models = {}

# Storage for thinking messages
thinking_messages = {}

# Storage for conversation context (user_id -> list of messages)
user_contexts = {}

# Maximum context length (number of message pairs to keep)
MAX_CONTEXT_LENGTH = 10

# Authorization settings
ALLOWED_USERS_FILE = 'allowed_users.json'

# Context persistence settings
USER_CONTEXTS_FILE = 'user_contexts.json'

def load_allowed_users():
    """
    Load allowed users from JSON file
    """
    try:
        if os.path.exists(ALLOWED_USERS_FILE):
            with open(ALLOWED_USERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('admin_user_id'), set(data.get('allowed_users', []))
        else:
            # Create empty file if it doesn't exist
            save_allowed_users(None, [])
            return None, set()
    except Exception as e:
        logging.error(f"Ошибка при загрузке списка пользователей: {e}")
        return None, set()

def save_allowed_users(admin_id, allowed_users_list):
    """
    Save allowed users to JSON file
    """
    try:
        data = {
            "admin_user_id": admin_id,
            "allowed_users": list(allowed_users_list),
            "description": "Файл с разрешенными пользователями для бота",
            "instructions": {
                "admin_user_id": "ID администратора бота",
                "allowed_users": "Список ID пользователей, которым разрешен доступ к боту",
                "format": "Добавляйте ID пользователей как числа в массив allowed_users"
            }
        }
        
        with open(ALLOWED_USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Список пользователей сохранен. Админ: {admin_id}, Пользователей: {len(allowed_users_list)}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении списка пользователей: {e}")

def is_user_allowed(user_id):
    """
    Check if user is allowed to use the bot
    """
    admin_id, allowed_users = load_allowed_users()
    
    # If no admin set, first user becomes admin
    if admin_id is None:
        admin_id = user_id
        allowed_users.add(user_id)
        save_allowed_users(admin_id, allowed_users)
        logging.info(f"Первый пользователь {user_id} установлен как администратор")
        return True, True  # allowed, is_admin
    
    # Check if user is admin
    if user_id == admin_id:
        return True, True  # allowed, is_admin
    
    # Check if user is in allowed list
    if user_id in allowed_users:
        return True, False  # allowed, not_admin
    
    return False, False  # not_allowed, not_admin

def add_user(user_id):
    """
    Add user to allowed list
    """
    admin_id, allowed_users = load_allowed_users()
    allowed_users.add(user_id)
    save_allowed_users(admin_id, allowed_users)

def remove_user(user_id):
    """
    Remove user from allowed list
    """
    admin_id, allowed_users = load_allowed_users()
    allowed_users.discard(user_id)
    save_allowed_users(admin_id, allowed_users)

def get_users_info():
    """
    Get information about users
    """
    admin_id, allowed_users = load_allowed_users()
    return admin_id, list(allowed_users)

def save_user_contexts():
    """
    Save user contexts to JSON file
    """
    try:
        # Convert set keys to strings for JSON serialization
        contexts_to_save = {}
        for user_id, context in user_contexts.items():
            contexts_to_save[str(user_id)] = context
        
        with open(USER_CONTEXTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contexts_to_save, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Контексты сохранены для {len(contexts_to_save)} пользователей")
    except Exception as e:
        logging.error(f"Ошибка при сохранении контекстов: {e}")

def load_user_contexts():
    """
    Load user contexts from JSON file
    """
    global user_contexts
    try:
        if os.path.exists(USER_CONTEXTS_FILE):
            with open(USER_CONTEXTS_FILE, 'r', encoding='utf-8') as f:
                contexts_data = json.load(f)
            
            # Convert string keys back to integers
            user_contexts = {}
            for user_id_str, context in contexts_data.items():
                user_contexts[int(user_id_str)] = context
            
            logging.info(f"Контексты загружены для {len(user_contexts)} пользователей")
        else:
            logging.info("Файл контекстов не найден, начинаем с пустого состояния")
            user_contexts = {}
    except Exception as e:
        logging.error(f"Ошибка при загрузке контекстов: {e}")
        user_contexts = {}

def check_access(func):
    """
    Decorator to check user access before executing command
    """
    def wrapper(message):
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        
        allowed, is_admin = is_user_allowed(user_id)
        
        if not allowed:
            logging.warning(f"Unauthorized access attempt from user {username} (ID: {user_id})")
            bot.reply_to(message, 
                "🚫 **Доступ запрещен**\n\nВы не авторизованы для использования этого бота.\nОбратитесь к администратору для получения доступа.", 
                parse_mode='Markdown')
            return
        
        # Add admin status to message object for admin commands
        message.is_admin = is_admin
        return func(message)
    
    return wrapper

def send_thinking_message(chat_id, model_name, delay=5):
    """
    Send a thinking message after delay if no response received
    """
    def delayed_send():
        time.sleep(delay)
        # Check if we still need to show thinking message
        if chat_id in thinking_messages and thinking_messages[chat_id] is None:
            try:
                model_display_names = {
                    "gemini": "Gemini 2.5 Flash",
                    "o4-mini": "o4-mini",
                    "gpt-5": "GPT-5"
                }
                thinking_text = f"🤔 {model_display_names.get(model_name, model_name)} думает над ответом..."
                message = bot.send_message(chat_id, thinking_text)
                thinking_messages[chat_id] = message.message_id
                logging.info(f"Отправлено сообщение о размышлении для модели {model_name}")
            except Exception as e:
                logging.error(f"Ошибка при отправке сообщения о размышлении: {e}")
    
    # Mark that we're waiting for response
    thinking_messages[chat_id] = None
    
    # Start timer in separate thread
    timer_thread = threading.Thread(target=delayed_send)
    timer_thread.daemon = True
    timer_thread.start()

def clear_thinking_message(chat_id):
    """
    Clear thinking message if it exists
    """
    if chat_id in thinking_messages:
        message_id = thinking_messages[chat_id]
        if message_id is not None:
            try:
                bot.delete_message(chat_id, message_id)
                logging.info(f"Удалено сообщение о размышлении")
            except Exception as e:
                logging.warning(f"Не удалось удалить сообщение о размышлении: {e}")
        
        # Remove from tracking
        del thinking_messages[chat_id]

def add_to_context(user_id: int, user_message: str, ai_response: str):
    """
    Add user message and AI response to conversation context
    """
    if user_id not in user_contexts:
        user_contexts[user_id] = []
    
    # Add user message and AI response
    user_contexts[user_id].append({"role": "user", "content": user_message})
    user_contexts[user_id].append({"role": "assistant", "content": ai_response})
    
    # Keep only last MAX_CONTEXT_LENGTH pairs (user + assistant messages)
    max_messages = MAX_CONTEXT_LENGTH * 2  # Each pair = 2 messages
    if len(user_contexts[user_id]) > max_messages:
        user_contexts[user_id] = user_contexts[user_id][-max_messages:]
    
    logging.info(f"Контекст пользователя {user_id} обновлен. Сообщений в контексте: {len(user_contexts[user_id])}")
    
    # Save contexts to file after each update
    save_user_contexts()

def get_context_messages(user_id: int, current_message: str) -> list:
    """
    Get conversation context for user with current message
    """
    messages = []
    
    # Add conversation history
    if user_id in user_contexts:
        messages.extend(user_contexts[user_id])
    
    # Add current user message
    messages.append({"role": "user", "content": current_message})
    
    return messages

def clear_context(user_id: int):
    """
    Clear conversation context for user
    """
    if user_id in user_contexts:
        del user_contexts[user_id]
        logging.info(f"Контекст пользователя {user_id} очищен")
        # Save contexts to file after clearing
        save_user_contexts()

def get_context_info(user_id: int) -> str:
    """
    Get information about current context
    """
    if user_id not in user_contexts or not user_contexts[user_id]:
        return "Контекст пуст"
    
    message_count = len(user_contexts[user_id])
    pair_count = message_count // 2
    return f"В контексте {pair_count} пар сообщений ({message_count} сообщений)"

def get_gemini_response(user_id: int, message: str) -> str:
    """
    Get response from Gemini 2.5 Flash with context
    """
    try:
        url = "https://api.proxyapi.ru/google/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        params = {
            "key": os.getenv('OPENAI_API_KEY')
        }
        
        # Get conversation context
        context_messages = get_context_messages(user_id, message)
        
        # Convert to Gemini format
        contents = []
        for msg in context_messages:
            role = "user" if msg["role"] == "user" else "model"  # Gemini uses "model" instead of "assistant"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        logging.info(f"Gemini: Отправляем {len(contents)} сообщений в контексте")
        
        response = requests.post(url, headers=headers, params=params, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"

def get_o4_mini_response(user_id: int, message: str) -> str:
    """
    Get response from o4-mini with context
    """
    try:
        url = "https://api.proxyapi.ru/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Get conversation context
        context_messages = get_context_messages(user_id, message)
        
        data = {
            "model": "o4-mini",
            "messages": context_messages
        }
        
        logging.info(f"o4-mini: Отправляем {len(context_messages)} сообщений в контексте")
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"

def get_gpt5_response(user_id: int, message: str) -> str:
    """
    Get response from GPT-5 with context
    """
    try:
        logging.info(f"GPT-5: Начинаем обработку запроса: {message[:50]}...")
        
        url = "https://api.proxyapi.ru/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Get conversation context
        context_messages = get_context_messages(user_id, message)
        
        data = {
            "model": "gpt-5",
            "messages": context_messages,
            "max_completion_tokens": 2048
        }
        
        logging.info(f"GPT-5: Отправляем запрос к {url}")
        logging.info(f"GPT-5: Отправляем {len(context_messages)} сообщений в контексте")
        
        response = requests.post(url, headers=headers, json=data)
        
        logging.info(f"GPT-5: Получен ответ со статус-кодом: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"GPT-5: Успешный ответ получен")
            logging.info(f"GPT-5: Структура ответа: {list(result.keys())}")
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                logging.info(f"GPT-5: Контент ответа получен, длина: {len(content)}")
                return content
            else:
                logging.error(f"GPT-5: Неожиданная структура ответа: {result}")
                return f"Ошибка: неожиданная структура ответа от GPT-5"
        else:
            logging.error(f"GPT-5: Ошибка API {response.status_code}: {response.text}")
            return f"Ошибка API: {response.status_code} - {response.text}"
    except Exception as e:
        logging.error(f"GPT-5: Исключение при обработке: {str(e)}")
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"

def get_ai_response(message: str, user_id: int) -> str:
    """
    Get AI response based on user's selected model with context
    """
    model = user_models.get(user_id, "gemini")  # Default to Gemini
    
    if model == "gemini":
        return get_gemini_response(user_id, message)
    elif model == "o4-mini":
        return get_o4_mini_response(user_id, message)
    elif model == "gpt-5":
        return get_gpt5_response(user_id, message)
    else:
        return get_gemini_response(user_id, message)  # Fallback to Gemini

def clean_markdown(text):
    """
    Clean problematic Markdown characters that might cause parsing errors
    """
    import re
    # Escape problematic characters for Telegram Markdown
    text = re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)
    return text

def split_message(text, max_length=4096):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Create model selection keyboard
def create_model_keyboard():
    keyboard = telebot.types.InlineKeyboardMarkup(row_width=1)
    
    gemini_btn = telebot.types.InlineKeyboardButton(
        "🤖 Gemini 2.5 Flash", 
        callback_data="model_gemini"
    )
    o4_mini_btn = telebot.types.InlineKeyboardButton(
        "🧠 o4-mini", 
        callback_data="model_o4-mini"
    )
    gpt5_btn = telebot.types.InlineKeyboardButton(
        "🚀 GPT-5", 
        callback_data="model_gpt-5"
    )
    
    keyboard.add(gemini_btn, o4_mini_btn, gpt5_btn)
    return keyboard

# Start command handler
@bot.message_handler(commands=['start'])
@check_access
def send_welcome(message):
    is_admin = getattr(message, 'is_admin', False)
    
    welcome_text = """
👋 Привет! Я AI-ассистент с поддержкой нескольких моделей.

Доступные модели:
🤖 **Gemini 2.5 Flash** - быстрая и умная модель Google
🧠 **o4-mini** - модель OpenAI для сложных задач  
🚀 **GPT-5** - новейшая модель OpenAI

**Команды:**
/model - выбор модели
/current - текущая модель
/context - информация о контексте
/clear - очистить контекст разговора"""

    if is_admin:
        welcome_text += """

🔑 **Админ команды:**
/users - управление пользователями
/add_user <ID> - добавить пользователя
/remove_user <ID> - удалить пользователя
/balance - информация об аккаунте
/account - подробная информация"""

    welcome_text += "\n\nПо умолчанию используется Gemini 2.5 Flash"
    
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

# Model selection command
@bot.message_handler(commands=['model'])
@check_access
def choose_model(message):
    current_model = user_models.get(message.from_user.id, "gemini")
    model_names = {
        "gemini": "Gemini 2.5 Flash",
        "o4-mini": "o4-mini", 
        "gpt-5": "GPT-5"
    }
    
    text = f"Текущая модель: **{model_names[current_model]}**\n\nВыберите модель для работы:"
    bot.send_message(
        message.chat.id, 
        text, 
        reply_markup=create_model_keyboard(),
        parse_mode='Markdown'
    )

# Current model command
@bot.message_handler(commands=['current'])
@check_access
def show_current_model(message):
    current_model = user_models.get(message.from_user.id, "gemini")
    model_names = {
        "gemini": "🤖 Gemini 2.5 Flash",
        "o4-mini": "🧠 o4-mini", 
        "gpt-5": "🚀 GPT-5"
    }
    
    bot.reply_to(message, f"Текущая модель: **{model_names[current_model]}**", parse_mode='Markdown')

# Context info command
@bot.message_handler(commands=['context'])
@check_access
def show_context_info(message):
    context_info = get_context_info(message.from_user.id)
    current_model = user_models.get(message.from_user.id, "gemini")
    model_names = {
        "gemini": "🤖 Gemini 2.5 Flash",
        "o4-mini": "🧠 o4-mini", 
        "gpt-5": "🚀 GPT-5"
    }
    
    response_text = f"""📝 **Информация о контексте**

{context_info}
Текущая модель: {model_names[current_model]}

Контекст помогает модели помнить предыдущие сообщения в разговоре.
Максимум: {MAX_CONTEXT_LENGTH} пар сообщений

Используйте /clear для очистки контекста."""
    
    bot.reply_to(message, response_text, parse_mode='Markdown')

# Clear context command
@bot.message_handler(commands=['clear'])
@check_access
def clear_context_command(message):
    clear_context(message.from_user.id)
    bot.reply_to(message, "🧹 **Контекст разговора очищен**\n\nТеперь модель не будет помнить предыдущие сообщения.", parse_mode='Markdown')

# Admin commands
@bot.message_handler(commands=['users'])
@check_access
def show_users(message):
    if not getattr(message, 'is_admin', False):
        bot.reply_to(message, "🚫 **Доступ запрещен**\n\nЭта команда доступна только администратору.", parse_mode='Markdown')
        return
    
    admin_id, allowed_users = get_users_info()
    
    response_text = f"""👥 **Управление пользователями**

🔑 **Администратор:** {admin_id}
📋 **Разрешенные пользователи:** {len(allowed_users)}

**Список пользователей:**"""
    
    if allowed_users:
        for user_id in allowed_users:
            response_text += f"\n• {user_id}"
    else:
        response_text += "\n_Пусто_"
    
    response_text += f"""

**Команды:**
/add_user <ID> - добавить пользователя
/remove_user <ID> - удалить пользователя
/users - показать список пользователей
/balance - информация об аккаунте
/account - подробная информация"""
    
    bot.reply_to(message, response_text, parse_mode='Markdown')

@bot.message_handler(commands=['add_user', 'adduser'])
@check_access
def add_user_command(message):
    logging.info(f"Получена команда /add_user от пользователя {message.from_user.id}")
    logging.info(f"Текст команды: '{message.text}'")
    logging.info(f"is_admin статус: {getattr(message, 'is_admin', 'НЕ УСТАНОВЛЕН')}")
    
    if not getattr(message, 'is_admin', False):
        logging.warning(f"Пользователь {message.from_user.id} попытался использовать /add_user без прав админа")
        bot.reply_to(message, "🚫 **Доступ запрещен**\n\nЭта команда доступна только администратору.", parse_mode='Markdown')
        return
    
    try:
        # Extract user ID from command
        command_parts = message.text.split()
        logging.info(f"Разбор команды: {command_parts}")
        
        if len(command_parts) != 2:
            logging.warning(f"Неверный формат команды: ожидается 2 части, получено {len(command_parts)}")
            bot.reply_to(message, "❌ **Неверный формат**\n\nИспользуйте: `/add_user <ID_пользователя>`", parse_mode='Markdown')
            return
        
        user_id = int(command_parts[1])
        logging.info(f"Попытка добавить пользователя с ID: {user_id}")
        
        add_user(user_id)
        logging.info(f"Пользователь {user_id} успешно добавлен")
        
        bot.reply_to(message, f"✅ **Пользователь добавлен**\n\nПользователь `{user_id}` добавлен в список разрешенных.", parse_mode='Markdown')
        logging.info(f"Администратор {message.from_user.id} добавил пользователя {user_id}")
        
    except ValueError as ve:
        logging.error(f"ValueError при парсинге ID пользователя: {ve}")
        bot.reply_to(message, "❌ **Ошибка**\n\nID пользователя должен быть числом.", parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Исключение при добавлении пользователя: {e}", exc_info=True)
        bot.reply_to(message, f"❌ **Ошибка при добавлении пользователя:** {str(e)}")

@bot.message_handler(commands=['remove_user', 'removeuser'])
@check_access
def remove_user_command(message):
    if not getattr(message, 'is_admin', False):
        bot.reply_to(message, "🚫 **Доступ запрещен**\n\nЭта команда доступна только администратору.", parse_mode='Markdown')
        return
    
    try:
        # Extract user ID from command
        command_parts = message.text.split()
        if len(command_parts) != 2:
            bot.reply_to(message, "❌ **Неверный формат**\n\nИспользуйте: `/remove_user <ID_пользователя>`", parse_mode='Markdown')
            return
        
        user_id = int(command_parts[1])
        admin_id, _ = get_users_info()
        
        # Prevent admin from removing themselves
        if user_id == admin_id:
            bot.reply_to(message, "❌ **Ошибка**\n\nАдминистратор не может удалить самого себя.", parse_mode='Markdown')
            return
        
        remove_user(user_id)
        
        bot.reply_to(message, f"✅ **Пользователь удален**\n\nПользователь `{user_id}` удален из списка разрешенных.", parse_mode='Markdown')
        logging.info(f"Администратор {message.from_user.id} удалил пользователя {user_id}")
        
    except ValueError:
        bot.reply_to(message, "❌ **Ошибка**\n\nID пользователя должен быть числом.", parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"❌ **Ошибка при удалении пользователя:** {str(e)}")

# Balance and account info command for admin
@bot.message_handler(commands=['balance', 'account'])
@check_access
def show_account_info(message):
    if not getattr(message, 'is_admin', False):
        bot.reply_to(message, "🚫 **Доступ запрещен**\n\nЭта команда доступна только администратору.", parse_mode='Markdown')
        return
    
    try:
        logging.info(f"Администратор {message.from_user.id} запросил информацию об аккаунте")
        
        # Get bot usage statistics
        admin_id, allowed_users = get_users_info()
        user_count = len(allowed_users)
        
        account_text = f"""💰 **Информация об аккаунте**

🔑 **ProxyAPI:**
• API ключ: ...{os.getenv('OPENAI_API_KEY', '')[-8:]}
• Статус: Активен
• Личный кабинет: https://proxyapi.ru

👥 **Пользователи бота:**
• Администратор: {admin_id}
• Всего пользователей: {user_count}
• Разрешенные ID: {', '.join(map(str, allowed_users[:5]))}{"..." if user_count > 5 else ""}

🤖 **Доступные модели:**
• Gemini 2.5 Flash (работает)
• o4-mini 
• GPT-5 (работает)

📊 **Мониторинг:**
• Логи: bot.log
• Контекст: До {MAX_CONTEXT_LENGTH} пар сообщений
• Сохраненных контекстов: {len(user_contexts)}
• Авторизация: Включена
• Файл контекстов: {USER_CONTEXTS_FILE}

💡 **Для проверки баланса:**
Войдите в личный кабинет ProxyAPI по ссылке выше.
ProxyAPI не предоставляет публичный API для баланса."""

        bot.reply_to(message, account_text, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Ошибка при получении информации об аккаунте: {e}", exc_info=True)
        bot.reply_to(message, f"❌ **Ошибка:** {str(e)}")

# Handler for unknown commands
@bot.message_handler(func=lambda message: message.text.startswith('/'))
@check_access
def handle_unknown_command(message):
    command = message.text.split()[0]
    logging.info(f"Неизвестная команда от пользователя {message.from_user.id}: {command}")
    
    available_commands = [
        "/start", "/model", "/current", "/context", "/clear"
    ]
    
    if getattr(message, 'is_admin', False):
        available_commands.extend(["/users", "/add_user", "/adduser", "/remove_user", "/removeuser", "/balance", "/account"])
    
    response = f"❓ **Неизвестная команда:** `{command}`\n\n**Доступные команды:**\n"
    for cmd in available_commands:
        response += f"• {cmd}\n"
    
    bot.reply_to(message, response, parse_mode='Markdown')

# Callback query handler for model selection
@bot.callback_query_handler(func=lambda call: call.data.startswith('model_'))
def handle_model_selection(call):
    # Check access for callback queries
    allowed, is_admin = is_user_allowed(call.from_user.id)
    if not allowed:
        bot.answer_callback_query(call.id, "🚫 Доступ запрещен")
        return
    
    model = call.data.replace('model_', '')
    user_models[call.from_user.id] = model
    
    model_names = {
        "gemini": "🤖 Gemini 2.5 Flash",
        "o4-mini": "🧠 o4-mini", 
        "gpt-5": "🚀 GPT-5"
    }
    
    bot.edit_message_text(
        f"✅ Выбрана модель: **{model_names[model]}**",
        call.message.chat.id,
        call.message.message_id,
        parse_mode='Markdown'
    )
    
    bot.answer_callback_query(call.id, f"Модель изменена на {model_names[model]}")

@bot.message_handler(content_types=['text'])
@check_access
def handle_text(message):
    """
    Handle incoming text messages
    """
    try:
        # Skip command messages
        if message.text.startswith('/'):
            logging.info(f"Пропускаем команду в handle_text: {message.text}")
            return
        
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        
        logging.info(f"Получено сообщение от пользователя {username} (ID: {user_id}): {message.text[:50]}...")
            
        # Get current model for user
        current_model = user_models.get(user_id, "gemini")
        model_names = {
            "gemini": "🤖 Gemini 2.5 Flash",
            "o4-mini": "🧠 o4-mini", 
            "gpt-5": "🚀 GPT-5"
        }
        
        logging.info(f"Пользователь {username} использует модель: {current_model}")
        
        # Show typing indicator
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Start thinking message timer
        send_thinking_message(message.chat.id, current_model)
        
        # Get response from selected AI model
        logging.info(f"Отправляем запрос к модели {current_model}...")
        response = get_ai_response(message.text, user_id)
        logging.info(f"Получен ответ от модели {current_model}, длина: {len(response)}")
        
        # Add to conversation context
        add_to_context(user_id, message.text, response)
        
        # Clear thinking message before sending response
        clear_thinking_message(message.chat.id)
        
        # Add model indicator to response
        model_indicator = f"\n\n_Ответ от {model_names[current_model]}_"
        
        # Send response back to user, splitting if too long
        parts = split_message(response)
        logging.info(f"Отправляем ответ пользователю в {len(parts)} частях")
        
        for i, part in enumerate(parts):
            try:
                # Try to send with Markdown formatting
                is_last_part = (i == len(parts) - 1)
                if is_last_part:  # Last part
                    logging.info(f"Отправляем последнюю часть с индикатором модели")
                    bot.reply_to(message, part + model_indicator, parse_mode='Markdown')
                else:
                    logging.info(f"Отправляем часть {i+1}/{len(parts)}")
                    bot.reply_to(message, part, parse_mode='Markdown')
            except Exception as markdown_error:
                logging.warning(f"Ошибка Markdown, пробуем очистить: {markdown_error}")
                # If Markdown parsing fails, clean the text and try again
                try:
                    cleaned_part = clean_markdown(part)
                    if is_last_part:  # Last part
                        bot.reply_to(message, cleaned_part + model_indicator, parse_mode='Markdown')
                    else:
                        bot.reply_to(message, cleaned_part, parse_mode='Markdown')
                except Exception as second_error:
                    logging.warning(f"Вторая попытка не удалась, отправляем без форматирования: {second_error}")
                    # If still fails, send without formatting
                    if is_last_part:  # Last part
                        bot.reply_to(message, part + f"\n\nОтвет от {model_names[current_model]}")
                    else:
                        bot.reply_to(message, part)
                
        logging.info(f"Ответ успешно отправлен пользователю {username}")
    except Exception as e:
        # Clear thinking message in case of error
        clear_thinking_message(message.chat.id)
        logging.error(f"Ошибка при обработке сообщения: {e}")
        bot.reply_to(message, f"Произошла ошибка: {str(e)}")

if __name__ == '__main__':
    print("Бот запущен...")
    
    # Load user contexts from file
    logging.info("Загружаем сохраненные контексты...")
    load_user_contexts()
    
    # Start bot
    bot.polling(none_stop=True)