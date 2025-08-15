# Yandex&GigaChat - AI Assistant Bots

Проект содержит несколько модулей для работы с AI-моделями GigaChat и YandexGPT через Telegram-ботов.

## Файлы проекта

### `get_token.py` - GigaChat Token Helper
Небольшой модуль для получения токена доступа GigaChat. Реализован в файле `get_token.py` и может использоваться как самостоятельный скрипт, так и импортироваться в другие проекты.

- Функция: `get_gigachat_token()` — возвращает объект `AccessToken` из пакета `gigachat`.
- Источник ключей: параметры функции или переменные окружения/`.env`.
- Поддержка SSL: путь к корневому сертификату и управление проверкой SSL.

### `giga_bot.py` - Telegram Bot для GigaChat
Телеграм-бот, использующий GigaChat для генерации ответов. Бот выступает в роли умного помощника.

**Особенности:**
- Автоматическое получение токена через `get_token.py`
- Очистка Markdown/LaTeX разметки из ответов
- Поддержка SSL-сертификатов
- Логирование операций

**Переменные окружения:**
```env
TELEGRAM_BOT_TOKEN=<токен_бота_от_BotFather>
# GigaChat настройки (см. раздел ниже)
GIGACHAT_CLIENT_ID=<client_id>
GIGACHAT_CLIENT_SECRET=<client_secret>
# или
GIGACHAT_API_KEY=<BASE64_СТРОКА>
```

### `Yandex_bot.py` - Telegram Bot для YandexGPT
Телеграм-бот, использующий YandexGPT через Yandex Cloud ML SDK. Адаптирован из `generate-deferred.py`.

**Особенности:**
- Использование deferred API для асинхронных запросов
- Системный промпт: "Ты вежливый и умный Телеграмм помощник, помогай кратко и понятно"
- Очистка Markdown/LaTeX разметки
- Автоматическое опрашивание статуса операции

**Переменные окружения:**
```env
TELEGRAM_BOT_TOKEN=<токен_бота_от_BotFather>
YANDEX_CLOUD_FOLDER=<ваш_folder_id>
YANDEX_CLOUD_API_KEY=<ваш_api_key>
# Опционально
YANDEX_GPT_MODEL=yandexgpt
YANDEX_GPT_TEMPERATURE=0.5
```

### `Universal_bot.py` - Универсальный Telegram-бот (GigaChat + YandexGPT)
Бот-помощник, который позволяет переключаться между моделями GigaChat и YandexGPT в одном Telegram-боте.

**Особенности:**
- Переключение моделей командами: `/model`, `/model gigachat`, `/model yandex`, `/gigachat`, `/yandex`.
- Перед каждым запросом добавляется системный промпт: "Ты вежливый и профессиональный личный помощник, работающий в Telegram.".
- Обрабатывает только текстовые сообщения.
- Очищает Markdown/HTML/LaTeX-разметку перед отправкой ответа пользователю.
- Все ключи берутся из переменных окружения; добавлено логирование.

**Переменные окружения:**
Используются те же переменные, что и для отдельных ботов (`TELEGRAM_BOT_TOKEN`, `GIGACHAT_*`, `YANDEX_*`). Дополнительно можно задать `UNIBOT_LOG_LEVEL` (например, `INFO`, `DEBUG`).

### `generate-deferred.py` - Тестовый скрипт для YandexGPT
Исходный скрипт для тестирования взаимодействия с YandexGPT через консоль.

**Использование:**
```bash
python generate-deferred.py
```
Запрашивает вопрос через `input()` и выводит ответ от YandexGPT.

## Установка
```bash
python -m pip install -r requirements.txt
```

## Зависимости

### Основные
- `python-dotenv` - загрузка переменных окружения
- `gigachat` - клиент для GigaChat API
- `pyTelegramBotAPI` - библиотека для Telegram-ботов
- `yandex-cloud-ml-sdk` - SDK для Yandex Cloud ML

### SSL-сертификаты
- `russian_trusted_root_ca.cer` - корневой сертификат для GigaChat

## Настройка окружения (.env)
Создайте файл `.env` в корне проекта. Поддерживаются несколько вариантов для GigaChat — используйте любой один из них:

### GigaChat (для `get_token.py` и `giga_bot.py`)

**Вариант A — client_id/client_secret (рекомендуется)**
```env
GIGACHAT_CLIENT_ID=<client_id>
GIGACHAT_CLIENT_SECRET=<client_secret>
```

**Вариант B — готовая base64 из «Authorization: Basic …»**
```env
# Вставьте значение после "Basic "
GIGACHAT_API_KEY=<BASE64_СТРОКА>
# или можно вставить весь заголовок — модуль сам обрежет
# GIGACHAT_AUTHORIZATION=Authorization: Basic <BASE64_СТРОКА>
```

**SSL (при работе в окружениях с кастомными сертификатами):**
```env
GIGACHAT_CA_BUNDLE_FILE=C:\Users\Олег\Documents\ДЗ Zerocoder\Yandex&GigaChat\russian_trusted_root_ca.cer
GIGACHAT_VERIFY_SSL=1
```

### YandexGPT (для `Yandex_bot.py`)
```env
YANDEX_CLOUD_FOLDER=<ваш_folder_id>
YANDEX_CLOUD_API_KEY=<ваш_api_key>
YANDEX_GPT_MODEL=yandexgpt
YANDEX_GPT_TEMPERATURE=0.5
```

### Telegram (для всех ботов)
```env
TELEGRAM_BOT_TOKEN=<токен_бота_от_BotFather>
```

Для Universal‑бота поддерживается лог‑уровень через `UNIBOT_LOG_LEVEL` (по умолчанию INFO).

## Запуск

### GigaChat Bot
```bash
python giga_bot.py
```

### YandexGPT Bot
```bash
python Yandex_bot.py
```

### Universal Bot (переключение между GigaChat и YandexGPT)
```bash
python Universal_bot.py
```
Бот поддерживает команды:
- `/model` — показать текущую модель и доступность;
- `/model gigachat` — переключиться на GigaChat;
- `/model yandex` — переключиться на YandexGPT;
- `/gigachat` — быстрый переключатель на GigaChat;
- `/yandex` — быстрый переключатель на YandexGPT.

### Тестовый скрипт
```bash
python generate-deferred.py
```

### Получение токена GigaChat
```bash
python get_token.py
```

## Использование как модуля в другом проекте

### GigaChat Token Helper
```python
from get_token import get_gigachat_token

# Токен как объект
token_obj = get_gigachat_token()
print(token_obj.access_token)  # строковое значение JWE
```

### Переопределение опций из кода (необязательно):
```python
from get_token import get_gigachat_token

# Передать готовый base64 или собрать из id/secret через .env
tok = get_gigachat_token(
    # credentials="<BASE64>",
    verify_ssl_certs=True,
    ca_bundle_file=r"C:\\path\\to\\russian_trusted_root_ca.cer",
)
print(tok.access_token)
```

## Устранение неполадок

### GigaChat
- **Invalid credentials format:**
  - Проверьте, что задали именно base64 Authorization Data (после "Basic "), без лишних пробелов/переносов
  - Либо задайте `GIGACHAT_CLIENT_ID` и `GIGACHAT_CLIENT_SECRET` — модуль сам соберёт base64
- **401 Authorization error: header is incorrect:**
  - Используйте правильные данные из ЛК GigaChat (не путайте с client secret)
  - Убедитесь, что base64 не повреждён
- **SSL: certificate verify failed:**
  - Укажите `GIGACHAT_CA_BUNDLE_FILE` на `russian_trusted_root_ca.cer` и `GIGACHAT_VERIFY_SSL=1`
  - Избегайте отключения SSL-проверки, это небезопасно

### Telegram Bots
- **Markdown/LaTeX разметка в ответах:**
  - Боты автоматически очищают разметку перед отправкой
  - Если проблемы остаются, проверьте логи: `GIGABOT_LOG_LEVEL=DEBUG` или `YANDEX_BOT_LOG_LEVEL=DEBUG`

## Короткая проверка окружения
```bash
python - << 'PY'
import os; from dotenv import load_dotenv
load_dotenv()
print("=== GigaChat ===")
print("HAS_ID_SECRET:", bool(os.getenv("GIGACHAT_CLIENT_ID") and os.getenv("GIGACHAT_CLIENT_SECRET")))
print("HAS_BASE64:", bool(os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_AUTHORIZATION")))
print("VERIFY_SSL:", os.getenv("GIGACHAT_VERIFY_SSL"))
print("CA_FILE:", os.getenv("GIGACHAT_CA_BUNDLE_FILE"))

print("\n=== YandexGPT ===")
print("FOLDER_ID:", bool(os.getenv("YANDEX_CLOUD_FOLDER")))
print("API_KEY:", bool(os.getenv("YANDEX_CLOUD_API_KEY")))

print("\n=== Telegram ===")
print("BOT_TOKEN:", bool(os.getenv("TELEGRAM_BOT_TOKEN")))
PY
```

## Безопасность
- Не логируйте полные токены. Для отладки выводите только первые/последние символы.
- Файл `.env` не должен попадать в систему контроля версий.
- Используйте HTTPS для всех API-вызовов.
