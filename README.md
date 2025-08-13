### GigaChat Token Helper

Небольшой модуль для получения токена доступа GigaChat. Реализован в файле `get_token.py` и может использоваться как самостоятельный скрипт, так и импортироваться в другие проекты.

- Функция: `get_gigachat_token()` — возвращает объект `AccessToken` из пакета `gigachat`.
- Источник ключей: параметры функции или переменные окружения/`.env`.
- Поддержка SSL: путь к корневому сертификату и управление проверкой SSL.

### Установка
```bash
python -m pip install -r requirements.txt
```

### Настройка окружения (.env)
Создайте файл `.env` в корне рядом с `get_token.py`. Поддерживаются несколько вариантов — используйте любой один из них:

Вариант A — client_id/client_secret (рекомендуется)
```env
GIGACHAT_CLIENT_ID=<client_id>
GIGACHAT_CLIENT_SECRET=<client_secret>
```

Вариант B — готовая base64 из «Authorization: Basic …»
```env
# Вставьте значение после "Basic "
GIGACHAT_API_KEY=<BASE64_СТРОКА>
# или можно вставить весь заголовок — модуль сам обрежет
# GIGACHAT_AUTHORIZATION=Authorization: Basic <BASE64_СТРОКА>
```

SSL (при работе в окружениях с кастомными сертификатами):
```env
GIGACHAT_CA_BUNDLE_FILE=C:\Users\Олег\Documents\ДЗ Zerocoder\Yandex&GigaChat\russian_trusted_root_ca.cer
GIGACHAT_VERIFY_SSL=1
```

Примечания:
- `GIGACHAT_API_KEY` — это именно Authorization Data (base64), а не client secret.
- Модуль автоматически:
  - удаляет префиксы `Authorization:` и `Basic`
  - очищает пробелы/переносы/кавычки
  - поддерживает base64url и добавляет недостающее `=` выравнивание

### Использование как скрипта
```bash
python get_token.py
```
Выведет объект `AccessToken`.

### Использование как модуля в другом проекте
1) Скопируйте `get_token.py` и (по необходимости) `russian_trusted_root_ca.cer` в ваш проект.
2) Установите зависимости из `requirements.txt`.
3) Импортируйте и получите токен:
```python
from get_token import get_gigachat_token

# Токен как объект
token_obj = get_gigachat_token()
print(token_obj.access_token)  # строковое значение JWE
```

Переопределение опций из кода (необязательно):
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

### Устранение неполадок
- Invalid credentials format:
  - Проверьте, что задали именно base64 Authorization Data (после "Basic "), без лишних пробелов/переносов
  - Либо задайте `GIGACHAT_CLIENT_ID` и `GIGACHAT_CLIENT_SECRET` — модуль сам соберёт base64
- 401 Authorization error: header is incorrect:
  - Используйте правильные данные из ЛК GigaChat (не путайте с client secret)
  - Убедитесь, что base64 не повреждён
- SSL: certificate verify failed:
  - Укажите `GIGACHAT_CA_BUNDLE_FILE` на `russian_trusted_root_ca.cer` и `GIGACHAT_VERIFY_SSL=1`
  - Избегайте отключения SSL-проверки, это небезопасно

Короткая проверка окружения:
```bash
python - << 'PY'
import os; from dotenv import load_dotenv
load_dotenv()
print("HAS_ID_SECRET:", bool(os.getenv("GIGACHAT_CLIENT_ID") and os.getenv("GIGACHAT_CLIENT_SECRET")))
print("HAS_BASE64:", bool(os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_AUTHORIZATION")))
print("VERIFY_SSL:", os.getenv("GIGACHAT_VERIFY_SSL"))
print("CA_FILE:", os.getenv("GIGACHAT_CA_BUNDLE_FILE"))
PY
```

### Безопасность
- Не логируйте полный токен. Для отладки выводите только первые/последние символы.
