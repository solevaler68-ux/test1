#!/usr/bin/env python3

from __future__ import annotations
import time
from yandex_cloud_ml_sdk import YCloudML

messages_1 = [
    {
        "role": "system",
        "text": "Ты - умный помощник. Отвечай на вопросы кратко и понятно.",
    },
    {
        "role": "user",
        "text": input("Введите ваш вопрос: "),
    },
]


def main():

    sdk = YCloudML(
        folder_id="YANDEX_CLOUD_FOLDER",
        auth="YANDEX_CLOUD_API_KEY",
    )

    model = sdk.models.completions("yandexgpt")

    # Variant 1: wait for the operation to complete using 5-second sleep periods

    print("Variant 1:")

    operation = model.configure(temperature=0.5).run_deferred(messages_1)

    status = operation.get_status()
    while status.is_running:
        time.sleep(5)
        status = operation.get_status()

    result = operation.get_result()
    print(result)


if __name__ == "__main__":
    main()
