import os
import requests


def send_message(text: str, chat_id: str = None) -> dict:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    res = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
    )
    res.raise_for_status()
    return res.json()
