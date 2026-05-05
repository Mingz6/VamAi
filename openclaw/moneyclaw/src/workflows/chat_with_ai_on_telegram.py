import os
import time
import requests
from src.llm import build_agent
from src.components.send_telegram import send_message
from langchain_core.messages import HumanMessage

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def get_updates(token: str, offset: int) -> list[dict]:
    res = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        params={"offset": offset, "timeout": 0},
        timeout=10,
    )
    if res.status_code == 409:
        return []  # another process is polling — skip silently
    res.raise_for_status()
    return res.json().get("result", [])


def run_once(model: str, offset: int) -> tuple[list[dict], int]:
    """Poll once. Returns (events, new_offset). Each event: {text, reply}."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    updates = get_updates(token, offset)
    events = []

    agent = build_agent(model)
    for update in updates:
        offset = update["update_id"] + 1
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        if not text:
            continue

        result = agent.invoke({"messages": [HumanMessage(content=text)]})
        reply = result["messages"][-1].content
        send_message(reply)
        events.append({"received": text, "reply": reply})

    return events, offset
