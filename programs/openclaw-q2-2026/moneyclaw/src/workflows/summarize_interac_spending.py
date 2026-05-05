import os
from langchain_core.messages import HumanMessage, SystemMessage
from src.components.access_gmail import fetch_emails
from src.components.send_telegram import send_message
from src.llm import build_agent
from src.workflows.chat_with_ai_on_telegram import get_updates


def fetch_subjects(month: int, year: int, max_results: int) -> list[str]:
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    date_from = f"{year}-{month:02d}-01"
    date_to = f"{year}-{month:02d}-{last_day:02d}"
    emails = fetch_emails(
        subject_keyword="interac",
        from_email="",
        date_from=date_from,
        date_to=date_to,
        max_results=max_results,
    )
    return [e["subject"] for e in emails]


def run_summary(model: str, month: int, year: int, max_results: int, system_prompt: str) -> str:
    subjects = fetch_subjects(month, year, max_results)
    if not subjects:
        return "No Interac transactions found for the selected period."

    lines = "\n".join(f"- {s}" for s in subjects)
    prompt = f"Here are the Interac e-Transfer email subjects from {year}-{month:02d}:\n{lines}\n\nPlease summarize the transactions."

    agent = build_agent(model, system_prompt or None)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    summary = result["messages"][-1].content
    send_message(summary)
    return summary


def poll_and_respond(model: str, month: int, year: int, max_results: int, system_prompt: str, offset: int) -> tuple[list[dict], int]:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    updates = get_updates(token, offset)
    events = []

    for update in updates:
        offset = update["update_id"] + 1
        text = update.get("message", {}).get("text", "").strip().lower()
        if not text:
            continue

        if text == "summary":
            summary = run_summary(model, month, year, max_results, system_prompt)
            events.append({"trigger": "summary", "reply": summary})
        else:
            reply = "i don't know what you mean by that"
            send_message(reply)
            events.append({"trigger": text, "reply": reply})

    return events, offset
