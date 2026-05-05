import json
import os
import uuid
from dotenv import load_dotenv
load_dotenv()  # must be before any component imports that read os.environ

from flask import Flask, render_template, request, jsonify, session, send_from_directory
from langchain_core.messages import HumanMessage, AIMessage
from src.llm import build_agent, MODELS
from src.components.access_gmail import fetch_emails
from src.components.access_notion import get_page, update_block, append_block
from src.components.send_telegram import send_message as telegram_send
from src.components.access_dashboard import load_and_parse, aggregate
from src.components.search_web import search as brave_search

app = Flask(__name__)
app.secret_key = "moneyclaw-secret"

# In-memory history store: session_id -> list of {role, content, in_tokens, out_tokens}
histories: dict[str, list] = {}


@app.route("/")
def landing():
    return render_template("landing_page.html")


@app.route("/flowchart")
def flowchart():
    return render_template("flowchart.html")


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)


@app.route("/components")
def components():
    return render_template("components.html")


@app.route("/workflows")
def workflows():
    return render_template("workflows.html")


@app.route("/agent")
def agent():
    return render_template("agent.html", models=MODELS)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    model = data["model"]
    user_msg = data["message"]
    session_id = data.get("session_id") or str(uuid.uuid4())
    system_prompt = data.get("system_prompt") or None

    history = histories.setdefault(session_id, [])

    lc_messages = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=user_msg))

    agent_graph = build_agent(model, system_prompt)
    result = agent_graph.invoke({"messages": lc_messages})
    ai_msg = result["messages"][-1]

    in_tok = ai_msg.usage_metadata.get("input_tokens", 0) if ai_msg.usage_metadata else 0
    out_tok = ai_msg.usage_metadata.get("output_tokens", 0) if ai_msg.usage_metadata else 0

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ai_msg.content, "in_tokens": in_tok, "out_tokens": out_tok})

    return jsonify({
        "reply": ai_msg.content,
        "session_id": session_id,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    })


@app.route("/api/sessions")
def sessions():
    return jsonify([
        {"id": sid, "label": histories[sid][0]["content"][:30] if histories[sid] else sid}
        for sid in reversed(list(histories.keys()))
    ])


@app.route("/api/sessions/<sid>")
def get_session(sid):
    return jsonify(histories.get(sid, []))


@app.route("/gmail")
def gmail():
    return render_template("gmail.html")


@app.route("/api/gmail/download", methods=["POST"])
def api_gmail_download():
    emails = request.json.get("emails", [])
    headers = [{"from": e["from"], "subject": e["subject"], "date": e["date"], "labels": e["labels"]} for e in emails]
    os.makedirs("data", exist_ok=True)
    with open("data/gmail_messages.json", "w") as f:
        json.dump(headers, f, indent=2)
    return jsonify({"message": f"Saved {len(headers)} emails to data/gmail_messages.json"})


@app.route("/api/gmail", methods=["POST"])
def api_gmail():
    data = request.json
    try:
        emails = fetch_emails(
            subject_keyword=data.get("subject", ""),
            from_email=data.get("from_email", ""),
            date_from=data.get("date_from", ""),
            date_to=data.get("date_to", ""),
            max_results=data.get("max_results", 5),
        )
        return jsonify({"emails": emails})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/telegram")
def telegram():
    return render_template("telegram.html")


@app.route("/api/telegram/send", methods=["POST"])
def api_telegram_send():
    data = request.json
    try:
        result = telegram_send(data["message"], data.get("chat_id") or None)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/search")
def search_web():
    return render_template("search_web.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    try:
        results = brave_search(request.json["query"])
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/dashboard")
def dashboard():
    files = [f for f in os.listdir("data") if f.endswith(".json")]
    return render_template("dashboard.html", files=sorted(files))


@app.route("/api/dashboard", methods=["POST"])
def api_dashboard():
    data = request.json
    try:
        filepath = os.path.join("data", data["file"])
        transactions = load_and_parse(filepath)
        agg = aggregate(transactions, data.get("breakdown", "user"))
        return jsonify({"transactions": transactions, "aggregated": agg})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/notion")
def notion():
    return render_template("notion.html")


@app.route("/api/notion/page")
def api_notion_page():
    try:
        data = get_page()
        title = data["page"]["properties"].get("title", {}).get("title", [{}])
        title_text = title[0].get("plain_text", "Untitled") if title else "Untitled"
        return jsonify({"title": title_text, "blocks": data["blocks"]})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/notion/block", methods=["POST"])
def api_notion_block():
    data = request.json
    try:
        update_block(data["block_id"], data["text"])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/notion/append", methods=["POST"])
def api_notion_append():
    data = request.json
    try:
        append_block(data["text"])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/workflows/summarize-interac")
def workflow_summarize_interac():
    return render_template("workflow_summarize_interac.html", models=MODELS)


@app.route("/api/workflows/summarize/run", methods=["POST"])
def api_summarize_run():
    from src.workflows.summarize_interac_spending import run_summary
    d = request.json
    try:
        summary = run_summary(d["model"], d["month"], d["year"], d["max_results"], d.get("system_prompt", ""))
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route("/api/workflows/summarize/poll", methods=["POST"])
def api_summarize_poll():
    from src.workflows.summarize_interac_spending import poll_and_respond
    d = request.json
    try:
        events, new_offset = poll_and_respond(d["model"], d["month"], d["year"], d["max_results"], d.get("system_prompt", ""), _tg_current())
        _tg_read_and_advance(new_offset)
        return jsonify({"events": events, "offset": new_offset})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/workflows/chat-telegram")
def workflow_chat_telegram():
    return render_template("workflow_chat_telegram.html", models=MODELS)


import requests as _req
import threading
_tg_lock = threading.Lock()
_tg_offset: int = 0

def _tg_init():
    global _tg_offset
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    try:
        r = _req.get(f"https://api.telegram.org/bot{token}/getUpdates",
                     params={"offset": -1, "timeout": 0}, timeout=5)
        results = r.json().get("result", [])
        _tg_offset = results[-1]["update_id"] + 1 if results else 0
    except Exception:
        _tg_offset = 0

_tg_init()  # runs once; use_reloader=False ensures only one process

def _tg_read_and_advance(new_offset: int):
    global _tg_offset
    with _tg_lock:
        _tg_offset = new_offset

def _tg_current() -> int:
    with _tg_lock:
        return _tg_offset

@app.route("/api/workflows/chat-telegram/poll", methods=["POST"])
def api_chat_telegram_poll():
    from src.workflows.chat_with_ai_on_telegram import run_once
    data = request.json
    try:
        events, new_offset = run_once(data["model"], _tg_current())
        _tg_read_and_advance(new_offset)
        return jsonify({"events": events, "offset": new_offset})
    except Exception as e:
        return jsonify({"error": str(e)})


def _mask(v): return ("*" * (len(v) - 4) + v[-4:]) if len(v) > 4 else "*" * len(v)

def _env_save(updates: dict):
    lines = open(".env").readlines()
    new_lines, replaced = [], set()
    for line in lines:
        key = line.split("=")[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}\n"); replaced.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in replaced: new_lines.append(f"{key}={val}\n")
    with open(".env", "w") as f: f.writelines(new_lines)
    for key, val in updates.items(): os.environ[key] = val


@app.route("/api/env/notion", methods=["GET"])
def api_env_notion_get():
    t, p = os.environ.get("NOTION_TOKEN",""), os.environ.get("NOTION_PAGE_ID","")
    return jsonify({"token_set": bool(t), "token_masked": _mask(t) if t else "",
                    "page_id_set": bool(p), "page_id_masked": _mask(p) if p else ""})

@app.route("/api/env/notion", methods=["POST"])
def api_env_notion_set():
    d = request.json; u = {}
    if d.get("token"): u["NOTION_TOKEN"] = d["token"]
    if d.get("page_id"): u["NOTION_PAGE_ID"] = d["page_id"]
    _env_save(u); return jsonify({"ok": True})


@app.route("/api/env/telegram", methods=["GET"])
def api_env_telegram_get():
    t, c = os.environ.get("TELEGRAM_BOT_TOKEN",""), os.environ.get("TELEGRAM_CHAT_ID","")
    return jsonify({"token_set": bool(t), "token_masked": _mask(t) if t else "",
                    "chat_id_set": bool(c), "chat_id_masked": _mask(c) if c else ""})

@app.route("/api/env/telegram", methods=["POST"])
def api_env_telegram_set():
    d = request.json; u = {}
    if d.get("token"): u["TELEGRAM_BOT_TOKEN"] = d["token"]
    if d.get("chat_id"): u["TELEGRAM_CHAT_ID"] = d["chat_id"]
    _env_save(u); return jsonify({"ok": True})


@app.route("/api/env/gmail", methods=["GET"])
def api_env_gmail_get():
    val = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")
    return jsonify({"set": bool(val), "masked": "JSON credentials set ✓" if val else ""})

@app.route("/api/env/gmail", methods=["POST"])
def api_env_gmail_set():
    d = request.json
    if d.get("credentials"): _env_save({"GOOGLE_CREDENTIALS_PATH": d["credentials"]})
    return jsonify({"ok": True})


@app.route("/api/env/openrouter", methods=["GET"])
def api_env_openrouter_get():
    k = os.environ.get("OPENROUTER_API_KEY","")
    return jsonify({"key_set": bool(k), "key_masked": _mask(k) if k else ""})

@app.route("/api/env/openrouter", methods=["POST"])
def api_env_openrouter_set():
    d = request.json; u = {}
    if d.get("key"): u["OPENROUTER_API_KEY"] = d["key"]
    _env_save(u); return jsonify({"ok": True})


@app.route("/api/env/brave", methods=["GET"])
def api_env_brave_get():
    k = os.environ.get("BRAVE_API_KEY","")
    return jsonify({"key_set": bool(k), "key_masked": _mask(k) if k else ""})

@app.route("/api/env/brave", methods=["POST"])
def api_env_brave_set():
    d = request.json; u = {}
    if d.get("key"): u["BRAVE_API_KEY"] = d["key"]
    _env_save(u); return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(port=7010, debug=True, use_reloader=False)
