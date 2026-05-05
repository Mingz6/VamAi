# Day 3 — Building MoneyClaw with Claude Code (Issam Laradji)

- **Source**: [VAM Official YouTube](https://www.youtube.com/watch?v=TuKEKMwUFkw)
- **Duration**: 1h 52m · ~13.7K words transcribed
- **Speaker**: Issam Laradji (host)
- **Mode**: Full audio + vision (whisper-large-v3-turbo for audio, qwen2.5vl:7b via Ollama for 75 keyframes @ 90s, Opus synthesis). All local, $0.
- **Transcript**: [`2026-04-25-day3-transcript.txt`](2026-04-25-day3-transcript.txt) · [`2026-04-25-day3-transcript.srt`](2026-04-25-day3-transcript.srt)
- **Frame log**: [`2026-04-25-day3-frame-by-frame.md`](2026-04-25-day3-frame-by-frame.md) (raw vision output)

---

## TL;DR

Issam live-codes **MoneyClaw** — a personal finance assistant — with Claude Code in VS Code, in ~1.5 hrs. The pattern is simple and repeatable: **draw a flowchart → spin up a Flask shell → build one component per file → wire them into workflows**. Stack: Python + Flask + LangGraph + OpenRouter + Claude Sonnet (for Claude Code itself). Components built live: AI Agent (chat), Gmail, Notion, Telegram, Web Search (Brave), Dashboard. Two workflows: chat-with-agent-via-telegram, and summarize-Interac-spend-and-push-to-telegram on the word "summary".

This is **Module 4 territory** (build the components and workflows from your Module 3 flowchart).

---

## Key Concepts

### The mental model: Agent = Brain + Data Sources + Tools
- **Data sources** (yellow in the flowchart): Gmail, Google Calendar, Notion — where data comes IN
- **Agent / Brain** (blue): the LLM + memory + soul (personality/system prompt) + planning harness
- **Tools / Services** (red): web search, send-to-telegram, generate-dashboard, create-notion-report — what the agent DOES
- The naming ("tool" vs "connector" vs "service") is semantic — pick one and stay consistent. Issam admits "data source" is the cleaner word for connectors.

### Soul / System prompt
- Default agent personality is "you are a helpful AI assistant" — bland.
- Soul = goals + values + constraints baked into system prompt. Example: *"You are an accountant focused on optimizing spending. Cap is $5K/mo. Anything over = red flag."*
- Notion can host the soul as a long-term, editable document.

### LangGraph
- Wraps the LLM with **harness**: planning, memory, tool execution, state tracking.
- Issam's framing: "LangGraph is to LLM what n8n is to a workflow — nodes + edges, but in code."
- **Honest caveat**: in this lecture he didn't actually use LangGraph's graph features — just had it as a library on hand. The MoneyClaw build is direct LLM calls via OpenRouter. He says: *"You don't have to use LangGraph for this — only when you want a complicated agentic flow."*

### Cloud Code vs Cursor vs Codex
- **Cursor**: gives you Sonnet/Opus access but uses Cursor's own harness.
- **Codex / Claude Code**: better harness — better planning, better command execution, better memory. Why? "It's more than the LLM — the harness matters."
- Claude Code = Anthropic's IDE-integrated coding agent (the one Issam uses; runs as a VS Code extension).
- Pavel said Codex ≈ Claude Code in capability. Issam hasn't tested Codex personally.

### Opus vs Sonnet vs Haiku
- **Sonnet**: default safe pick for iterative builds.
- **Opus**: 2x cost. Use for "ambitious one-shots" (e.g. "build me a whole personal assistant"). Bad for iteration — overkill.
- **Haiku**: cheap but needs more iterations to land.
- Recommendation: **iterative bottom-up with Sonnet**. Don't overwhelm yourself OR the agent.

### Lost in the middle
- When prompts get long, instructions in the middle get ignored.
- Fix: **repeat important instructions inline, even if they're already in CLAUDE.md**. Issam repeated "minimal code" inside prompts despite having it in CLAUDE.md for this exact reason.

### Caveman prompting
- Reduces token usage by ~75%.
- Style: "bug here. fix this. wrong here." — like a caveman.
- He didn't use it in this demo but mentioned the technique exists; there's a public CLAUDE.md for it on GitHub.

---

## Code & Commands Shown

### CLAUDE.md (project root, applies to every Claude Code prompt)
Issam's system prompt fragments (will be posted on the hackathon site):
- Always be concise. Use minimum code and minimum libraries.
- Keep things modular. One file per component under `src/components/`. One file per workflow under `src/workflows/`.
- Don't repeat code. Use templates. Consolidate duplicates.
- Short answers, no fillers.
- **Always end responses with "all done amigo"** (his personal signal that the task completed).

### Project layout that emerged
```
moneyclaw/
  app.py                    # Flask entry, port 7010
  requirements.txt          # auto-managed by Claude Code
  templates/                # HTML pages
  CLAUDE.md                 # agent instructions
  .gitignore                # MUST include .env
  .env                      # API keys, NEVER committed
  data/                     # downloaded emails, dashboard inputs
  src/
    llms/                   # OpenRouter wrapper
    components/             # one file per component
      access_gmail.py
      access_notion.py
      send_telegram.py
      access_dashboard.py
      web_search.py
    workflows/              # one file per workflow
      chat_with_ai_on_telegram.py
      summarize_interac_spendings.py
```

### The repeating prompt pattern
Every component and workflow built with the same shape:
```
Create a component at src/components/<name>.py
similar to the screenshot I'll share.
- It should allow me to <action>.
- Add a screenshot-style card to the dashboard.
- Add to the README how to obtain the API key for <service>, with all steps and URLs.
- Use minimal code.
- For components not yet implemented, tag them "not implemented yet".
```

### Specific prompts called out
1. **Initial scaffold**: *"Create a minimal Flask app called Moneyclaw with a nice landing page. Use port 7010. Use templates. Make the theme bright. Use minimal code."*
2. **Components dashboard**: *"In the navbar, remove features/pricing/login and add a 'components' tab linking to components.html. Show three subsections: data connectors, agents, tools — as cards similar to [screenshot]. Don't implement them yet — tag unimplemented as 'not implemented yet'."*
3. **AI Agent chat**: *"Create the agent in src/llms similar to [screenshot]. Use OpenRouter and LangGraph. Add libraries to requirements.txt. Allow me to chat with the agent. Add to README how to obtain the OpenRouter API key. Clicking the AI agent card opens this interface."*
4. **Workflow combo**: *"Add a workflows tab. Create two workflow cards listing which components each uses. Workflow 1: chat-with-AI-on-telegram (uses agent + telegram). Workflow 2: summarize-interac-spend-from-gmail-to-telegram (uses gmail + agent + telegram). Implement under src/workflows/."*

### Key APIs / services used
| Service | Purpose | Notes |
|---|---|---|
| **OpenRouter** | Model gateway (GPT-4o-mini, Sonnet, etc.) | Has a free tier. No CC needed to start. |
| **Google APIs** (Gmail OAuth) | Read inbox | OAuth dance: project → API & Services → OAuth consent screen → create OAuth client (Desktop) → download JSON. **Use `gmail.readonly` scope** — never write/delete. |
| **Notion API** | Read/edit goal pages | Easier than Gmail: my-integrations → create → copy token → connect integration to specific page. |
| **Telegram Bot API** | Send/receive messages | Talk to `@BotFather` → `/newbot` → name → username (must end in `bot`) → token. Then get your chat ID via `/getUpdates`. |
| **Brave Search API** | Web search inside the agent | 1K-2K free searches/day. Cleaner than scraping Google. |

### The "listening mode" / heartbeat pattern
Asked by Farah: *"Can the agent run a periodic check?"*
Issam's prompt: *"Add to the run button: 'send every X seconds where user can select X' — that sends a recurring message."*
Result: a basic cron-loop inside the workflow. Same pattern reused for "listen for `summary` keyword on telegram" — the workflow stays in a polling loop and triggers the summarizer when keyword arrives.

---

## Diagrams & Visuals (confirmed by vision pass)

### The Excalidraw flowchart (built live, t=1:30 → t=13:30)

Final title: **"MoneyClaw Optimize Spendings"**

Exact node labels (vision-confirmed):

| Group | Color | Nodes |
|---|---|---|
| **Data Sources** | 🟡 yellow rectangles | `Gmail (transactions)`, `Google Calendar (meetings)`, `Notion (goals)` |
| **Agent (Brain)** | 🔵 blue | `AI Agent (OpenRouter + LangGraph)` with `Memory` and `Soul` as sub-nodes |
| **Tools (Services)** | 🌸 pink | `Do Web Search (alternatives)`, `Send to Telegram (summarize)`, `Create Report on Notion (spendings)`, `Generate Dashboard (Web App)` |
| **Legend** | corner box | maps color → role |

Edges: data sources → agent · agent → each tool. **This is your Module 3 deliverable** — copy the structure, swap node labels for CartClaw (Gmail receipts/Amazon emails → agent → web-search-price-compare + telegram-deal-alert + notion-cart-history).

### The MoneyClaw landing page (t=20:00–22:30, 56:00)

Claude Code generated:
- Headline: **"Stop Overspending. Start Winning."**
- Sub: *"MoneyClaw analyzes your spending, finds waste, and gives you a clear path to financial freedom."*
- Orange `Get Started Free` CTA
- 4 feature cards: Spending Insights · Smart Cuts · Goal Tracker · Overspend Alerts
- Light yellow background, white cards, orange accents (the bright theme he asked for)

### The Components dashboard (t=24:00, 50:00, 74:00)

Three sections, each card showing a status tag:
- **Data Connectors**: Gmail · Google Calendar (`not implemented yet`) · Notion
- **Agent (Brain)**: Memory (`not implemented yet`) · Soul (`not implemented yet`) · AI Agent
- **Tools (Services)**: Web Search · Send to Telegram · Create Report on Notion · Generate Dashboard

### The AI Agent chat UI (t=45:00–48:00)

Tagline: **"7 models · 1 personality"**. Model dropdown defaults to `anthropic/claude-sonnet-4-5`; also lists `openai/gpt-4`, `openai/gpt-4-mini`, others. Plain message input + Send button. Standard chat-history layout.

### Gmail component UI (t=61:30)

`MoneyClaw — Access Gmail`. Form fields: `SUBJECT KEYWORD`, `FROM (EMAIL)`, `MAX RESULTS`, `DATE FROM`, `DATE TO`, with a `Run` button. Defaults: subject=`Interac`, max=5, dates=Jan 1–31.

### Notion editor (t=66:00–75:00)

Issam's actual `AgentWatson` page in his Notion workspace `Issam's Workspace`. Goal text:

> My goals are to optimize my money for the year. I want to travel to Spain on December for 2 week vacation. I want to buy a house in 5 years. I want to travel 3 times a year. **make sure that all your answers are in 3 bullet points.**

That last line is the **Soul / system prompt** baked into a data source. Clever pattern — the goal doc IS the personality.

### Telegram component (t=79:30–88:30)

Form: text field + `CHAT ID (LEAVE BLANK TO USE DEFAULT)` + Run button. First run failed with `Failed 'TELEGRAM_BOT_TOKEN'` env error → fixed → `400 Client Error: Bad Request for url: https://api.telegram.org/bot/token` → fixed by setting actual token. Then sent `hola amigo` → success → showed the message landing in Telegram Web. Heartbeat mode added later ("don't spend any money" looped every 5 sec).

### Dashboard component (t=90:00, 91:30)

Title: `MoneyClaw Dashboard`. Tiles: `TOTAL RECEIVED $0.00` · `TRANSACTIONS 0` · `BREAKDOWN BY User` dropdown. Bar chart / pie chart toggle. Hover reveals per-user totals.

### Web Search component (t=96:00–99:00)

Simple `QUERY` field (defaulted to `netflix`) + Run button. README explained Brave Search API key setup. First result was a YouTube link.

### Workflows page (t=102:00–111:00)

`MoneyClaw Workflows — End-to-end automations built from components`. Two cards:
1. **Chat with AI on Telegram** (uses AI Agent + Telegram)
2. **Summarize Interac Spending from Gmail to Telegram** (uses Gmail + AI Agent + Telegram)

The summarize workflow UI: model dropdown, month, year, max messages, plus two buttons: `Run` (one-shot) and `Listen to Summary` (polls Telegram for the keyword `summary`).

### Vision-confirmed code in `app.py` (t=48:00)

```python
import json
import uuid
from flask import Flask, render_template, request, jsonify, session
from langchain_core.messages import HumanMessage, AIMessage
from src.llm import build_agent, MODELS

app = Flask(__name__)
app.secret_key = "moneyclaw-secret"

@app.route("/")
def landing():
    return render_template("landing_page.html")
```

Workflow file imports (t=103:30):
```python
import requests
from src.llm import build_agent
from src.components.send_telegram import send
```

---

## Decisions & Recommendations

| Decision | Issam's call | Reasoning |
|---|---|---|
| Web framework | **Flask** (Python) | Issam's personal preference. Node.js is the industry default but "Python is readable." |
| Vibe-coding tool | **Claude Code** | Better harness than Cursor's built-in or Codex (per his testing). |
| Default model | **Sonnet** | Good iteration cost/quality. Use Opus for one-shot ambitious tasks only. |
| Repo visibility | **Start private** | Risk of accidentally pushing API keys. Open it after the .env audit. |
| Module file | **CLAUDE.md** (or `agents.md`) | Thomas suggested `agents.md` is more universal; both work. |
| Scope discipline | **Read-only by default** | Gmail scope = `gmail.readonly`. Never request write/delete unless absolutely needed. |
| Modularity | **One file per component** | Easy to test individually. The agent learned the convention without being told (when adding telegram). |
| Storage | **`data/` folder** for downloaded emails, JSON snapshots | Predictable, easy to gitignore. |
| Deployment | **render.com** when local works | Brought up by an attendee. Issam confirmed render is the path. Will cover later in the week. |

---

## Action Items For You

Mapped to your CartClaw situation:

1. **Submit Module 3 NOW** (today's deadline). The flowchart is straightforward — yellow data sources, blue agent (with memory + soul sub-bubbles), red tools, edges, legend. Use Excalidraw. CartClaw equivalent: Gmail receipts/Amazon order emails → agent → Web Search (price compare) + Notion (cart history) + Telegram (deal alert).
2. **Clone the pattern, not the code.** Issam's MoneyClaw repo (already cloned at `vam/openclaw-hackathon-q2-2026/moneyclaw/`) gives you the full project skeleton. For CartClaw with Sino, fork the structure: `src/components/` per data source/tool, `src/workflows/` for combos.
3. **Write your CLAUDE.md first**, before any code. Steal Issam's defaults (concise, minimal libs, modular, one file per component) + add CartClaw-specific rules (e.g., "all price data in CAD; flag USD prices explicitly").
4. **Use OpenRouter + free-tier models** to start. Your existing personal stack (`neuro-ming` uses Azure OpenAI directly) is overkill for hackathon iteration speed.
5. **Skip LangGraph until you actually need it.** Issam imported it but never used the graph API. Direct `openrouter.chat()` calls are fine for Module 4-6.
6. **Plan your Module 5 deliverable** (due Mon Apr 28): team confirmed (Sino), flowchart submitted, repo created. The CartClaw repo is your single Module 5 asset.

---

## Quotes Worth Saving

> "Sometimes you have to mention things three or four times so it stays in the agent's brain. There's a thing called 'lost in the middle' — long prompts lose middle instructions. Repeat what matters."

> "Cloud Code is more than the LLM. The harness — planning, memory, executing commands — that's what makes it work."

> "Always start private. The risk of pushing API keys is high. Open it later."

> "It's always good to start simple and add layers. Don't overwhelm yourself OR the agent."

> "I know this is overwhelming, but trust me — the other ways are way more overwhelming."

---

## Resources Mentioned

- **Excalidraw** — flowchart tool ([excalidraw.com](https://excalidraw.com))
- **OpenRouter** — model gateway ([openrouter.ai](https://openrouter.ai))
- **LangGraph** — agent harness ([langchain.com/langgraph](https://www.langchain.com/langgraph))
- **Brave Search API** — web search ([brave.com/search/api](https://brave.com/search/api))
- **Telegram BotFather** — `@BotFather` in Telegram
- **render.com** — deployment target
- **Issam's MoneyClaw repo** — [github.com/IssamLaradji/moneyclaw](https://github.com/IssamLaradji/moneyclaw) (already cloned locally)
- **caveman.md prompt** — exists publicly on GitHub (Issam didn't link it, search "caveman claude.md")

---

## Pipeline Notes

Full video-watcher run on this lecture:

| Step | Tool | Time | Output |
|---|---|---|---|
| Audio download | yt-dlp | 6s | 49MB mp3 16kHz mono |
| Video download (480p) | yt-dlp | 26s | 106MB mp4 |
| Transcription | whisper-cli (large-v3-turbo, Metal) | 4m 12s | 85KB transcript + SRT |
| Keyframe extraction | ffmpeg (`fps=1/90`) | 5s | 75 frames @ ~50KB each |
| Vision pass | qwen2.5vl:7b via Ollama HTTP API | ~17 min | 75 frame descriptions |
| Synthesis | Claude Opus | inline | this digest |
| **Total wall-clock** | | **~25 min** | **$0 (all local)** |

### Lessons from this run

1. ⚠️ **Ollama CLI silently drops images** — `ollama run model "prompt" image.jpg` ignores the image and the model hallucinates. **Must use the HTTP API** at `http://localhost:11434/api/generate` with base64-encoded `images` array. Recorded in user memory.
2. **90s sampling is the right cadence** for code walkthroughs. 75 frames captured all UI transitions cleanly without redundancy.
3. **Vision adds ~30%, not 20%** as initially estimated. The Excalidraw exact labels, model dropdown contents, error messages, and code snippets were all genuinely missing from the audio. Worth the extra 17 min.
4. **qwen2.5vl:7b quality on M3 Max** is solid for UI/diagram/code identification. Some minor stutters (`a a` repetitions, occasional fabrication of generic detail) but never invented code that wasn't there.

---

*Digest generated 2026-04-27. Full pipeline: yt-dlp → whisper-cli (large-v3-turbo) → ffmpeg keyframes → qwen2.5vl:7b vision → Claude Opus synthesis. Wall-clock ~25 min, $0.*
