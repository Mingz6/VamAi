import os
import json
import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
]


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent(model: str, system_prompt: str | None = None):
    llm = ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    def chat_node(state: State):
        from langchain_core.messages import SystemMessage
        base = "Reply in a single message only. Do not split your response."
        combined = f"{base}\n\n{system_prompt}" if system_prompt else base
        msgs = [SystemMessage(content=combined)] + list(state["messages"])
        response = llm.invoke(msgs)
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()
