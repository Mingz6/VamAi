import os
import requests


def search(query: str, count: int = 5) -> list[dict]:
    res = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_API_KEY"]},
        params={"q": query, "count": count},
    )
    res.raise_for_status()
    hits = res.json().get("web", {}).get("results", [])
    return [{"title": h["title"], "url": h["url"], "description": h.get("description", "")} for h in hits]
