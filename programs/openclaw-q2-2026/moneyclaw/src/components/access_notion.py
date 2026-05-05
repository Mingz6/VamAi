import os
from notion_client import Client

PAGE_ID = os.environ["NOTION_PAGE_ID"]


def get_client():
    return Client(auth=os.environ["NOTION_TOKEN"])


def get_page() -> dict:
    client = get_client()
    page = client.pages.retrieve(page_id=PAGE_ID)
    blocks = client.blocks.children.list(block_id=PAGE_ID)
    return {"page": page, "blocks": blocks["results"]}


def update_block(block_id: str, text: str):
    client = get_client()
    client.blocks.update(
        block_id=block_id,
        paragraph={"rich_text": [{"type": "text", "text": {"content": text}}]},
    )


def append_block(text: str):
    client = get_client()
    client.blocks.children.append(
        block_id=PAGE_ID,
        children=[{"object": "block", "type": "paragraph",
                   "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}}],
    )
