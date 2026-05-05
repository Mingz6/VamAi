import os
import json
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


def get_gmail_service():
    token_data = json.loads(os.environ["GOOGLE_CREDENTIALS_PATH"])
    creds = Credentials.from_authorized_user_info(token_data)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("gmail", "v1", credentials=creds)


def fetch_emails(subject_keyword: str, from_email: str, date_from: str, date_to: str, max_results: int = 5) -> list[dict]:
    service = get_gmail_service()

    parts = []
    if subject_keyword:
        parts.append(f"subject:{subject_keyword}")
    if from_email:
        parts.append(f"from:{from_email}")
    if date_from:
        d = datetime.strptime(date_from, "%Y-%m-%d")
        parts.append(f"after:{d.strftime('%Y/%m/%d')}")
    if date_to:
        d = datetime.strptime(date_to, "%Y-%m-%d")
        parts.append(f"before:{d.strftime('%Y/%m/%d')}")

    query = " ".join(parts)
    result = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    messages = result.get("messages", [])

    emails = []
    for msg in messages:
        detail = service.users().messages().get(userId="me", id=msg["id"], format="metadata",
                                                 metadataHeaders=["From", "Subject", "Date"]).execute()
        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        labels = detail.get("labelIds", [])
        emails.append({
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "labels": labels,
        })
    return emails
