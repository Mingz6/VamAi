import re
import json


def parse_emails(emails: list[dict]) -> list[dict]:
    results = []
    for e in emails:
        subject = e.get("subject", "")
        date = e.get("date", "")

        # received patterns:
        # "You've received $X from NAME and..."
        # "You received $X from NAME"
        m = re.search(r"received \$([\d,]+\.?\d*) from ([A-Za-z][A-Za-z\s\-]+?)(?:\s+and|\.|$)", subject)
        if m:
            results.append({
                "type": "received",
                "amount": float(m.group(1).replace(",", "")),
                "user": m.group(2).strip().title(),
                "date": date,
                "subject": subject,
            })
            continue

        # sent patterns:
        # "Your $X transfer to NAME has been..."
        # "Transfer of $X to NAME"
        m = re.search(r"(?:Your \$([\d,]+\.?\d*) transfer to ([A-Za-z][A-Za-z\s\-]+?) has|Transfer of \$([\d,]+\.?\d*) to ([A-Za-z][A-Za-z\s\-]+?)(?:\.|,|$))", subject)
        if m:
            amount = m.group(1) or m.group(3)
            user = m.group(2) or m.group(4)
            results.append({
                "type": "sent",
                "amount": float(amount.replace(",", "")),
                "user": user.strip().title(),
                "date": date,
                "subject": subject,
            })

    return results


def aggregate(transactions: list[dict], breakdown: str) -> dict:
    """breakdown: 'user' | 'type'"""
    data: dict[str, dict] = {}

    for t in transactions:
        if breakdown == "user":
            key = t["user"]
        else:
            key = t["type"].capitalize()

        if key not in data:
            data[key] = {"received": 0.0, "sent": 0.0, "count": 0}
        data[key][t["type"]] += t["amount"]
        data[key]["count"] += 1

    return data


def load_and_parse(filepath: str) -> list[dict]:
    with open(filepath) as f:
        emails = json.load(f)
    return parse_emails(emails)
