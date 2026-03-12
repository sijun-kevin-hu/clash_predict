# card_api.py

import requests
import json
from pathlib import Path

BASE_URL = "https://api.clashroyale.com/v1"
CARD_CACHE = Path("processed/card_list.json")

def fetch_cards(api_token: str) -> list[str]:
    """
    Fetch the latest card list from the Clash Royale API.
    Caches locally so you don't need to download every time.
    """

    # If cached file exists, load and return it
    if CARD_CACHE.exists():
        with open(CARD_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{BASE_URL}/cards"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # Extract card names from "items"
    items = data.get("items", [])
    card_names = [card["name"] for card in items if "name" in card]
    card_names.sort()  # for stable ordering

    # cache the result
    with open(CARD_CACHE, "w", encoding="utf-8") as f:
        json.dump(card_names, f, indent=2)

    return card_names