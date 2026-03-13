# scrape_royaleapi_stats.py

import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path

# Where to save the scraped stats
OUT_FILE = Path("data/processed/card_stats.json")

# RoyaleAPI Ladder card stats URL
ROYALEAPI_STATS_URL = "https://royaleapi.com/cards/popular?cat=Ladder&lang=en&mode=grid&sort=usage&time=7d"

def scrape_card_stats():
    """
    Scrape card usage and win rate from RoyaleAPI Ladder card stats page.
    Stores a simple mapping: { card_name: { "usage": float, "win_rate": float } }
    """

    print(f"Fetching card stats from {ROYALEAPI_STATS_URL} …")
    resp = requests.get(ROYALEAPI_STATS_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # This site uses dynamic rendering — we look for <script> tags that include stats JSON
    # RoyaleAPI embeds JSON in a <script id="__NEXT_DATA__"> block
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if not script_tag:
        print("Failed to find embedded JSON data.")
        return {}

    # Parse JSON
    data = json.loads(script_tag.string)
    card_list = []

    # Traverse JSON to find card entries; RoyaleAPI structure varies per region
    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "cards" and isinstance(v, list):
                    card_list.extend(v)
                else:
                    _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)

    _search(data)

    stats_map = {}
    print(f"Found {len(card_list)} card entries.")

    for card in card_list:
        name = card.get("name")
        # Some cards include stats
        usage = card.get("usageRate")
        win_rate = card.get("winRate")

        if name and usage is not None and win_rate is not None:
            stats_map[name] = {
                "usage": usage,
                "win_rate": win_rate
            }

    if stats_map:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(stats_map, f, indent=2)
        print(f"Saved card stats to {OUT_FILE}")

    return stats_map


if __name__ == "__main__":
    scrape_card_stats()
