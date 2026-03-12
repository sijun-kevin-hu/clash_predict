# preprocess_cards.py

import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from util.card_api import fetch_cards

load_dotenv(".env.local")

RAW_PATH = Path("raw_data/battles.json1")
OUT_PATH = Path("processed/battles_cards.csv")
OUT_PATH.parent.mkdir(exist_ok=True)

def load_card_list():
    return fetch_cards(os.environ["CLASH_ROYALE_API_TOKEN"])

def create_one_hot(deck_cards, card_list, prefix):
    """
    Create one-hot presence flags for a deck.
    """
    row = {}
    card_names = [c.get("name") for c in deck_cards]
    for card in card_list:
        key = f"{prefix}_has_{card.replace(' ', '_')}"
        row[key] = 1 if card in card_names else 0
    return row

def preprocess_battle(record, card_list):
    battle = record["battle"]

    # You can include your existing filtering (e.g., 1v1 ladder)
    team_list = battle.get("team", [])
    opp_list = battle.get("opponent", [])
    if len(team_list) != 1 or len(opp_list) != 1:
        return None

    team = team_list[0]
    opp = opp_list[0]

    team_cards = team.get("cards", [])
    opp_cards = opp.get("cards", [])

    # Basic numeric features
    row = {
        "team_trophies": team.get("startingTrophies"),
        "opp_trophies": opp.get("startingTrophies"),
        "trophy_diff": (team.get("startingTrophies") or 0) - (opp.get("startingTrophies") or 0),
        "team_avg_elixir": sum(c.get("elixirCost", 0) for c in team_cards) / len(team_cards),
        "opp_avg_elixir": sum(c.get("elixirCost", 0) for c in opp_cards) / len(opp_cards),
    }

    # One-hot card features
    row.update(create_one_hot(team_cards, card_list, "team"))
    row.update(create_one_hot(opp_cards, card_list, "opp"))

    # Optional interaction features
    for card in card_list:
        team_key = f"team_has_{card.replace(' ', '_')}"
        opp_key = f"opp_has_{card.replace(' ', '_')}"
        row[f"{card.replace(' ', '_')}_diff"] = row.get(team_key, 0) - row.get(opp_key, 0)
        row[f"{card.replace(' ', '_')}_both"] = row.get(team_key, 0) * row.get(opp_key, 0)

    # Label win/loss
    team_crowns = team.get("crowns", 0)
    opp_crowns = opp.get("crowns", 0)
    row["label"] = 1 if team_crowns > opp_crowns else 0

    return row

def main():
    card_list = load_card_list()
    print("Loaded", len(card_list), "cards.")

    rows = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            row = preprocess_battle(record, card_list)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    print("Processed rows:", len(df))
    df.to_csv(OUT_PATH, index=False)
    print("Saved enhanced dataset with card presence to", OUT_PATH)

if __name__ == "__main__":
    main()