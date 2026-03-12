# predict.py
import json
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/clash_model.joblib")
CARD_CACHE = Path("processed/card_list.json")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def load_card_list():
    if not CARD_CACHE.exists():
        raise FileNotFoundError(
            f"Card list not found at {CARD_CACHE}. Run preprocessing.py first."
        )
    with open(CARD_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_features(team_trophies, opp_trophies, team_cards, opp_cards, card_list):
    """
    Build the same feature vector as preprocess_battle() in preprocessing.py.

    Args:
        team_trophies / opp_trophies: int
        team_cards / opp_cards: list of dicts with "name" and "elixirCost"
        card_list: list of card name strings (from card cache)
    """
    team_card_names = [c["name"] for c in team_cards]
    opp_card_names  = [c["name"] for c in opp_cards]

    row = {
        "team_trophies":   team_trophies,
        "opp_trophies":    opp_trophies,
        "trophy_diff":     team_trophies - opp_trophies,
        "team_avg_elixir": sum(c.get("elixirCost", 0) for c in team_cards) / len(team_cards),
        "opp_avg_elixir":  sum(c.get("elixirCost", 0) for c in opp_cards)  / len(opp_cards),
    }

    # One-hot card presence (mirrors create_one_hot in preprocessing.py)
    for card in card_list:
        key = card.replace(" ", "_")
        row[f"team_has_{key}"] = 1 if card in team_card_names else 0
        row[f"opp_has_{key}"]  = 1 if card in opp_card_names  else 0

    # Interaction features (mirrors preprocessing.py loop)
    for card in card_list:
        key = card.replace(" ", "_")
        row[f"{key}_diff"] = row[f"team_has_{key}"] - row[f"opp_has_{key}"]
        row[f"{key}_both"] = row[f"team_has_{key}"] * row[f"opp_has_{key}"]

    return row


def predict_win_prob(team_trophies, opp_trophies, team_cards, opp_cards):
    """
    Predict win probability for a matchup.

    Args:
        team_trophies: int — team's starting trophies
        opp_trophies:  int — opponent's starting trophies
        team_cards:    list of dicts with keys "name" and "elixirCost"
        opp_cards:     list of dicts with keys "name" and "elixirCost"

    Returns:
        (P(loss), P(win)) tuple of floats
    """
    card_list = load_card_list()
    features  = build_features(team_trophies, opp_trophies, team_cards, opp_cards, card_list)

    df = pd.DataFrame([features])

    model = load_model()
    # Align to the exact columns the model was trained on
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    proba = model.predict_proba(df)[0]
    return proba[0], proba[1]  # (P(loss), P(win))


if __name__ == "__main__":
    # Example: Hog Rider cycle deck vs. Golem beatdown
    team_cards = [
        {"name": "Hog Rider",       "elixirCost": 4},
        {"name": "Musketeer",       "elixirCost": 4},
        {"name": "Valkyrie",        "elixirCost": 4},
        {"name": "Skeletons",       "elixirCost": 1},
        {"name": "Ice Spirit",      "elixirCost": 1},
        {"name": "The Log",         "elixirCost": 2},
        {"name": "Fireball",        "elixirCost": 4},
        {"name": "Cannon",          "elixirCost": 3},
    ]
    opp_cards = [
        {"name": "Golem",           "elixirCost": 8},
        {"name": "Baby Dragon",     "elixirCost": 4},
        {"name": "Mega Minion",     "elixirCost": 3},
        {"name": "Dark Prince",     "elixirCost": 4},
        {"name": "Mega Knight",     "elixirCost": 7},
        {"name": "Lightning",       "elixirCost": 6},
        {"name": "Zap",             "elixirCost": 2},
        {"name": "Elixir Collector","elixirCost": 6},
    ]

    pl, pw = predict_win_prob(
        team_trophies=7500,
        opp_trophies=7500,
        team_cards=team_cards,
        opp_cards=opp_cards,
    )
    print("P(loss) =", pl)
    print("P(win)  =", pw)
