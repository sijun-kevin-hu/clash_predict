# predict.py
import json
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/xgb_clash_model.joblib")
CARD_CACHE = Path("data/processed/card_list.json")
SUPPORT_CACHE = Path("data/processed/support_list.json")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run models/train.py first.")
    return joblib.load(MODEL_PATH)


def load_card_list():
    if not CARD_CACHE.exists():
        raise FileNotFoundError(
            f"Card list not found at {CARD_CACHE}. Run preprocessing/preprocessing.py first."
        )
    with open(CARD_CACHE, "r", encoding="utf-8") as f:
        data = json.load(f)
        # card_list.json maps card names to types; extract just the names
        if isinstance(data, dict):
            return sorted(data.keys())
        return data

def load_support_list():
    if not SUPPORT_CACHE.exists():
        raise FileNotFoundError(
            f"Support list not found at {SUPPORT_CACHE}. Run preprocessing/preprocessing.py first."
        )
    with open(SUPPORT_CACHE, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_card_feature(deck_cards, card_list, prefix):
    """Mirrors create_card_feature() in preprocessing/preprocessing.py."""
    row = {}
    card_norm_level_map = {c.get("name"): c.get("level", 0) / c.get("maxLevel", 1) for c in deck_cards}
    for card in card_list:
        key = f"{prefix}_norm_level_{card.replace(' ', '_')}"
        row[key] = card_norm_level_map[card] if card in card_norm_level_map else 0
    return row


def create_support_card_feature(support_cards, support_list, prefix):
    """Mirrors create_support_card_feature() in preprocessing/preprocessing.py."""
    row = {}
    support_norm_level_map = {c.get("name"): c.get("level", 0) / c.get("maxLevel", 1) for c in support_cards}
    for support in support_list:
        key = f"{prefix}_support_norm_level_{support.replace(' ', '_')}"
        row[key] = support_norm_level_map[support] if support in support_norm_level_map else 0
    return row


def build_features(team_trophies, opp_trophies, team_cards, opp_cards, team_support, opp_support, card_list, support_list):
    """
    Build the same feature vector as preprocess_battle() in preprocessing/preprocessing.py.

    Args:
        team_trophies / opp_trophies: int
        team_cards / opp_cards: list of dicts with "name", "elixirCost", "level", "maxLevel"
        team_support / opp_support: list of dicts with "name", "level", "maxLevel"
        card_list: sorted list of card name strings
        support_list: sorted list of support card name strings
    """
    row = {
        "team_trophies":   team_trophies,
        "opp_trophies":    opp_trophies,
        "trophy_diff":     team_trophies - opp_trophies,
        "team_avg_elixir": sum(c.get("elixirCost", 0) for c in team_cards) / len(team_cards) if team_cards else 0,
        "opp_avg_elixir":  sum(c.get("elixirCost", 0) for c in opp_cards) / len(opp_cards) if opp_cards else 0,
    }

    row.update(create_card_feature(team_cards, card_list, "team"))
    row.update(create_card_feature(opp_cards, card_list, "opp"))
    row.update(create_support_card_feature(team_support, support_list, "team"))
    row.update(create_support_card_feature(opp_support, support_list, "opp"))

    return row


def predict_win_prob(team_trophies, opp_trophies, team_cards, opp_cards, team_support=None, opp_support=None):
    """
    Predict win probability for a matchup.

    Args:
        team_trophies: int — team's starting trophies
        opp_trophies:  int — opponent's starting trophies
        team_cards:    list of dicts with keys "name", "elixirCost", "level", "maxLevel"
        opp_cards:     list of dicts with keys "name", "elixirCost", "level", "maxLevel"
        team_support:  list of dicts with keys "name", "level", "maxLevel" (optional)
        opp_support:   list of dicts with keys "name", "level", "maxLevel" (optional)

    Returns:
        (P(loss), P(win)) tuple of floats
    """
    card_list = load_card_list()
    support_list = load_support_list()

    features = build_features(
        team_trophies, opp_trophies,
        team_cards, opp_cards,
        team_support or [], opp_support or [],
        card_list, support_list,
    )

    df = pd.DataFrame([features])

    model = load_model()
    # Align to the exact columns the model was trained on
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    proba = model.predict_proba(df)[0]
    return proba[0], proba[1]  # (P(loss), P(win))


def predict_matchup(team_player, opp_player):
    """
    Predict win probability from two raw Clash Royale player API responses.

    Uses preprocess_matchup from preprocessing to build features, then runs
    the trained model.

    Args:
        team_player: dict — full player API response (must have currentDeck,
                     currentDeckSupportCards, trophies)
        opp_player:  dict — same structure for opponent

    Returns:
        (P(loss), P(win)) tuple of floats
    """
    from preprocessing.preprocessing import preprocess_matchup, load_card_roles

    card_list = load_card_list()
    support_list = load_support_list()
    card_roles = load_card_roles()

    df = preprocess_matchup(team_player, opp_player, card_list, support_list, card_roles)

    model = load_model()
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    proba = model.predict_proba(df)[0]
    return proba[0], proba[1]  # (P(loss), P(win))


if __name__ == "__main__":
    # Example: Hog Rider cycle deck vs. Golem beatdown
    team_cards = [
        {"name": "Hog Rider",        "elixirCost": 4, "level": 11, "maxLevel": 11},
        {"name": "Musketeer",        "elixirCost": 4, "level": 11, "maxLevel": 11},
        {"name": "Valkyrie",         "elixirCost": 4, "level": 11, "maxLevel": 11},
        {"name": "Skeletons",        "elixirCost": 1, "level": 11, "maxLevel": 11},
        {"name": "Ice Spirit",       "elixirCost": 1, "level": 11, "maxLevel": 11},
        {"name": "The Log",          "elixirCost": 2, "level": 11, "maxLevel": 11},
        {"name": "Fireball",         "elixirCost": 4, "level": 11, "maxLevel": 11},
        {"name": "Cannon",           "elixirCost": 3, "level": 11, "maxLevel": 11},
    ]
    opp_cards = [
        {"name": "Golem",            "elixirCost": 8, "level": 11, "maxLevel": 11},
        {"name": "Baby Dragon",      "elixirCost": 4, "level": 11, "maxLevel": 11},
        {"name": "Mega Minion",      "elixirCost": 3, "level": 11, "maxLevel": 11},
        {"name": "Dark Prince",      "elixirCost": 4, "level": 11, "maxLevel": 11},
        {"name": "Mega Knight",      "elixirCost": 7, "level": 11, "maxLevel": 11},
        {"name": "Lightning",        "elixirCost": 6, "level": 11, "maxLevel": 11},
        {"name": "Zap",              "elixirCost": 2, "level": 11, "maxLevel": 11},
        {"name": "Elixir Collector", "elixirCost": 6, "level": 11, "maxLevel": 11},
    ]

    pl, pw = predict_win_prob(
        team_trophies=7500,
        opp_trophies=7500,
        team_cards=team_cards,
        opp_cards=opp_cards,
    )
    print("P(loss) =", pl)
    print("P(win)  =", pw)
