"""
Clash Royale Matchup Predictor — Streamlit app.

Enter two player tags, fetch their current decks from the API,
and predict the win probability using the trained XGBoost model.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

from inference.predict import predict_matchup
from preprocessing.preprocessing import ROLE_CATEGORIES

load_dotenv(".env.local")

BASE_URL = "https://api.clashroyale.com/v1"


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def _get_api_token() -> str:
    """Return the Clash Royale API token from st.secrets or environment."""
    try:
        return st.secrets["CLASH_ROYALE_API_TOKEN"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("CLASH_ROYALE_API_TOKEN", "")


def fetch_player(player_tag: str) -> dict:
    """Fetch a player profile from the Clash Royale API."""
    token = _get_api_token()
    if not token:
        st.error(
            "API token not found. "
            "Add `CLASH_ROYALE_API_TOKEN` to `.streamlit/secrets.toml` "
            "(Streamlit Cloud) or `.env.local` (local dev)."
        )
        st.stop()

    tag = player_tag.strip().lstrip("#").upper()
    encoded = "%23" + tag
    url = f"{BASE_URL}/players/{encoded}"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_deck(player_data: dict, label: str):
    """Render a player's current deck in a compact grid."""
    st.subheader(f"{label}: {player_data.get('name', '???')}")
    st.caption(f"Trophies: {player_data.get('trophies', 0)}")

    cards = player_data.get("currentDeck", [])
    if not cards:
        st.warning("No current deck found.")
        return

    cols = st.columns(4)
    for i, card in enumerate(cards):
        with cols[i % 4]:
            icon_url = card.get("iconUrls", {}).get("medium", "")
            if icon_url:
                st.image(icon_url, width=64)
            st.caption(f"{card.get('name')}  Lv.{card.get('level', '?')}")

    support = player_data.get("currentDeckSupportCards", [])
    if support:
        st.markdown("**Support cards:** "
                     + ", ".join(f"{s.get('name')} Lv.{s.get('level','?')}"
                                for s in support))


def _pretty_feature_name(feat: str) -> str:
    """Turn a raw feature column name into a readable label."""
    feat = feat.replace("_", " ")
    for prefix in ("team norm level ", "opp norm level ",
                    "team support norm level ", "opp support norm level "):
        if feat.startswith(prefix):
            card = feat[len(prefix):]
            side = "You" if feat.startswith("team") else "Opp"
            kind = "support " if "support" in prefix else ""
            return f"{side}: {kind}{card} level"
    for prefix in ("team ", "opp "):
        if feat.startswith(prefix) and feat.endswith(" count"):
            role = feat[len(prefix):-len(" count")]
            side = "You" if prefix == "team " else "Opp"
            return f"{side}: {role} count"
    return feat


def show_top_features(feature_df: pd.DataFrame, model, n: int = 10):
    """Show the top features driving the prediction for this matchup."""
    row = feature_df.iloc[0]
    importances = model.feature_importances_
    feat_names = model.feature_names_in_

    # Weight = global importance * this matchup's feature value
    contributions = importances * np.abs(row.values)
    contrib_df = pd.DataFrame({
        "feature": feat_names,
        "value": row.values,
        "importance": importances,
        "contribution": contributions,
    })
    # Only keep features that are non-zero in this matchup
    contrib_df = contrib_df[contrib_df["value"] != 0]
    contrib_df = contrib_df.sort_values("contribution", ascending=False).head(n)
    contrib_df["label"] = contrib_df["feature"].apply(_pretty_feature_name)

    chart_df = contrib_df.set_index("label")["contribution"].sort_values(ascending=True)
    st.bar_chart(chart_df)


def show_elixir_comparison(team_data: dict, opp_data: dict):
    """Side-by-side elixir cost breakdown."""
    team_cards = team_data.get("currentDeck", [])
    opp_cards = opp_data.get("currentDeck", [])

    team_costs = sorted(c.get("elixirCost", 0) for c in team_cards)
    opp_costs = sorted(c.get("elixirCost", 0) for c in opp_cards)

    # Build histogram-style comparison
    all_costs = sorted(set(team_costs + opp_costs))
    hist = pd.DataFrame({
        "You": [team_costs.count(c) for c in all_costs],
        "Opponent": [opp_costs.count(c) for c in all_costs],
    }, index=[f"{c}" for c in all_costs])
    hist.index.name = "Elixir"

    st.bar_chart(hist)

    team_avg = np.mean(team_costs) if team_costs else 0
    opp_avg = np.mean(opp_costs) if opp_costs else 0
    c1, c2 = st.columns(2)
    c1.metric("Your avg elixir", f"{team_avg:.1f}")
    c2.metric("Opp avg elixir", f"{opp_avg:.1f}")


def show_level_comparison(team_data: dict, opp_data: dict):
    """Compare card levels between the two decks."""
    team_cards = team_data.get("currentDeck", [])
    opp_cards = opp_data.get("currentDeck", [])

    team_levels = [c.get("level", 0) / c.get("maxLevel", 1) for c in team_cards]
    opp_levels = [c.get("level", 0) / c.get("maxLevel", 1) for c in opp_cards]

    team_avg = np.mean(team_levels) if team_levels else 0
    opp_avg = np.mean(opp_levels) if opp_levels else 0

    team_names = [c.get("name", "?") for c in team_cards]
    opp_names = [c.get("name", "?") for c in opp_cards]

    df = pd.DataFrame({
        "Card": team_names + opp_names,
        "Normalized Level": team_levels + opp_levels,
        "Player": ["You"] * len(team_names) + ["Opponent"] * len(opp_names),
    })

    c1, c2 = st.columns(2)
    c1.metric("Your avg level %", f"{team_avg:.0%}")
    c2.metric("Opp avg level %", f"{opp_avg:.0%}")

    # Show as a table grouped by player
    left, right = st.columns(2)
    with left:
        t = df[df["Player"] == "You"][["Card", "Normalized Level"]].reset_index(drop=True)
        t["Normalized Level"] = t["Normalized Level"].apply(lambda x: f"{x:.0%}")
        st.dataframe(t, use_container_width=True, hide_index=True)
    with right:
        o = df[df["Player"] == "Opponent"][["Card", "Normalized Level"]].reset_index(drop=True)
        o["Normalized Level"] = o["Normalized Level"].apply(lambda x: f"{x:.0%}")
        st.dataframe(o, use_container_width=True, hide_index=True)


def show_role_comparison(team_data: dict, opp_data: dict):
    """Compare deck archetype roles between the two decks."""
    from preprocessing.preprocessing import load_card_roles, create_deck_balance_features

    card_roles = load_card_roles()
    team_cards = team_data.get("currentDeck", [])
    opp_cards = opp_data.get("currentDeck", [])

    team_roles = create_deck_balance_features(team_cards, card_roles, "team")
    opp_roles = create_deck_balance_features(opp_cards, card_roles, "opp")

    labels = [r.replace("_", " ").title() for r in ROLE_CATEGORIES]
    team_vals = [team_roles.get(f"team_{r.replace(' ', '_')}_count", 0) for r in ROLE_CATEGORIES]
    opp_vals = [opp_roles.get(f"opp_{r.replace(' ', '_')}_count", 0) for r in ROLE_CATEGORIES]

    role_df = pd.DataFrame({
        "Role": labels,
        "You": team_vals,
        "Opponent": opp_vals,
    }).set_index("Role")

    st.bar_chart(role_df)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="CR Matchup Predictor", page_icon="\u2694\ufe0f",
                       layout="wide")
    st.title("\u2694\ufe0f Clash Royale Matchup Predictor")
    st.markdown("Enter two player tags to predict who has the edge based on "
                "current decks, card levels, and trophies.")
    st.info("This model is intended for players with close trophy counts. "
            "Predictions may be less accurate for matchups with large trophy differences.")

    col1, col2 = st.columns(2)
    with col1:
        team_tag = st.text_input("Your player tag", placeholder="#ABC123")
    with col2:
        opp_tag = st.text_input("Opponent player tag", placeholder="#XYZ789")

    if st.button("Predict", type="primary"):
        if not team_tag or not opp_tag:
            st.warning("Please enter both player tags.")
            return

        with st.spinner("Fetching player data..."):
            try:
                team_data = fetch_player(team_tag)
                opp_data = fetch_player(opp_tag)
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
                return
            except Exception as e:
                st.error(f"Failed to fetch player data: {e}")
                return

        # Show decks side by side
        left, right = st.columns(2)
        with left:
            show_deck(team_data, "You")
        with right:
            show_deck(opp_data, "Opponent")

        st.divider()

        # Predict
        with st.spinner("Running prediction..."):
            try:
                p_loss, p_win, feature_df, model = predict_matchup(team_data, opp_data)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        # Display results
        st.subheader("Prediction")

        res_left, res_mid, res_right = st.columns(3)
        with res_left:
            st.metric("Win %", f"{p_win:.1%}")
        with res_mid:
            st.metric("Loss %", f"{p_loss:.1%}")
        with res_right:
            trophy_diff = team_data.get("trophies", 0) - opp_data.get("trophies", 0)
            st.metric("Trophy diff", f"{trophy_diff:+d}")

        # Color-coded progress bar
        st.progress(float(p_win))
        if p_win >= 0.55:
            st.success("Model favors you!")
        elif p_win <= 0.45:
            st.error("Model favors your opponent.")
        else:
            st.info("Close matchup — could go either way.")

        # -------------------------------------------------------------------
        # Explainability visualizations
        # -------------------------------------------------------------------
        st.divider()
        st.subheader("Why this prediction?")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Top Features", "Deck Roles", "Elixir Curve", "Card Levels",
        ])

        with tab1:
            st.markdown("Features that mattered most for **this specific matchup**, "
                        "weighted by the model's global importance and each feature's value.")
            show_top_features(feature_df, model, n=12)

        with tab2:
            st.markdown("How each deck breaks down by archetype role.")
            show_role_comparison(team_data, opp_data)

        with tab3:
            st.markdown("Elixir cost distribution for each deck.")
            show_elixir_comparison(team_data, opp_data)

        with tab4:
            st.markdown("Normalized card levels (level / max level) for each deck.")
            show_level_comparison(team_data, opp_data)


if __name__ == "__main__":
    main()
