"""
Clash Royale Matchup Predictor — Streamlit app.

Enter two player tags, fetch their current decks from the API,
and predict the win probability using the trained XGBoost model.
"""

import os
import streamlit as st
import requests
from dotenv import load_dotenv

from inference.predict import predict_matchup

load_dotenv(".env.local")

BASE_URL = "https://api.clashroyale.com/v1"


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def fetch_player(player_tag: str) -> dict:
    """Fetch a player profile from the Clash Royale API."""
    token = os.environ.get("CLASH_ROYALE_API_TOKEN", "")
    if not token:
        st.error("Set CLASH_ROYALE_API_TOKEN in .env.local")
        st.stop()

    encoded = player_tag.strip().replace("#", "%23")
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


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="CR Matchup Predictor", page_icon="\u2694\ufe0f")
    st.title("\u2694\ufe0f Clash Royale Matchup Predictor")
    st.markdown("Enter two player tags to predict who has the edge based on "
                "current decks, card levels, and trophies.")

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
                p_loss, p_win = predict_matchup(team_data, opp_data)
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
        st.progress(p_win)
        if p_win >= 0.55:
            st.success("Model favors you!")
        elif p_win <= 0.45:
            st.error("Model favors your opponent.")
        else:
            st.info("Close matchup — could go either way.")


if __name__ == "__main__":
    main()
