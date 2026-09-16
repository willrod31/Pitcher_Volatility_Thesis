# Pitcher Valuation Dashboard

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Pitcher Valuation Dashboard",
    layout="wide",
)

PLACEHOLDER_TEAMS = {
    "American League": ["Team A", "Team B", "Team C"],
    "National League": ["Team D", "Team E", "Team F"],
}
PLACEHOLDER_PITCHERS = ["Pitcher 1", "Pitcher 2", "Pitcher 3"]

st.sidebar.title("Filters")

league_choice = st.sidebar.selectbox("League", list(PLACEHOLDER_TEAMS.keys()))
team_choice = st.sidebar.selectbox("Team", PLACEHOLDER_TEAMS[league_choice])
pitcher_choice = st.sidebar.selectbox("Pitcher", PLACEHOLDER_PITCHERS)


# Main page header
st.title("Pitcher Valuation & Mispricing Dashboard")
st.caption(
    "Goal is identify pitchers whose volatility, not raw talent, is "
    "being mispriced by the market."
)

tab_report, tab_board = st.tabs(["Pitcher Report", "League Mispricing Board"])


# Tab 1: single-pitcher report
with tab_report:
    st.subheader(f"{pitcher_choice} - {team_choice}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stuff+ (proxy)", "--")
    col2.metric("Command+ (proxy)", "--")
    col3.metric("Volatility percentile", "--")
    col4.metric("Risk-adj. WAR proxy", "--")
    col5.metric("Surplus value", "--")

    st.info("verdict line goes here (e.g. \"Likely undervalued\").")

    left, right = st.columns(2)
    with left:
        st.markdown("**Pitch movement (placeholder chart)**")
        st.caption("scatter plot of horizontal vs. vertical break.")
        st.empty()

    with right:
        st.markdown("**Velocity by start (placeholder chart)**")
        st.caption("line chart of velocity across starts.")
        st.empty()

    st.markdown("**Stuff+ vs Command+ across the staff (placeholder chart)**")
    st.caption("scatter plot comparing every pitcher on the roster.")
    st.empty()


# Tab 2: whole-roster ranking table
with tab_board:
    st.markdown(
        "Ranked by estimated surplus value (how much a pitcher's "
        "risk-adjusted production is worth compared to what they're paid)."
    )

    placeholder_rows = [
        {"Pitcher": p, "Team": team_choice, "League": league_choice,
         "Stuff+": "--", "Command+": "--", "Risk-adj. WAR": "--",
         "Salary ($M)": "--", "Surplus ($M)": "--"}
        for p in PLACEHOLDER_PITCHERS
    ]
    st.dataframe(pd.DataFrame(placeholder_rows), use_container_width=True)

    st.markdown("**League mispricing map (placeholder chart)**")
    st.caption("bubble chart of Stuff+ vs Command+, sized by WAR and colored by surplus value.")
    st.empty()