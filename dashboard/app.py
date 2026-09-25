# Pitcher Valuation Dashboard
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Pitcher Valuation Dashboard",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
STUFF_PLUS_SUMMARY_PATH = DATA_DIR / "stuff_plus_pitch_types.csv"
STUFF_PLUS_PITCHES_PATH = DATA_DIR / "stuff_plus_pitches.csv"

# Every MLB team, by league -- (Statcast team code, full name). Lets the
# sidebar navigate the whole league even though only a handful of pitchers
# have real Stuff+ data loaded so far (see analysis/stuff_plus_proxy.py --pitcher).
# Codes match pybaseball/Statcast's home_team/away_team values exactly
# (confirmed against a real pull -- notably AZ, not ARI).
TEAMS_BY_LEAGUE = {
    "American League": [
        ("BAL", "Baltimore Orioles"), ("BOS", "Boston Red Sox"), ("NYY", "New York Yankees"),
        ("TB", "Tampa Bay Rays"), ("TOR", "Toronto Blue Jays"),
        ("CWS", "Chicago White Sox"), ("CLE", "Cleveland Guardians"), ("DET", "Detroit Tigers"),
        ("KC", "Kansas City Royals"), ("MIN", "Minnesota Twins"),
        ("HOU", "Houston Astros"), ("LAA", "Los Angeles Angels"), ("ATH", "Athletics"),
        ("SEA", "Seattle Mariners"), ("TEX", "Texas Rangers"),
    ],
    "National League": [
        ("ATL", "Atlanta Braves"), ("MIA", "Miami Marlins"), ("NYM", "New York Mets"),
        ("PHI", "Philadelphia Phillies"), ("WSH", "Washington Nationals"),
        ("CHC", "Chicago Cubs"), ("CIN", "Cincinnati Reds"), ("MIL", "Milwaukee Brewers"),
        ("PIT", "Pittsburgh Pirates"), ("STL", "St. Louis Cardinals"),
        ("AZ", "Arizona Diamondbacks"), ("COL", "Colorado Rockies"), ("LAD", "Los Angeles Dodgers"),
        ("SD", "San Diego Padres"), ("SF", "San Francisco Giants"),
    ],
}

# Statcast team code -> MLB Advanced Media's numeric team id, for team-logos.
# Pulled from https://statsapi.mlb.com/api/v1/teams?sportId=1 (the
# authoritative source) and cross-checked against the codes above.
TEAM_MLB_ID = {
    "LAA": 108, "AZ": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113,
    "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAD": 119,
    "WSH": 120, "NYM": 121, "ATH": 133, "PIT": 134, "SD": 135, "SEA": 136,
    "SF": 137, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "MIN": 142,
    "PHI": 143, "ATL": 144, "CWS": 145, "MIA": 146, "NYY": 147, "MIL": 158,
}


def team_logo_url(team_code: str) -> str:
    return f"https://www.mlbstatic.com/team-logos/{TEAM_MLB_ID[team_code]}.svg"


def player_headshot_url(mlbam_id: int) -> str:
    # "silo" is MLB's pre-cut transparent-background headshot variant
    # (confirmed: real alpha channel, not just a plain PNG).
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "w_213,q_auto:best,f_auto/"
        f"v1/people/{mlbam_id}/headshot/silo/current"
    )

# Fixed hue order (Okabe-Ito, colorblind-safe) so a pitch type keeps the same
# color everywhere it appears. Canonical Statcast pitch-type order, not
# alphabetical, so fastball variants group together.
PITCH_TYPE_ORDER = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "FO", "SV"]
PITCH_TYPE_COLORS = dict(zip(PITCH_TYPE_ORDER, [
    "#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00",
    "#CC79A7", "#F0E442", "#000000", "#999999", "#882255", "#44AA99",
]))
PITCH_TYPE_NAMES = {
    "FF": "Four-Seam Fastball", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle Curve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball", "SV": "Slurve",
    "EP": "Eephus", "KN": "Knuckleball", "SC": "Screwball",
}

ABOVE_AVG_COLOR = "#0072B2"   # Stuff+ > 100
BELOW_AVG_COLOR = "#D55E00"   # Stuff+ < 100
BASELINE_COLOR = "#8A8F98"    # the 100 reference line


@st.cache_data
def load_real_stuff_plus():
    if not STUFF_PLUS_SUMMARY_PATH.exists():
        return None, None
    summary = pd.read_csv(STUFF_PLUS_SUMMARY_PATH)
    pitches = pd.read_csv(STUFF_PLUS_PITCHES_PATH) if STUFF_PLUS_PITCHES_PATH.exists() else pd.DataFrame()
    return summary, pitches


def pitch_order(pitch_types) -> list:
    known = [p for p in PITCH_TYPE_ORDER if p in set(pitch_types)]
    unknown = sorted(set(pitch_types) - set(known))
    return known + unknown


def movement_chart(pitches: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for pitch_type in pitch_order(pitches["PitchType"].unique()):
        group = pitches[pitches["PitchType"] == pitch_type]
        fig.add_trace(go.Scatter(
            x=group["hb"], y=group["ivb"],
            mode="markers",
            name=pitch_type,
            marker=dict(size=8, color=PITCH_TYPE_COLORS.get(pitch_type, "#888"), line=dict(width=0)),
            hovertemplate="%{fullData.name}<br>Velo: %{customdata[0]:.1f} mph<br>Stuff+: %{customdata[1]:.0f}<extra></extra>",
            customdata=group[["release_speed", "StuffPlus"]].values,
        ))
    fig.update_layout(
        xaxis_title="Horizontal break (in, catcher's view)",
        yaxis_title="Induced vertical break (in)",
        legend_title_text="Pitch type",
        margin=dict(t=10, l=10, r=10, b=10),
        height=380,
    )
    return fig


def stuff_plus_bar_chart(summary_for_pitcher: pd.DataFrame) -> go.Figure:
    ordered = summary_for_pitcher.set_index("PitchType").loc[
        [p for p in pitch_order(summary_for_pitcher["PitchType"]) if p in summary_for_pitcher["PitchType"].values]
    ].reset_index()

    colors = [ABOVE_AVG_COLOR if v >= 100 else BELOW_AVG_COLOR for v in ordered["StuffPlus"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ordered["PitchType"], y=ordered["StuffPlus"],
        marker=dict(color=colors),
        text=ordered["StuffPlus"].round(0).astype(int),
        textposition="outside",
        hovertemplate="%{x}: Stuff+ %{y:.1f}<br>%{customdata} pitches<extra></extra>",
        customdata=ordered["Pitches"],
    ))
    fig.add_hline(y=100, line_dash="dash", line_color=BASELINE_COLOR, annotation_text="League avg (100)")
    fig.update_layout(
        yaxis_title="Stuff+ (100 = league avg whiff prob.)",
        xaxis_title="Pitch type",
        showlegend=False,
        margin=dict(t=30, l=10, r=10, b=10),
        height=380,
    )
    return fig


stuff_plus_summary, stuff_plus_pitches = load_real_stuff_plus()

st.sidebar.title("Filters")

league_choice = st.sidebar.selectbox("League", list(TEAMS_BY_LEAGUE.keys()))
team_options = TEAMS_BY_LEAGUE[league_choice]
team_labels = [f"{name} ({code})" for code, name in team_options]
team_label_choice = st.sidebar.selectbox("Team", team_labels)
team_choice, team_name_choice = team_options[team_labels.index(team_label_choice)]

st.sidebar.image(team_logo_url(team_choice), width=120)

team_pitchers = (
    sorted(stuff_plus_summary.loc[stuff_plus_summary["PitcherTeam"] == team_choice, "Pitcher"].unique())
    if stuff_plus_summary is not None else []
)

if team_pitchers:
    pitcher_choice = st.sidebar.selectbox("Pitcher", team_pitchers)
else:
    pitcher_choice = None
    st.sidebar.caption(
        f"No Stuff+ data loaded for the {team_name_choice} yet. Run "
        "`python -m analysis.stuff_plus_proxy --pitcher \"<name>\"` for a "
        "pitcher on this team."
    )


# Main page header
st.title("Pitcher Valuation & Mispricing Dashboard")
st.caption(
    "Goal is identify pitchers whose volatility, not raw talent, is "
    "being mispriced by the market."
)

tab_report, tab_board = st.tabs(["Pitcher Report", "League Mispricing Board"])


# Tab 1: single-pitcher report
with tab_report:
    if pitcher_choice is None:
        st.subheader(f"{team_name_choice} ({team_choice})")
        st.info(f"No pitcher on the {team_name_choice} has Stuff+ data loaded yet. Pick a team with data loaded (e.g. Pittsburgh Pirates) from the sidebar.")
    else:
        pitcher_id = int(stuff_plus_summary.loc[stuff_plus_summary["Pitcher"] == pitcher_choice, "PitcherId"].iloc[0])

        photo_col, logo_col, text_col = st.columns([1, 1, 4])
        with photo_col:
            st.image(player_headshot_url(pitcher_id), width=120)
        with logo_col:
            st.image(team_logo_url(team_choice), width=90)
        with text_col:
            st.subheader(pitcher_choice)
            st.caption(team_name_choice)

        overall_stuff_plus = stuff_plus_summary.loc[
            stuff_plus_summary["Pitcher"] == pitcher_choice, "OverallStuffPlus"
        ].iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Stuff+ (proxy)", f"{overall_stuff_plus:.1f}")
        col2.metric("Command+ (proxy)", "--")
        col3.metric("Volatility percentile", "--")
        col4.metric("Risk-adj. WAR proxy", "--")
        col5.metric("Surplus value", "--")

        st.info(
            "Stuff+ is real: trained on the full 2025 league-wide season, then "
            "scored against this pitcher's own full 2025 season -- including time "
            "with another team if they were traded, since a pitcher shown here is "
            "graded on their whole season, not just pitches thrown as a "
            f"{team_name_choice}. The other four metrics come from analysis "
            "modules not run on real data yet."
        )

        pitcher_pitches = stuff_plus_pitches[stuff_plus_pitches["Pitcher"] == pitcher_choice]
        pitcher_summary = stuff_plus_summary[stuff_plus_summary["Pitcher"] == pitcher_choice]

        left, right = st.columns(2)
        with left:
            st.markdown("**Pitch movement**")
            st.plotly_chart(movement_chart(pitcher_pitches), width='stretch')

        with right:
            st.markdown("**Velocity by start (placeholder chart)**")
            st.caption("line chart of velocity across starts.")
            st.empty()

        st.markdown("**Stuff+ by pitch type**")
        st.plotly_chart(stuff_plus_bar_chart(pitcher_summary), width='stretch')

        used_types = pitch_order(
            set(pitcher_pitches["PitchType"].unique()) | set(pitcher_summary["PitchType"].unique())
        )
        key_line = " · ".join(f"**{p}** {PITCH_TYPE_NAMES.get(p, 'Unknown')}" for p in used_types)
        st.caption(f"Pitch type key: {key_line}")


# Tab 2: whole-roster ranking table
with tab_board:
    st.markdown(
        "Ranked by estimated surplus value (how much a pitcher's "
        "risk-adjusted production is worth compared to what they're paid)."
    )

    if stuff_plus_summary is not None:
        loaded = (
            stuff_plus_summary[["Pitcher", "PitcherTeam", "OverallStuffPlus"]]
            .drop_duplicates()
            .rename(columns={"PitcherTeam": "Team", "OverallStuffPlus": "Stuff+"})
            .sort_values("Stuff+", ascending=False)
        )
        loaded["Command+"] = "--"
        loaded["Risk-adj. WAR"] = "--"
        loaded["Salary ($M)"] = "--"
        loaded["Surplus ($M)"] = "--"
        st.caption(
            "Only Stuff+ has been run on real data so far -- the other columns "
            "need the Command Proxy, Volatility, and Risk-Adjusted WAR modules "
            "run league-wide (see README known limitations)."
        )
        st.dataframe(loaded[["Pitcher", "Team", "Stuff+", "Command+", "Risk-adj. WAR", "Salary ($M)", "Surplus ($M)"]], width='stretch')
    else:
        st.caption("No pitchers loaded yet.")

    st.markdown("**League mispricing map (placeholder chart)**")
    st.caption("bubble chart of Stuff+ vs Command+, sized by WAR and colored by surplus value.")
    st.empty()
