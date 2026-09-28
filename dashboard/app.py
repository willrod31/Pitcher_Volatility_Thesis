# Pitcher Valuation Dashboard
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from colors import (  # noqa: E402  (dashboard-local helper module)
    BAD_COLOR, GOOD_COLOR, NEUTRAL_COLOR, NEUTRAL_TEXT_LIGHT, LeaguePools, colored_numbers, legend_html, rgba, stat_color, stat_html,
)

st.set_page_config(
    page_title="Pitcher Valuation Dashboard",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
STUFF_PLUS_SUMMARY_PATH = DATA_DIR / "stuff_plus_pitch_types.csv"
STUFF_PLUS_PITCHES_PATH = DATA_DIR / "stuff_plus_pitches.csv"
MOVEMENT_PITCHES_PATH = DATA_DIR / "movement_pitches.csv"   # every pitch, for the movement chart
LOCATION_PLUS_PATH = DATA_DIR / "location_plus_pitch_types.csv"  # from analysis/location_plus_proxy.py
REPORT_PATH = DATA_DIR / "pitcher_report_data.csv"          # from analysis/export_report_data.py
SURPLUS_BY_YEAR_PATH = DATA_DIR / "surplus_by_year.csv"     # from analysis/export_report_data.py

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

BASELINE_COLOR = "#8A8F98"    # the 100 reference line


@st.cache_data
def load_real_stuff_plus():
    if not STUFF_PLUS_SUMMARY_PATH.exists():
        return None, None
    summary = pd.read_csv(STUFF_PLUS_SUMMARY_PATH)
    # movement chart uses every pitch; fall back to scored swings if not exported yet
    path = MOVEMENT_PITCHES_PATH if MOVEMENT_PITCHES_PATH.exists() else STUFF_PLUS_PITCHES_PATH
    pitches = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return summary, pitches


@st.cache_data
def load_location_plus():
    """League-wide Location+ per pitcher x pitch type (+ overall), keyed on PitcherId. None until run."""
    return pd.read_csv(LOCATION_PLUS_PATH) if LOCATION_PLUS_PATH.exists() else None


@st.cache_data
def load_report_data():
    """Risk-adjusted value + multi-year surplus, keyed on PitcherId. (None, None) until exported."""
    if not REPORT_PATH.exists():
        return None, None
    report = pd.read_csv(REPORT_PATH)
    by_year = pd.read_csv(SURPLUS_BY_YEAR_PATH) if SURPLUS_BY_YEAR_PATH.exists() else pd.DataFrame()
    return report, by_year


# Same good/bad scale as every other number: blue = overpaid, gray = break-even, red = surplus.
SURPLUS_COLORSCALE = [[0.0, BAD_COLOR], [0.5, NEUTRAL_COLOR], [1.0, GOOD_COLOR]]


def value_scatter(team_table: pd.DataFrame) -> go.Figure:
    """x = risk-adjusted WAR, y = salary, size = years of control, color = multi-year surplus.

    Undervalued pitchers with long control land bottom-right as large blue points.
    """
    data = team_table.dropna(subset=["RiskAdjWAR", "Salary_M", "MultiYearSurplus_M"])
    limit = max(abs(data["MultiYearSurplus_M"]).max(), 1) if not data.empty else 1
    fig = go.Figure(go.Scatter(
        x=data["RiskAdjWAR"], y=data["Salary_M"],
        mode="markers+text",
        text=data["Pitcher"].str.split().str[-1],
        textposition="top center",
        textfont=dict(size=10, color="#555"),
        marker=dict(
            size=10 + 6 * data["ControlYearsUsed"],
            color=data["MultiYearSurplus_M"],
            colorscale=SURPLUS_COLORSCALE, cmin=-limit, cmax=limit,
            colorbar=dict(title="Multi-yr<br>surplus ($M)"),
            line=dict(width=2, color="white"),
        ),
        customdata=data[["Pitcher", "ControlYearsUsed", "MultiYearSurplus_M"]].values,
        hovertemplate=(
            "%{customdata[0]}<br>Risk-adj. WAR: %{x:.2f}<br>Salary: $%{y:.2f}M"
            "<br>Control years: %{customdata[1]:.0f}<br>Multi-yr surplus: $%{customdata[2]:.1f}M<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Risk-adjusted WAR (proxy)",
        yaxis_title="2025 salary ($M)",
        margin=dict(t=10, l=10, r=10, b=10),
        height=440,
    )
    return fig


def surplus_by_year_chart(pitcher_years: pd.DataFrame, pools: LeaguePools) -> go.Figure:
    """Each year's discounted surplus, colored vs. the league's single-season surplus pool."""
    values = pitcher_years["DiscountedSurplus"] / 1_000_000
    colors = [pools.color("SurplusCurrentSeason_M", v) or NEUTRAL_COLOR for v in values]
    fig = go.Figure(go.Bar(
        x=pitcher_years["Season"].astype(str), y=values,
        marker=dict(color=colors),
        text=[f"${v:.1f}M" for v in values], textposition="outside", cliponaxis=False,
        customdata=pitcher_years[["Stage", "Salary", "SalarySource"]].assign(Salary=pitcher_years["Salary"] / 1_000_000).values,
        hovertemplate=(
            "%{x} (%{customdata[0]})<br>Discounted surplus: $%{y:.1f}M"
            "<br>Salary: $%{customdata[1]:.2f}M (%{customdata[2]})<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_color=BASELINE_COLOR)
    fig.update_layout(
        yaxis_title="Discounted surplus ($M)",
        xaxis_title="Control year",
        showlegend=False,
        margin=dict(t=25, l=10, r=10, b=10),
        height=300,
    )
    return fig


def pitch_order(pitch_types) -> list:
    known = [p for p in PITCH_TYPE_ORDER if p in set(pitch_types)]
    unknown = sorted(set(pitch_types) - set(known))
    return known + unknown


MOVEMENT_AXIS_LIMIT = 25          # inches, both axes fixed at +/- this
MOVEMENT_GRID_STEP = 12.5
MIN_PITCHES_FOR_BUBBLE = 15
BUBBLE_SD = 2.0                   # ellipse drawn at this many SDs (2.0 ~ 86% of pitches for a 2D normal)
ARM_ANGLE_LINE_LENGTH = 20        # inches along the arm angle line


def is_dark_theme() -> bool:
    # st.context.theme reports the viewer's active theme; theme.base covers a dark theme
    # forced in .streamlit/config.toml (context still reports "light" in that case)
    try:
        return st.context.theme.type == "dark" or st.get_option("theme.base") == "dark"
    except Exception:
        return False


def chart_ink() -> str:
    """Near-black on the light theme, near-white on the dark theme (lines/labels on the movement chart)."""
    return "#F2F2F2" if is_dark_theme() else "#111111"


def covariance_ellipse(x: pd.Series, y: pd.Series, n_sd: float = BUBBLE_SD, points: int = 60):
    """(xs, ys) outline of the n_sd covariance ellipse of (x, y), from the eigenvectors/eigenvalues."""
    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    t = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack([np.cos(t), np.sin(t)])
    outline = eigvecs @ (n_sd * np.sqrt(eigvals)[:, None] * circle)
    return x.mean() + outline[0], y.mean() + outline[1]


def movement_chart(pitches: pd.DataFrame, stuff_rows: pd.DataFrame | None = None, location_rows: pd.DataFrame | None = None) -> go.Figure:
    """Pitcher's-view movement chart: faint pitch dots, a covariance bubble per pitch type, arm angle line.

    hb_arm flips hb (catcher's view) so arm-side movement is always positive:
    RHP fastballs/sinkers/changeups land on the right, sweepers/curves on the left.
    """
    ink = chart_ink()
    pitches = pitches.copy()
    pitches["hb_arm"] = np.where(pitches["p_throws"] == "R", -pitches["hb"], pitches["hb"])
    counts = pitches["PitchType"].value_counts()
    total = len(pitches)
    stuff = stuff_rows.set_index("PitchType")["StuffPlus_Scaled"] if stuff_rows is not None and "StuffPlus_Scaled" in stuff_rows else pd.Series(dtype=float)
    location = location_rows.set_index("PitchType")["LocationPlus"] if location_rows is not None and not location_rows.empty else pd.Series(dtype=float)

    fig = go.Figure()
    shown_types = [p for p in pitch_order(counts.index) if counts[p] >= MIN_PITCHES_FOR_BUBBLE]
    for pitch_type in shown_types:
        group = pitches[pitches["PitchType"] == pitch_type].dropna(subset=["hb_arm", "ivb"])
        color = PITCH_TYPE_COLORS.get(pitch_type, "#888888")
        if color == "#000000":
            color = ink  # black changeup bubble would vanish on the dark theme
        fig.add_trace(go.Scatter(
            x=group["hb_arm"], y=group["ivb"], mode="markers",
            marker=dict(size=4, color=color, opacity=0.15, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))

        xs, ys = covariance_ellipse(group["hb_arm"], group["ivb"])
        stuff_text = f"{stuff[pitch_type]:.0f}" if pitch_type in stuff.index and pd.notna(stuff[pitch_type]) else "--"
        location_text = f"{location[pitch_type]:.0f}" if pitch_type in location.index and pd.notna(location[pitch_type]) else "--"
        hover = (
            f"<b>{pitch_type}</b> {PITCH_TYPE_NAMES.get(pitch_type, '')}<br>"
            f"Usage: {counts[pitch_type] / total:.1%}<br>"
            f"Velo: {group['release_speed'].mean():.1f} mph<br>"
            f"IVB: {group['ivb'].mean():.1f} in<br>"
            f"HB (arm side +): {group['hb_arm'].mean():.1f} in<br>"
            f"Stuff+: {stuff_text} · Location+: {location_text}<extra></extra>"
        )
        stuff_value = stuff.get(pitch_type) if pitch_type in stuff.index else None
        outline = stat_color(stuff_value, 100, 10, True)  # red/blue by this pitch's Stuff+ vs. its pitch type
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=rgba(color, 0.35),
            line=dict(width=3 if outline else 1, color=outline or rgba(color, 0.6)),
            hoveron="fills", hovertemplate=hover, name=pitch_type, showlegend=False,
        ))
        fig.add_annotation(
            x=group["hb_arm"].mean(), y=group["ivb"].mean(), text=f"<b>{pitch_type}</b>",
            showarrow=False, font=dict(size=13, color=ink),
        )

    arm = pitches["arm_angle"].dropna()
    if not arm.empty:
        angle = arm.mean()
        estimated = (pitches.get("ArmAngleSource") == "release-point estimate").any()
        x_end = ARM_ANGLE_LINE_LENGTH * np.cos(np.radians(angle))
        y_end = ARM_ANGLE_LINE_LENGTH * np.sin(np.radians(angle))
        label = f"Arm angle: {angle:.0f}°" + (" (est.)" if estimated else "")
        fig.add_trace(go.Scatter(
            x=[0, x_end], y=[0, y_end], mode="lines", line=dict(width=5, color=ink),
            hovertemplate=(
                f"Season avg arm angle: {angle:.1f}°<br>"
                + ("APPROXIMATION from release point" if estimated else "Statcast arm_angle")
                + "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers", marker=dict(symbol="square", size=10, color=ink),
            hoverinfo="skip", showlegend=False,
        ))
        # label sits just above the midpoint of the line, rotated to follow it
        offset = 1.8
        fig.add_annotation(
            x=x_end * 0.55 - offset * np.sin(np.radians(angle)),
            y=y_end * 0.55 + offset * np.cos(np.radians(angle)),
            text=label, textangle=-angle, showarrow=False, font=dict(size=12, color=ink),
        )

    axis = dict(
        range=[-MOVEMENT_AXIS_LIMIT, MOVEMENT_AXIS_LIMIT], dtick=MOVEMENT_GRID_STEP,
        showgrid=True, gridcolor="rgba(128,128,128,0.25)", griddash="dash",
        zeroline=False, fixedrange=True,
    )
    fig.update_xaxes(title="Horizontal break (in, arm side +)", **axis)
    fig.update_yaxes(title="Induced vertical break (in)", scaleanchor="x", scaleratio=1, **axis)
    fig.add_hline(y=0, line_dash="dash", line_color=ink, line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color=ink, line_width=1)
    fig.update_layout(showlegend=False, height=520, margin=dict(t=10, l=60, r=10, b=55))
    return fig


SCALE_CAPTION = "100 = league average, 10 points = 1 standard deviation."
STUFF_COLOR = "#0072B2"      # team scatter points


def stuff_location_bar_chart(stuff_rows: pd.DataFrame, location_rows: pd.DataFrame) -> go.Figure:
    """Grouped bars per pitch type: Stuff+ (scaled, solid) next to Location+ (hatched).

    Bar color = stat_color of that bar's value (red = better than league
    average, blue = worse); the solid/hatched pattern tells the two apart.
    """
    stuff = stuff_rows.set_index("PitchType")["StuffPlus_Scaled"] if "StuffPlus_Scaled" in stuff_rows else pd.Series(dtype=float)
    location = location_rows.set_index("PitchType")["LocationPlus"] if location_rows is not None else pd.Series(dtype=float)
    types = [p for p in pitch_order(set(stuff.index) | set(location.index))]

    fig = go.Figure()
    for name, values, pattern in [("Stuff+", stuff, ""), ("Location+", location, "/")]:
        y = [values.get(p) for p in types]
        colors = [stat_color(v, 100, 10, True) or BASELINE_COLOR for v in y]
        fig.add_trace(go.Bar(
            name=name, x=types, y=y, showlegend=False,
            marker=dict(color=colors, pattern=dict(shape=pattern, fgcolor="rgba(255,255,255,0.8)", size=6, fillmode="overlay"),
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            text=[f"{v:.0f}" if v is not None and pd.notna(v) else "n/a" for v in y],
            textposition="outside",
            hovertemplate=f"%{{x}}: {name} %{{y:.1f}}<extra></extra>",
        ))
        # legend-only swatch in neutral gray, so the key shows the pattern, not one bar's color
        fig.add_trace(go.Bar(
            name=name, x=[None], y=[None],
            marker=dict(color=BASELINE_COLOR, pattern=dict(shape=pattern, fgcolor="rgba(255,255,255,0.8)", size=6, fillmode="overlay")),
        ))
    fig.add_hline(y=100, line_dash="dash", line_color=BASELINE_COLOR)  # axis title carries "100 = league avg"
    top = np.nanmax([v for v in list(stuff) + list(location) if v is not None] + [100])
    fig.update_yaxes(range=[0, top * 1.15])
    fig.update_layout(
        barmode="group",
        yaxis_title="Scaled (100 = league avg)",
        xaxis_title="Pitch type",
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
        height=380,
    )
    return fig


def stuff_location_scatter(staff: pd.DataFrame) -> go.Figure:
    """Stuff+ (x) vs Location+ (y) for a staff. Bottom-right = good stuff, poor location."""
    data = staff.dropna(subset=["StuffScaled", "LocationPlus"])
    fig = go.Figure()
    for qualified, label, symbol in [(True, "Qualified", "circle"), (False, "Under pitch minimum", "circle-open")]:
        group = data[data["Qualified"] == qualified]
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["StuffScaled"], y=group["LocationPlus"], name=label,
            mode="markers+text",
            text=group["Pitcher"].str.split().str[-1],
            textposition="top center",
            textfont=dict(size=10, color="#555"),
            marker=dict(size=11, color=STUFF_COLOR, symbol=symbol, line=dict(width=2, color="white" if qualified else STUFF_COLOR)),
            customdata=group[["Pitcher"]].values,
            hovertemplate="%{customdata[0]}<br>Stuff+: %{x:.1f}<br>Location+: %{y:.1f}<extra></extra>",
        ))
    fig.add_vline(x=100, line_dash="dash", line_color=BASELINE_COLOR)
    fig.add_hline(y=100, line_dash="dash", line_color=BASELINE_COLOR)
    if not data.empty:
        x_max = max(data["StuffScaled"].max(), 100) + 2
        y_min = min(data["LocationPlus"].min(), 100) - 2
        fig.add_annotation(
            x=x_max, y=y_min, xanchor="right", yanchor="bottom", showarrow=False,
            text="<b>Asymmetric upside</b><br>good stuff, poor location", font=dict(size=11, color="#333"),
            align="right",
        )
    fig.update_layout(
        xaxis_title="Stuff+ (scaled)",
        yaxis_title="Location+",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=30, l=10, r=10, b=10),
        height=460,
    )
    return fig


stuff_plus_summary, stuff_plus_pitches = load_real_stuff_plus()
location_plus = load_location_plus()
report_data, surplus_by_year = load_report_data()
# league-wide qualified pool for every color; darker league-average gray for numbers on white
pools = LeaguePools(report_data, neutral=NEUTRAL_COLOR if is_dark_theme() else NEUTRAL_TEXT_LIGHT)
chart_pools = LeaguePools(report_data)   # chart fills keep the standard #BFBFBF midpoint

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

tab_report, tab_board = st.tabs(["Pitcher Report", "Team Value Board"])


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

        pitcher_stuff_rows = stuff_plus_summary[stuff_plus_summary["PitcherId"] == pitcher_id]
        stuff_scaled = (
            pitcher_stuff_rows["OverallStuffPlus_Scaled"].iloc[0]
            if "OverallStuffPlus_Scaled" in pitcher_stuff_rows else float("nan")
        )
        pitcher_location_rows = (
            location_plus[location_plus["PitcherId"] == pitcher_id] if location_plus is not None else None
        )
        location_overall = (
            pitcher_location_rows["OverallLocationPlus"].iloc[0]
            if pitcher_location_rows is not None and not pitcher_location_rows.empty else float("nan")
        )

        report_row = (
            report_data[report_data["PitcherId"] == pitcher_id].iloc[0]
            if report_data is not None and (report_data["PitcherId"] == pitcher_id).any() else None
        )

        st.markdown(legend_html(), unsafe_allow_html=True)

        def report_value(col):
            return None if report_row is None or pd.isna(report_row[col]) else report_row[col]

        def money(v):
            return f"{'-' if v < 0 else ''}${abs(v):.1f}M"

        # (label, metric key in colors.METRIC_DIRECTION, value, formatter)
        cards = [
            ("Stuff+", "StuffPlus_Scaled", None if pd.isna(stuff_scaled) else stuff_scaled, "{:.1f}".format),
            ("Location+", "LocationPlus", None if pd.isna(location_overall) else location_overall, "{:.1f}".format),
            ("Volatility percentile", "VolatilityPercentile", report_value("VolatilityPercentile"), "{:.0f}".format),
            ("Risk-adj. WAR proxy", "RiskAdjWAR", report_value("RiskAdjWAR"), "{:.2f}".format),
            ("Multi-yr surplus", "MultiYearSurplus_M", report_value("MultiYearSurplus_M"), money),
        ]
        for col, (label, metric, value, formatter) in zip(st.columns(5), cards):
            text = "n/a" if value is None else formatter(value)
            col.markdown(
                stat_html(label, text, pools.color(metric, value), pools.percentile(metric, value)),
                unsafe_allow_html=True,
            )

        st.caption(SCALE_CAPTION + " Percentile = share of the league pool this pitcher is better than.")

        if report_row is None:
            st.info(
                "Stuff+ is real: trained on the full 2025 league-wide season, then "
                "scored against this pitcher's own full 2025 season (including time "
                "with another team if traded). This pitcher is under "
                "config.MIN_PITCHES_FOR_INCLUSION pitches, so has no risk-adjusted value "
                "(Stuff+/Location+ above are still graded against the league's qualified pool)."
            )
        else:
            st.caption(
                f"Graded against every qualified MLB pitcher. Volatility path: {report_row['VolatilityPath']}. "
                f"Contract: {report_row['ContractStatus']}, {report_row['YearsControl']:.0f} year(s) of control "
                f"({report_row['ControlSource']}). WAR is an uncalibrated proxy (see README)."
            )

        pitcher_years = (
            surplus_by_year[surplus_by_year["PitcherId"] == pitcher_id]
            if surplus_by_year is not None and not surplus_by_year.empty else pd.DataFrame()
        )
        if not pitcher_years.empty:
            st.markdown("**Surplus by control year** (discounted, $M)")
            st.plotly_chart(surplus_by_year_chart(pitcher_years, chart_pools), width='stretch')

        pitcher_pitches = stuff_plus_pitches[stuff_plus_pitches["PitcherId"] == pitcher_id]
        pitcher_summary = stuff_plus_summary[stuff_plus_summary["Pitcher"] == pitcher_choice]

        left, right = st.columns(2)
        with left:
            st.markdown("**Pitch movement** (pitcher's view)")
            if {"p_throws", "arm_angle"} <= set(pitcher_pitches.columns):
                st.plotly_chart(movement_chart(pitcher_pitches, pitcher_summary, pitcher_location_rows), width='stretch')
                st.caption(
                    f"Bubbles cover ~{1 - np.exp(-BUBBLE_SD ** 2 / 2):.0%} of each pitch type ({BUBBLE_SD:g} SD); "
                    f"pitch types with {MIN_PITCHES_FOR_BUBBLE}+ pitches. Arm angle is the season average."
                )
            else:
                st.caption("Re-run `python -m analysis.stuff_plus_proxy --team XXX` to export movement data.")

        with right:
            st.markdown("**Velocity by start (placeholder chart)**")
            st.caption("line chart of velocity across starts.")
            st.empty()

        st.markdown("**Stuff+ and Location+ by pitch type**")
        st.plotly_chart(stuff_location_bar_chart(pitcher_summary, pitcher_location_rows), width='stretch')
        st.caption(SCALE_CAPTION + " Stuff+ here is StuffPlus_Scaled (ratio StuffPlus stays in the data files). Hatched bars = Location+.")


        used_types = pitch_order(
            set(pitcher_pitches["PitchType"].unique()) | set(pitcher_summary["PitchType"].unique())
        )
        key_line = " · ".join(f"**{p}** {PITCH_TYPE_NAMES.get(p, 'Unknown')}" for p in used_types)
        st.caption(f"Pitch type key: {key_line}")

        if not pitcher_pitches.empty and {"p_throws", "ivb", "hb"} <= set(pitcher_pitches.columns):
            arsenal = pitcher_pitches.assign(
                hb_arm=np.where(pitcher_pitches["p_throws"] == "R", -pitcher_pitches["hb"], pitcher_pitches["hb"])
            ).groupby("PitchType").agg(
                Pitches=("ivb", "size"), Velo=("release_speed", "mean"), IVB=("ivb", "mean"), HB=("hb_arm", "mean"),
            )
            arsenal["Usage"] = arsenal["Pitches"] / arsenal["Pitches"].sum()
            arsenal["Stuff+"] = (
                pitcher_summary.set_index("PitchType")["StuffPlus_Scaled"] if "StuffPlus_Scaled" in pitcher_summary else np.nan
            )
            arsenal["Location+"] = (
                pitcher_location_rows.set_index("PitchType")["LocationPlus"]
                if pitcher_location_rows is not None and not pitcher_location_rows.empty else np.nan
            )
            arsenal = arsenal.sort_values(["Usage", "Pitches"], ascending=False)
            arsenal.insert(0, "Pitch", [PITCH_TYPE_NAMES.get(p, p) for p in arsenal.index])
            arsenal = arsenal.rename(columns={"Velo": "Velo (mph)", "HB": "HB (in, arm side +)", "IVB": "IVB (in)"})

            # Stuff+/Location+ are already graded within each pitch type (100 = league avg for that pitch)
            cell_colors = pd.DataFrame(None, index=arsenal.index, columns=arsenal.columns, dtype=object)
            cell_colors["Stuff+"] = [stat_color(v, 100, 10, True, pools.neutral) for v in arsenal["Stuff+"]]
            cell_colors["Location+"] = [stat_color(v, 100, 10, True, pools.neutral) for v in arsenal["Location+"]]

            st.markdown("**Pitch arsenal**")
            st.dataframe(
                colored_numbers(arsenal, cell_colors, {
                    "Velo (mph)": "{:.1f}", "IVB (in)": "{:.1f}", "HB (in, arm side +)": "{:.1f}",
                    "Usage": "{:.1%}", "Stuff+": "{:.0f}", "Location+": "{:.0f}",
                }),
                width='stretch',
            )
            st.caption("Stuff+ and Location+ compare each pitch to the league's pitches of that type. " + SCALE_CAPTION)


# Tab 2: team view, ranked by multi-year surplus
with tab_board:
    st.markdown(
        f"**{team_name_choice}** ranked by multi-year surplus: risk-adjusted production over "
        "every remaining year of team control, minus what they're paid, discounted to today. "
        "Graded against the whole league, not just this staff."
    )

    st.markdown(legend_html(), unsafe_allow_html=True)

    if not team_pitchers:
        st.caption("No pitchers loaded for this team yet.")
    elif report_data is None:
        st.caption(
            "No value data yet -- run asymmetric_upside, volatility_discount, "
            "risk_adjusted_value, then export_report_data (see README)."
        )
    else:
        team_ids = (
            stuff_plus_summary.loc[stuff_plus_summary["PitcherTeam"] == team_choice]
            .drop_duplicates("PitcherId")
            .assign(StuffScaled=lambda t: t["OverallStuffPlus_Scaled"] if "OverallStuffPlus_Scaled" in t else float("nan"))
            [["PitcherId", "Pitcher", "StuffScaled"]]
        )
        if location_plus is not None:
            overall_location = location_plus.drop_duplicates("PitcherId")[["PitcherId", "OverallLocationPlus", "OverallQualified"]]
            team_ids = team_ids.merge(overall_location, on="PitcherId", how="left")
        else:
            team_ids = team_ids.assign(OverallLocationPlus=float("nan"), OverallQualified=False)
        team_table = team_ids.merge(
            report_data.drop(columns=["Pitcher", "StuffPlus", "StuffPlus_Scaled", "LocationPlus"]), on="PitcherId", how="left")
        team_table = team_table.sort_values("MultiYearSurplus_M", ascending=False, na_position="last")

        shown = team_table.rename(columns={
            "StuffScaled": "Stuff+", "OverallLocationPlus": "Location+", "RiskAdjWAR": "Risk-adj. WAR", "Salary_M": "2025 salary ($M)",
            "YearsControl": "Years of control", "MultiYearSurplus_M": "Multi-yr surplus ($M)",
            "SurplusCurrentSeason_M": "2025 surplus ($M)",
        })[["Pitcher", "Stuff+", "Location+", "Risk-adj. WAR", "2025 salary ($M)", "Years of control",
            "Multi-yr surplus ($M)", "2025 surplus ($M)"]].reset_index(drop=True)

        metric_for_column = {
            "Stuff+": "StuffPlus_Scaled", "Location+": "LocationPlus",
            "Risk-adj. WAR": "RiskAdjWAR", "2025 salary ($M)": "Salary_M",
            "Multi-yr surplus ($M)": "MultiYearSurplus_M", "2025 surplus ($M)": "SurplusCurrentSeason_M",
        }
        cell_colors = pd.DataFrame(None, index=shown.index, columns=shown.columns, dtype=object)
        for col, metric in metric_for_column.items():
            cell_colors[col] = [pools.color(metric, v) for v in shown[col]]
        st.dataframe(
            colored_numbers(shown, cell_colors, {
                "Stuff+": "{:.1f}", "Location+": "{:.1f}", "Risk-adj. WAR": "{:.2f}",
                "2025 salary ($M)": "{:.2f}", "Years of control": "{:.0f}",
                "Multi-yr surplus ($M)": "{:.1f}", "2025 surplus ($M)": "{:.1f}",
            }),
            width='stretch', hide_index=True,
        )
        ungraded = team_table["RiskAdjWAR"].isna().sum()
        if ungraded:
            st.caption(f"{ungraded} pitcher(s) are under config.MIN_PITCHES_FOR_INCLUSION pitches and have no value row.")

        st.markdown("**Stuff+ vs. Location+**")
        st.plotly_chart(stuff_location_scatter(team_table.rename(columns={
            "OverallLocationPlus": "LocationPlus", "OverallQualified": "Qualified"})), width='stretch')
        st.caption(SCALE_CAPTION + " Bottom right = good stuff held back by location: the asymmetric upside group.")

        st.markdown("**Value map**: bottom-right, large, blue = cheap production with long control")
        st.plotly_chart(value_scatter(team_table), width='stretch')
        st.caption("Point size = years of control used (capped at config.MAX_CONTROL_YEARS). Color = multi-year surplus.")
