# Home page: what the project is, how to read the dashboard, what every number means
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from colors import legend_html  # noqa: E402
from pitch_types import PITCH_TYPE_ABBR, pitch_name  # noqa: E402
from config import (  # noqa: E402
    DISCOUNT_RATE, DOLLARS_PER_WAR, DOLLARS_PER_WAR_GROWTH, INJURY_LOOKBACK_SEASONS, INJURY_SHRINKAGE, MAX_CONTROL_YEARS,
    MIN_PITCHES_DASHBOARD, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_TO_DISPLAY, SEASON, VOLATILITY_SHRINKAGE,
)

st.title("Valuing Pitcher Volatility")
st.markdown("#### Finding undervalued pitchers for low-payroll MLB teams")

st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")

st.header("The question")
st.markdown(
    "Can public Statcast data find pitchers with **elite pitch quality (Stuff+)** whose results are held back by "
    "**fixable problems** like command or pitch mix? If so, small-market teams could get them at a discount.\n\n"
    "Big-market teams pay a premium for proven, consistent starters. Small-market teams can't afford that premium, "
    "so they need talent the market is mispricing: good stuff, shaky command, and some outing-to-outing volatility "
    "that scares other teams off."
)

st.header("How it works")
st.markdown(
    f"""
1. **Grade the stuff.** Every pitch in the {SEASON} season is scored on its shape (velocity, movement, spin,
   extension, release point) → **Stuff+**.
2. **Grade the command.** Separately, every pitch is scored on where it was thrown, given the count → **Location+**.
3. **Find the gap.** Pitchers whose Stuff+ is well above their Location+ have great stuff that is held back by
   command, which is more fixable than stuff.
4. **Measure the risk.** **Volatility** (how much velocity and release point change from outing to outing) and
   **injury history** (IL stints over the last {INJURY_LOOKBACK_SEASONS} seasons).
5. **Put a price on it.** Turn talent into a projected WAR, shrink it for risk, and compare it to salary over every
   remaining year of team control → **surplus value**.
"""
)

st.header("Using the dashboard")
st.markdown(
    """
Pick a **league**, **team** and **pitcher** in the sidebar of the Dashboard page. There are two tabs:

- **Pitcher Report**: one pitcher in detail. Headline grades, surplus by contract year, a pitch movement chart,
  fastball velocity by game (with IL stints marked), and Stuff+ / Location+ for each pitch type.
- **Team Value Board**: the team's whole staff ranked by multi-year surplus, plus two charts: Stuff+ vs. Location+
  (who has upside) and a value map (who is cheap production).
"""
)

st.header("Reading the colors")
st.markdown(legend_html(MIN_PITCHES_DASHBOARD), unsafe_allow_html=True)
st.markdown(
    f"""
- **Red is good, blue is bad**, for every stat, even ones where a lower number is better (salary, volatility).
- Colors compare a pitcher to **every qualified MLB pitcher** ({MIN_PITCHES_FOR_INCLUSION}+ pitches), not just his
  own team.
- **Faded color** or a **"small"** tag means the sample is small and the number is less reliable.
- **n/a** means there isn't enough data to grade it (for example, a pitch type thrown fewer than
  {MIN_PITCHES_PER_TYPE_TO_DISPLAY} times).
- Hover over a number to see its raw value before regression.
- Some stats, like movement, usage, arm angle and years of control, have no color because there's no "better" value.
"""
)

st.header("What each stat means")

with st.expander("**Stuff+**: how hard a pitch is to hit", expanded=True):
    st.markdown(
        """
Built only from the pitch's physical traits: velocity, movement, spin, extension and release point. It says nothing
about where the pitch was thrown.

- **100 = league average, and every 10 points = one standard deviation.** 110 is very good, 120 is elite,
  90 is below average.
- By pitch type, each pitch is compared to the league's pitches **of the same type** (a slider to other sliders).
- Each pitcher is scored by a model that never saw his own pitches (out-of-fold), so the grade isn't inflated.
"""
    )

with st.expander("**Location+**: command"):
    st.markdown(
        """
Built only from where the pitch crossed the plate and the count. It shares no inputs with Stuff+, so the two can be
compared directly.

- Same scale: **100 = average, 10 points = one standard deviation**.
- It can't see where the catcher set up, so it measures location, not intent.
"""
    )

with st.expander("**Upside**: great stuff, weaker command"):
    st.markdown(
        """
The gap between Stuff+ and Location+. On the Team Value Board's **Stuff+ vs. Location+** chart, pitchers in the
**bottom right** (high Stuff+, low Location+) are the target group: the stuff is already there, and command is the
part a coaching staff can work on.
"""
    )

with st.expander("**Volatility percentile**: outing-to-outing inconsistency"):
    st.markdown(
        """
How much a pitcher's velocity and release point move around from one outing to the next.

- **Higher percentile = more volatile = riskier** (shown in blue).
- Starters are measured start to start; relievers appearance to appearance. The report shows which path was used.
"""
    )

with st.expander("**Risk-adjusted WAR proxy**"):
    st.markdown(
        f"""
Talent (Stuff+ and Location+ combined) converted into a projected WAR, then pulled back toward replacement level
for risk:

- up to **{VOLATILITY_SHRINKAGE:.0%}** of the upside is removed for the most volatile pitchers, and
- up to **{INJURY_SHRINKAGE:.0%}** for the most injury-prone.

This is an **uncalibrated placeholder**. Use it to rank pitchers against each other, not as a real WAR forecast.
It currently overstates relievers.
"""
    )

with st.expander("**Multi-year surplus**: the bottom line"):
    st.markdown(
        f"""
What the pitcher is worth minus what he's paid, added up over every remaining year of team control
(capped at {MAX_CONTROL_YEARS} years) and converted to today's dollars.

- **Worth** = risk-adjusted WAR × ${DOLLARS_PER_WAR / 1e6:.1f}M per WAR (rising {DOLLARS_PER_WAR_GROWTH:.0%} a year).
- **Paid** = actual salary this year; league minimum during pre-arbitration years; a share of market value during
  arbitration years.
- Future years are discounted by **{DISCOUNT_RATE:.0%}** a year, because a win next year is worth less than a win now.
- WAR is held flat across years (no aging curve yet).

A big positive number means cheap production with years of control left, which is what a low-payroll team wants.
"""
    )

st.header("Pitch abbreviations")
st.markdown("Charts label pitches with these standard abbreviations. Hover a pitch on any chart to see its full name.")
st.dataframe(
    pd.DataFrame({"Abbreviation": list(PITCH_TYPE_ABBR.values()), "Pitch": [pitch_name(c) for c in PITCH_TYPE_ABBR]}),
    hide_index=True, width=360,
)

st.header("Small samples")
st.markdown(
    f"""
A reliever with 150 pitches shouldn't look as certain as a starter with 3,000. Every stat is pulled toward the league
average based on how reliable it is at that sample size:
"""
)
st.code("Adjusted value = (n × observed + k × league average) / (n + k)", language=None)
st.markdown(
    f"""
`n` is the pitcher's sample size, and `k` is the sample size where that stat becomes 50% reliable. Stuff+ is reliable
almost immediately (k ≈ 1 pitch); Location+ takes about 286 pitches.

- League averages come only from pitchers with **{MIN_PITCHES_FOR_INCLUSION}+ pitches**. Pitchers under that are
  still shown, with regressed values, but don't affect anyone else's grade.
- Pitchers with fewer than **{MIN_PITCHES_DASHBOARD} pitches** in {SEASON} aren't shown at all.
"""
)

st.header("Contracts")
st.markdown(
    """
Salaries for all 30 teams were entered by hand from FanGraphs RosterResource payroll pages.

- **Traded pitchers** are valued at their full salary. The report notes how much each team actually paid.
- **"est. contract"** means the pitcher wasn't in the salary file. He's valued at the league minimum, with team control
  estimated from his debut date (or one year if that isn't possible).
- **Shohei Ohtani's** pitching is graded, but his contract isn't valued, because his salary pays for hitting too.
"""
)

st.header("What we've found so far")
st.markdown(
    """
- **Stuff+ is consistent but only a weak predictor.** It ranks pitchers the same way from start to start, but it
  predicts whiffs on new pitchers only weakly. Treat it as a ranking of pitch shape, not a whiff probability.
- **Some command stats are mostly noise.** Edge% barely stabilizes even after thousands of pitches; Location+ is far
  more reliable.
"""
)

st.header("Limitations")
st.markdown(
    """
- Stuff+ and Location+ are simpler, homemade versions of the commercial stats.
- Location+ can't see the catcher's target.
- The WAR projection isn't calibrated against real WAR yet, and it overstates relievers.
- Value assumptions (arbitration raises, $/WAR, discount rate) are set by hand, not fitted.
- Some contract and control values are estimates.
"""
)

st.divider()
st.caption(f"Data: Statcast via pybaseball, {SEASON} regular season. Salaries: FanGraphs RosterResource (hand-entered).")
st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")
