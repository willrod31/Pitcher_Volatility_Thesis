# Valuing Pitcher Volatility: Identifying Asymmetric Upside for Low-Payroll Front Offices

## Thesis

Can elite raw pitch characteristics (Stuff+) that are being suppressed by fixable command or pitch-usage issues be systematically identified using public Statcast data, allowing small-market teams to acquire mispriced pitching talent at a discount?

## Problem

Large-market teams pay a premium for low-volatility, proven starters. Small-market teams can't afford that premium. This project builds a framework, using public Statcast data, to identify pitchers whose volatility is mispriced: elite raw pitch characteristics depressed by fixable factors like command or pitch usage, rather than inherent limitations.

## Deliverable

A Streamlit dashboard: select a team, select a player, get a generated report combining the four modules' output for that pitcher. Analysis and dashboard are both Python.

## Data note

Stuff+ and Location+ are proprietary (FanGraphs/PitcherList), not available via pybaseball or the public Statcast feed. This project builds its own Stuff+ Proxy from raw pitch physics (velocity, movement, spin, extension) regressed against outcome value, instead of relying on a black-box metric. See analysis/stuff_plus_proxy.py.

## Repo structure

pitcher-volatility-thesis/
├── README.md
├── requirements.txt
├── config.py                    # season, thresholds, constants
├── data/                        # cached Statcast pulls + each script's output (gitignored)
├── notebooks/
│   └── 01_exploratory.ipynb     # scratch space for exploring data before it becomes a script
├── analysis/
│   ├── data_acquisition.py      # pybaseball pulls + caching, shared by every script below
│   ├── stuff_plus_proxy.py      # Stuff+ proxy model
│   ├── location_plus_proxy.py   # Location+ proxy (command from location + count only)
│   ├── asymmetric_upside.py     # stuff vs. command mispricing index (incl. Command Proxy)
│   ├── volatility_discount.py   # release point / velo variance modeling
│   ├── injury_history.py        # IL stints from the MLB Stats API transactions feed
│   ├── contract_status.py       # estimated contract status / years of control from MLB debut dates
│   ├── risk_adjusted_value.py   # risk-adjusted WAR proxy + $ surplus value ranking
│   ├── export_report_data.py    # joins all analysis output into one CSV for the dashboard
│   └── utils.py
└── dashboard/
    ├── app.py                   # Streamlit dashboard
    └── data/                    # exports the analysis scripts hand off to app.py (gitignored)
        ├── stuff_plus_pitch_types.csv   # from stuff_plus_proxy.py --pitcher
        ├── stuff_plus_pitches.csv       # from stuff_plus_proxy.py --pitcher (scored swings, with per-pitch StuffPlus)
        ├── movement_pitches.csv         # from stuff_plus_proxy.py --pitcher/--team (every pitch + arm angle, for the movement chart)
        ├── location_plus_pitch_types.csv # from location_plus_proxy.py (league-wide)
        ├── pitcher_report_data.csv      # from export_report_data.py (team value board + pitcher metrics)
        └── surplus_by_year.csv          # from export_report_data.py (surplus by control year chart)

## Analysis scripts

stuff_plus_proxy.py — Stuff+ Proxy. Trains a whiff-probability model per pitch type on raw pitch physics, scaled to a 100-average scale.

location_plus_proxy.py — Location+ Proxy. Command measured from pitch location and count only (see "Location+ Proxy" below).

asymmetric_upside.py — Asymmetric Upside Index. Flags pitchers with elite Stuff+ but poor results explained by command, not shape. Index = z(Stuff+) - z(Location+) by default; set config.COMMAND_METRIC = "legacy" to use the original Command Proxy (z(Edge%) - z(Meatball%) - z(BB%)), which is always kept as the CommandProxyLegacy column for comparison. Also flags pitch-mix inefficiency: best-performing pitch type vs. most-used pitch type.

volatility_discount.py — Volatility Discount Metric. Computes outing-to-outing variance in release point and velocity. Starters (8+ starts) are measured start to start; everyone else goes through a reliever path (15+ appearances of 10+ pitches), labeled in the VolatilityPath column, and percentiles are within each path. Regresses future performance (next-outing results) on current volatility to test whether volatility actually predicts decline, versus how much the market appears to penalize it.

asymmetric_upside.py, volatility_discount.py and risk_adjusted_value.py always grade against the whole league (loaded one month at a time, so no OOM) and save league-wide tables; `--team PIT` only filters what's printed, so Pirates pitchers are compared to the league, not to each other. Run order:

    python -m analysis.injury_history --team PIT
    python -m analysis.contract_status --team PIT
    python -m analysis.location_plus_proxy --team PIT
    python -m analysis.asymmetric_upside --team PIT
    python -m analysis.volatility_discount --team PIT
    python -m analysis.risk_adjusted_value --team PIT
    python -m analysis.export_report_data

injury_history.py — IL history. Pulls each pitcher's transactions from the MLB Stats API and builds one row per IL stint (data/injury_stints.csv) and one per pitcher (data/injury_summary.csv). volatility_discount.py joins the summary and also tests whether volatility predicts an IL stint in the next 30 days.

    python -m analysis.injury_history --pitcher "Paul Skenes"
    python -m analysis.injury_history --team PIT

contract_status.py — estimates ContractStatus (pre-arb / arb / FA-eligible) and years of control from each pitcher's mlbDebutDate (MLB Stats API /people), writes data/contract_status_est.csv, and flags pitchers where the estimate disagrees with data/salaries.csv. EstServiceYears is SEASON minus debut year -- an approximation that ignores option years, IL, partial seasons and Super Two.

risk_adjusted_value.py — Risk-Adjusted Value Projection. Combines the upside index, volatility metric and IL history into a projected WAR proxy, shrunk toward replacement level as volatility and injury risk increase. Joins the hand-collected data/salaries.csv on PitcherId (blank contract fields fall back to contract_status.py's estimate; blank pre-arb salaries get the league minimum). Ranks pitchers by MultiYearSurplus: discounted surplus over each remaining year of team control (year-by-year table in data/surplus_by_year.csv). SurplusCurrentSeason keeps the old single-season number for comparison.

export_report_data.py — joins all four scripts' output into the one CSV the dashboard reads.

## Setup

pip install -r dashboard/requirements.txt

## Stuff+ Proxy: real-data demo

python -m analysis.stuff_plus_proxy --pitcher "Paul Skenes"   # one pitcher
python -m analysis.stuff_plus_proxy --team PIT                # a whole staff

A single league-wide statcast() call for the whole season is ~700k+ rows and
can OOM on a memory-limited machine (confirmed on an 8GB box) during
pybaseball's internal per-day concat. The whiff model is trained on the full
season anyway (TRAIN_START/TRAIN_END in stuff_plus_proxy.py default to the
whole config.py range) -- data_acquisition.load_season_monthly() pulls it one
calendar month at a time instead, keeping each pybaseball call's internal
concat bounded (~120k rows/~3.3GB peak per month, confirmed safe; ~5.1GB
peak for the full 6-month training pool once combined) and dropping the
~100 unused Statcast columns per month before accumulating across months.
Both --pitcher and --team then score against that full-season model via
targeted pulls instead of a league-wide one. --pitcher uses pybaseball's
per-player endpoint directly. --team is two steps: pybaseball's `team=`
filter (own-pitching rows only, ~20-25k for a full season) finds who's on
the roster, then each of those pitchers gets a full per-player pull covering
their *entire* season -- including time with another team if they were
traded mid-year, not just their innings on this roster (a pitcher shown
under a team here is graded on their whole season, labeled with that team,
per data_acquisition.load_team_roster_full_seasons()). Either way it writes
dashboard/data/stuff_plus_pitch_types.csv and stuff_plus_pitches.csv, which
dashboard/app.py reads directly -- --team just writes more rows to them at
once. `--pitcher` and `--team` upsert independently, so you can build up
coverage team by team without re-running ones you've already done.

## Location+ Proxy

Location+ is the command counterpart to Stuff+: Stuff+ sees only pitch shape, Location+ sees only where the pitch ended up and the count. It replaces the old Command Proxy, which pooled all pitch types, ignored the count, and used BB% (partly driven by stuff, and itself a surface result).

    python -m analysis.location_plus_proxy                       # league-wide
    python -m analysis.location_plus_proxy --pitcher "Paul Skenes"
    python -m analysis.location_plus_proxy --team PIT

- **Features (location and count only):** norm_x (plate_x / half the plate width, flipped for left-handed batters so positive is always the same side), norm_z (height relative to the batter's own sz_top/sz_bot), balls, strikes, same_hand. Never velocity, movement, spin, extension or release point.
- **Target:** delta_run_exp per pitch (pitcher's view, lower = better). On balls in play it's replaced by an expected value from a league regression of delta_run_exp on xwOBA + balls + strikes, so batted-ball luck isn't counted as command.
- **Model:** one HistGradientBoostingRegressor per pitch group (Fastball FF/SI/FC, Breaking SL/ST/CU/KC/SV, Offspeed CH/FS/FO/SC), trained on every pitch league-wide (full config.py season, loaded month by month).
- **Out-of-fold scoring:** 5-fold GroupKFold by pitcher, so a pitcher's own pitches never train the model that scores him. Fold R^2 is low (~0.06) because single-pitch run value is mostly noise; the pitcher-level averages are what matter.
- **Scale:** LocationPlus = 100 + 10 * z(-mean predicted run value), z against the league-wide qualified pool (config.MIN_PITCHES_FOR_INCLUSION pitches; per pitch type, 100+ pitches and z within that pitch type). 100 = league average, 10 points = 1 SD. --pitcher/--team only filter the output.
- **Reliability:** each run also writes data/command_reliability.csv: odd/even split-half correlation for LocationPlus, the legacy CommandProxy, Edge%, Meatball%, BB%, first-pitch strike % and 3-ball-count %, plus each metric's correlation with Stuff+.

**Two Stuff+ columns.** StuffPlus is the original ratio (predicted whiff probability / league average for the pitch type x 100), unchanged. StuffPlus_Scaled = 100 + 10 * z(StuffPlus) against the league-wide qualified pool (per pitch type at the pitch-type level), which puts it on exactly the same scale as Location+. The dashboard's "Stuff+" tiles, bar chart, table and Stuff+ vs. Location+ scatter show StuffPlus_Scaled; only the pitch movement chart's hover shows the per-pitch ratio StuffPlus.

## Dashboard

streamlit run dashboard/app.py

The Team Value Board tab ranks the selected team by multi-year surplus (Stuff+, risk-adjusted WAR, 2025 salary, years of control) with a WAR vs. salary scatter (point size = years of control, color = multi-year surplus). The Pitcher Report shows that pitcher's surplus by control year. Both need export_report_data.py to have been run.

The sidebar lets you navigate every MLB team in both leagues. Picking a team
with no Stuff+ data loaded says so; picking the Pittsburgh Pirates (after
running the --team command above) lists their real 2025 pitching staff, each
with real Stuff+ Proxy results. There's no sample/placeholder data mode
anymore -- everything the dashboard shows is real, or explicitly says it
hasn't been run yet.

analysis/export_report_data.py is the hand-off point between the full
analysis pipeline and the dashboard: it joins all four modules' output into
one flat CSV with a fixed column contract. It requires all four modules to
have been run league-wide first (see "Known limitations" below on the OOM
risk of that for stuff_plus_proxy.py).

## Known limitations

- A full league-wide season pull in one `load_season()` call can OOM on a memory-limited machine during pybaseball's concat step -- confirmed on an 8GB box around the 6-month mark. `load_season_monthly()` works around this (see above) and is used by stuff_plus_proxy.py's --pitcher/--team training step and by asymmetric_upside.py and volatility_discount.py (~4GB peak each); only the plain no-flag league-wide path in stuff_plus_proxy.py still uses `load_season()`.
- Stuff+ Proxy is a gradient-boosted classifier on shape features, not a full replacement for proprietary gradient-boosted Stuff+ models.
- IL history comes from the MLB Stats API transactions feed (analysis/injury_history.py), not manual collection anymore. The feed occasionally leaves out an activation, so when one's missing the return date is taken from the pitcher's first MLB appearance after the placement. Only pitchers you've run it for (--team/--pitcher) have injury data; everyone else gets 0 injury risk in risk_adjusted_value.py (InjuryDataPulled = False).
- Salaries are hand-collected in data/salaries.csv (not scriptable). Contract status and years of control can be estimated from debut dates (contract_status.py), but EstServiceYears is SEASON minus debut year: it ignores option years, IL, partial seasons and Super Two, so hand-entered values always win.
- simple_projected_war in risk_adjusted_value.py is an uncalibrated placeholder until regressed against actual historical WAR for a training sample. It uses the same starter-based WAR anchor for relievers, so reliever WAR is overstated.
- Multi-year surplus holds ProjectedWAR flat across every control year (no aging curve yet), and the arb salary percentages, $/WAR growth, discount rate, injury shrinkage and control-year cap in config.py are assumptions, not fitted values.
- Pitchers under config.MIN_PITCHES_FOR_INCLUSION (500) pitches aren't graded by asymmetric_upside.py, so they have no risk-adjusted value (risk_adjusted_value.py --team lists them).
- Location+ still can't know the catcher's target: it measures location quality given the count, not intent. A pitcher who hits a spot the catcher set on the edge looks the same as one who missed there. Because the count is a feature, it also partly reflects getting into bad counts, not only where each pitch lands.
- The legacy Command Proxy (CommandProxyLegacy in asymmetric_upside.py) is a zone-discipline stand-in (edge%/meatball%/BB%) and has the same no-target limitation.
