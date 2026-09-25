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
│   ├── asymmetric_upside.py     # stuff vs. command mispricing index (incl. Command Proxy)
│   ├── volatility_discount.py   # release point / velo variance modeling
│   ├── injury_history.py        # IL stints from the MLB Stats API transactions feed
│   ├── risk_adjusted_value.py   # risk-adjusted WAR proxy + $ surplus value ranking
│   ├── export_report_data.py    # joins all analysis output into one CSV for the dashboard
│   └── utils.py
└── dashboard/
    ├── app.py                   # Streamlit dashboard
    └── data/                    # exports the analysis scripts hand off to app.py (gitignored)
        ├── stuff_plus_pitch_types.csv   # from stuff_plus_proxy.py --pitcher
        ├── stuff_plus_pitches.csv       # from stuff_plus_proxy.py --pitcher
        └── pitcher_report_data.csv      # from export_report_data.py (not wired into app.py yet)

## Analysis scripts

stuff_plus_proxy.py — Stuff+ Proxy. Trains a whiff-probability model per pitch type on raw pitch physics, scaled to a 100-average scale.

asymmetric_upside.py — Asymmetric Upside Index. Flags pitchers with elite Stuff+ Proxy but poor surface results explained by command, not shape. Also computes the Command Proxy itself (edge%, meatball%, walk rate, since Location+ is proprietary). Index = z(Stuff+ Proxy) - z(Command Proxy). Also flags pitch-mix inefficiency: best-performing pitch type vs. most-used pitch type.

volatility_discount.py — Volatility Discount Metric. Computes per-start variance in release point and velocity. Regresses future performance (next-start results) on current volatility to test whether volatility actually predicts decline, versus how much the market appears to penalize it.

injury_history.py — IL history. Pulls each pitcher's transactions from the MLB Stats API and builds one row per IL stint (data/injury_stints.csv) and one per pitcher (data/injury_summary.csv). volatility_discount.py joins the summary and also tests whether volatility predicts an IL stint in the next 30 days.

    python -m analysis.injury_history --pitcher "Paul Skenes"
    python -m analysis.injury_history --team PIT

risk_adjusted_value.py — Risk-Adjusted Value Projection. Combines the upside index and volatility metric into a projected WAR proxy, shrunk toward replacement level as volatility increases. Ranks pitchers by surplus value (risk-adjusted projected value minus contract cost).

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

## Dashboard

streamlit run dashboard/app.py

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

- A full league-wide season pull in one `load_season()` call can OOM on a memory-limited machine during pybaseball's concat step -- confirmed on an 8GB box around the 6-month mark. `load_season_monthly()` works around this (see above) and is used by stuff_plus_proxy.py's --pitcher/--team training step; the plain no-flag full-league-wide path in stuff_plus_proxy.py and the other three analysis modules don't use it yet.
- Stuff+ Proxy is a gradient-boosted classifier on shape features, not a full replacement for proprietary gradient-boosted Stuff+ models.
- IL history comes from the MLB Stats API transactions feed (analysis/injury_history.py), not manual collection anymore. The feed occasionally leaves out an activation, so when one's missing the return date is taken from the pitcher's first MLB appearance after the placement. Only pitchers you've run it for (--team/--pitcher) have injury data.
- Contract cost data is not scriptable via pybaseball — requires manual collection (data/salaries.csv) for risk_adjusted_value.py.
- simple_projected_war in risk_adjusted_value.py is an uncalibrated placeholder until regressed against actual historical WAR for a training sample.
- Command Proxy (in asymmetric_upside.py) is a zone-discipline stand-in (edge%/meatball%/BB%), not a real command measurement — there's no intended-target data in the public Statcast feed.
