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
├── requirements.txt             # analysis scripts (includes dashboard/requirements.txt)
├── config.py                    # season, thresholds, constants
├── data/                        # cached Statcast pulls + each script's output (gitignored)
├── notebooks/
│   └── 01_exploratory.ipynb     # scratch space for exploring data before it becomes a script
├── analysis/
│   ├── data_acquisition.py      # pybaseball pulls + caching, shared by every script below
│   ├── stuff_plus_proxy.py      # Stuff+ proxy model
│   ├── location_plus_proxy.py   # Location+ proxy (command from location + count only)
│   ├── stabilization.py         # split-half stabilization points (k) + regression toward the mean
│   ├── asymmetric_upside.py     # stuff vs. command mispricing index (incl. Command Proxy)
│   ├── volatility_discount.py   # release point / velo variance modeling
│   ├── injury_history.py        # IL stints from the MLB Stats API transactions feed
│   ├── contract_status.py       # estimated contract status / years of control from MLB debut dates
│   ├── risk_adjusted_value.py   # risk-adjusted WAR proxy + $ surplus value ranking
│   ├── export_report_data.py    # joins all analysis output into one CSV for the dashboard
│   ├── run_all.py               # whole pipeline for all 30 teams + missing-contract report
│   └── utils.py
└── dashboard/
    ├── app.py                   # Streamlit dashboard
    ├── requirements.txt         # dashboard-only deps (what Streamlit Cloud installs)
    └── data/                    # exports the analysis scripts hand off to app.py (committed; the deployed app reads these)
        ├── stuff_plus_pitch_types.csv   # from stuff_plus_proxy.py --pitcher
        ├── stuff_plus_pitches.csv.gz    # from stuff_plus_proxy.py --pitcher (scored swings, with per-pitch StuffPlus; gzipped)
        ├── movement_pitches.csv.gz      # from stuff_plus_proxy.py --pitcher/--team (every pitch + arm angle, for the movement chart; gzipped)
        ├── location_plus_pitch_types.csv # from location_plus_proxy.py (league-wide)
        ├── pitcher_report_data.csv      # from export_report_data.py (team value board + pitcher metrics)
        ├── surplus_by_year.csv          # from export_report_data.py (surplus by control year chart)
        ├── injury_stints.csv            # from export_report_data.py (IL spans on the velocity chart)
        └── velocity_by_game.csv         # from stuff_plus_proxy.py --pitcher/--team (primary fastball velo per appearance)

## Analysis scripts

stuff_plus_proxy.py — Stuff+ Proxy. Trains a whiff-probability model per pitch type on raw pitch physics, scaled to a 100-average scale. See "Stuff+ v2" below for the handedness fix, out-of-fold scoring and validation.

stabilization.py — split-half stabilization points (k) for every metric, and the regression toward the league mean every module uses for small samples. See "Small samples" below.

location_plus_proxy.py — Location+ Proxy. Command measured from pitch location and count only (see "Location+ Proxy" below).

asymmetric_upside.py — Asymmetric Upside Index. Flags pitchers with elite Stuff+ but poor results explained by command, not shape. Index = z(Stuff+) - z(Location+) by default; set config.COMMAND_METRIC = "legacy" to use the original Command Proxy (z(Edge%) - z(Meatball%) - z(BB%)), which is always kept as the CommandProxyLegacy column for comparison. Also flags pitch-mix inefficiency: best-performing pitch type vs. most-used pitch type.

volatility_discount.py — Volatility Discount Metric. Computes outing-to-outing variance in release point and velocity. Starters (8+ starts) are measured start to start; everyone else goes through a reliever path (15+ appearances of 10+ pitches), labeled in the VolatilityPath column, and percentiles are within each path. Regresses future performance (next-outing results) on current volatility to test whether volatility actually predicts decline, versus how much the market appears to penalize it.

asymmetric_upside.py, volatility_discount.py and risk_adjusted_value.py always grade against the whole league (loaded one month at a time, so no OOM) and save league-wide tables; `--team PIT` only filters what's printed, so Pirates pitchers are compared to the league, not to each other. Run order:

    python -m analysis.injury_history --team PIT
    python -m analysis.contract_status --team PIT
    python -m analysis.stabilization              # builds/uses the cached league scores; writes data/stabilization.csv
    python -m analysis.stuff_plus_proxy --team PIT
    python -m analysis.location_plus_proxy --team PIT
    python -m analysis.asymmetric_upside --team PIT
    python -m analysis.volatility_discount --team PIT
    python -m analysis.risk_adjusted_value --team PIT
    python -m analysis.export_report_data

### Run every team at once

    python -m analysis.run_all                  # all 30 teams
    python -m analysis.run_all --teams PIT,LAD  # a subset (Statcast codes; ARI is accepted for AZ)

run_all.py runs the same modules for every team but trains and grades the league only once:

1. Each team's roster cache (data/statcast_team_<TEAM>_*.parquet) is split out of the cached league monthly pulls instead of one Statcast pull per team (identical to the per-team pull; checked on PIT).
2. injury_history and contract_status run for every team first, since volatility and risk-adjusted value use every team's IL history.
3. The league season is loaded month by month once (peak ~1.2GB on the PIT test, fine on 8GB) and shared by Stuff+, the Stuff+ dashboard exports and asymmetric_upside. Stuff+ and Location+ reuse their cached out-of-fold scores, then upside, volatility and risk-adjusted value are graded league-wide.
4. Per team: each module's team view and the Stuff+ dashboard exports. Each roster pitcher's full season (any team) is taken from the shared league frame by PitcherId, not pulled by name.
5. export_report_data runs once, then the missing-contract report is written (below).

Each team's console output goes to logs/run_<TEAM>.txt and the league steps to logs/run_league.txt (logs/ is gitignored). If a step fails, its traceback goes to that log, the run continues, and every failure is listed at the end.

A traded pitcher is listed under every team he pitched for in stuff_plus_pitch_types.csv (keyed on PitcherId x PitcherTeam), graded on his whole season under each.

### Missing-contract report

run_all.py writes data/missing_contract_report.csv: one row per pitcher x 2025 team shown in the dashboard whose contract data has a gap:

- `no salary`: no Salary after all fallbacks (not in salaries.csv and not estimated pre-arb), so there's no surplus value.
- `missing (1 yr assumed)`: no YearsControl in salaries.csv, and the debut-date estimate gives 0 years or none (usually veterans not in salaries.csv).
- `status mismatch`: the salaries.csv ContractStatus disagrees with the debut-date estimate. Often the estimate is the one that's wrong (see contract_status.py), so this is a list to double-check, not a list of errors.

Columns: PitcherId, Pitcher, Team, Problem (several are joined with "; "), ContractStatus_est, EstYearsControl, Pitches2025 (cleaned Statcast pitches, all teams). Sorted by Pitches2025, so the gaps that matter most come first. The run also prints per-team counts of each problem, and how many pitchers per team got their 2025 Salary from salaries.csv vs. the league-minimum default. Fix gaps by adding rows to data/salaries.csv and re-running.

injury_history.py — IL history. Pulls each pitcher's transactions from the MLB Stats API and builds one row per IL stint (data/injury_stints.csv) and one per pitcher (data/injury_summary.csv). volatility_discount.py joins the summary and also tests whether volatility predicts an IL stint in the next 30 days.

    python -m analysis.injury_history --pitcher "Paul Skenes"
    python -m analysis.injury_history --team PIT

contract_status.py — estimates ContractStatus (pre-arb / arb / FA-eligible) and years of control from each pitcher's mlbDebutDate (MLB Stats API /people), writes data/contract_status_est.csv, and flags pitchers where the estimate disagrees with data/salaries.csv. EstServiceYears is SEASON minus debut year -- an approximation that ignores option years, IL, partial seasons and Super Two.

risk_adjusted_value.py — Risk-Adjusted Value Projection. Combines the upside index, volatility metric and IL history into a projected WAR proxy, shrunk toward replacement level as volatility and injury risk increase. Joins data/salaries.csv on PitcherId (blank contract fields fall back to contract_status.py's estimate; blank pre-arb salaries get the league minimum). Ranks pitchers by MultiYearSurplus: discounted surplus over each remaining year of team control (year-by-year table in data/surplus_by_year.csv). SurplusCurrentSeason keeps the old single-season number for comparison.

export_report_data.py — joins all four scripts' output into the one CSV the dashboard reads.

## Setup

Requires Python 3.10+ (the code uses `X | None` type hints). On Streamlit Community Cloud, pick Python 3.10 or newer under Advanced settings when deploying.

pip install -r requirements.txt              # analysis scripts + dashboard
pip install -r dashboard/requirements.txt    # dashboard only

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
dashboard/data/stuff_plus_pitch_types.csv and stuff_plus_pitches.csv.gz, which
dashboard/app.py reads directly -- --team just writes more rows to them at
once. `--pitcher` and `--team` upsert independently, so you can build up
coverage team by team without re-running ones you've already done.

## Stuff+ v2: handedness, every pitch, out-of-fold

Three changes to how Stuff+ is trained and scored. The original numbers are kept as StuffPlus_v1 / StuffPlus_Scaled_v1 (in the league tables, the dashboard CSVs and pitcher_report_data.csv) for comparison.

- **Handedness.** hb (pfx_x * 12), release_pos_x and spin_axis are in the catcher's view, so the same shape has opposite signs for a LHP and a RHP, and both hands were trained in one per-pitch-type model. `to_rhp_frame()` mirrors left-handers into the right-hander frame (hb = -hb, release_pos_x = -release_pos_x, spin_axis = 360 - spin_axis) before fitting and scoring. The saved raw columns used by the movement chart are not changed.
- **Score every pitch.** The model is still trained on swings (P(whiff | swing, shape)), but every pitch of that type is scored, roughly doubling each pitcher's sample. Rows carry Pitches (scored) and Swings.
- **Out-of-fold.** 5-fold GroupKFold by pitcher: a pitcher's own pitches never train the model that grades him (same as Location+). The 100 in the ratio StuffPlus is the league's average predicted whiff probability for that pitch type over every pitch. StuffPlus_Scaled = 100 + 10 z vs. the same qualified pool as before.

Out-of-fold validation (data/stuff_plus_validation.csv; held-out pitchers' swings; baseline = always predict the training folds' whiff rate):

| Pitch | Swings | AUC | Log loss | Baseline log loss |
|---|---|---|---|---|
| FF | 106,816 | 0.588 | 0.4731 | 0.4804 |
| SI | 48,225 | 0.548 | 0.3873 | 0.3678 |
| FC | 25,519 | 0.529 | 0.5088 | 0.5034 |
| SL | 48,547 | 0.563 | 0.6179 | 0.6235 |
| ST | 23,773 | 0.540 | 0.6142 | 0.6131 |
| CU | 19,084 | 0.556 | 0.6058 | 0.6069 |
| KC | 5,505 | 0.579 | 0.6558 | 0.6456 |
| CH | 35,337 | 0.534 | 0.6064 | 0.6056 |
| FS | 9,484 | 0.513 | 0.6371 | 0.6279 |
| SV | 1,490 | 0.511 | 0.6817 | 0.6173 |
| CS | 134 | 0.505 | 0.9291 | 0.6186 |

Honest reading: AUC is above 0.5 for every pitch type, so shape carries *some* ranking signal about whiffs on pitchers the model has never seen, but it is weak (0.51 to 0.59). Log loss only beats the constant baseline for FF, SL and CU; for the rest the probabilities are over-confident on new pitchers (worst: SI, SV, CS). Single-swing whiffs depend heavily on location, count and the batter, none of which are features here. Stuff+ is therefore best read as a relative ranking of pitch shape, not a calibrated whiff probability. FO (1 pitcher) and KN are not scored, since GroupKFold needs 5+ pitchers.

v1 vs. v2 (qualified pitchers, data/stuff_plus_v1_comparison.csv): Pearson r = 0.43 overall, 0.40 RHP, 0.78 LHP. The RHP number is dominated by one outlier: Tyler Rogers (submarine release) goes from 100 to 280 scaled, because when he is held out no training pitcher has a release like his, so the out-of-fold model extrapolates. Excluding him, RHP r = 0.81 vs. LHP 0.78, and by rank (Spearman) LHP moved more (0.70 vs. 0.87). A diagnostic refit of v1 with *only* the mirroring change correlates 0.983 with v1 for RHP and 0.968 for LHP: mirroring moves lefties more, as expected, but most of the total change comes from out-of-fold, every-pitch scoring.

## Small samples: stabilization and regression toward the mean

Lowering MIN_PITCHES_FOR_INCLUSION would make an 80-pitch reliever look as certain as a 3,000-pitch starter. Instead every metric is regressed toward the qualified league mean by how much it can be trusted at that sample size.

**Stabilization (analysis/stabilization.py, data/stabilization.csv).** For each metric and each sample size n, every qualified pitcher with at least 2n units gets two random, non-overlapping samples of n units; the metric is computed on each half and correlated across pitchers (5 random draws, averaged; a point needs 30+ pitchers). r(n) = n / (n + k) is then fit by least squares, so k is the sample size at which reliability = 0.5. Grids: 25/50/100/200/400 pitches, 25/50/100/200 PA, 5/10/15 appearances.

| Metric | Unit | k |
|---|---|---|
| StuffPlus_Scaled (overall) | pitches | 1 |
| StuffPlus_Scaled per pitch type | pitches | 0 to 2 |
| LocationPlus (overall) | pitches | 286 |
| LocationPlus per pitch type | pitches | 81 (FS) to 244 (FF) |
| EdgePct | pitches | 5,765 |
| MeatballPct | pitches | 930 |
| BBPct | PA | 241 |
| VolatilityScore, starters | appearances | 7 |
| VolatilityScore, relievers | appearances | 8 |

Stuff+ stabilizes almost immediately because it's a function of pitch shape, which barely changes from pitch to pitch within a pitcher. That says the *model's output* is repeatable, not that it predicts outcomes well (see the validation above). Edge% is close to pure noise at any realistic sample. Pitch types with too few pitchers for their own k (CS, SV, FO) use the overall k.

**Regression.** For each metric, `Regressed = (n * observed + k * league_mean) / (n + k)` and `Reliability = n / (n + k)`, where league_mean is the qualified pool's mean on that metric's scale (100 for Stuff+ and Location+, the qualified mean for Edge%, Meatball%, BB%, VolatilityScore). Raw columns are never overwritten; each metric gets a *_Reg column (StuffPlus_Scaled_Reg, LocationPlus_Reg, EdgePct_Reg, MeatballPct_Reg, BBPct_Reg, CommandProxyLegacy_Reg, VolatilityScore_Reg / VolatilityPercentile_Reg, AsymmetricUpsideIndex_Reg) plus a *_Reliability column. League means, SDs, z-scores, percentiles and colors still come from the **qualified pool only**, so noisy small samples don't move anyone else's grade. The qualified pool is unchanged (470 pitchers for Stuff+/Location+/upside; 205 starters + 296 relievers for volatility).

**Minimums (config.py).**
- MIN_PITCHES_FOR_INCLUSION = 500: the grading pool (unchanged).
- MIN_PITCHES_TO_DISPLAY = 50: any pitcher at or above this gets regressed values for every metric; below it shows "n/a".
- MIN_PITCHES_PER_TYPE_TO_DISPLAY = 15: per-pitch-type Stuff+/Location+ is shown (regressed) from 15 pitches.
- USE_REGRESSED = True: asymmetric_upside.py, risk_adjusted_value.py and export_report_data.py use the *_Reg columns (raw columns stay in every output).
- Volatility for non-qualified pitchers uses whichever unit (starts or relief appearances) they have more of, falling back to all their appearances of 10+ pitches; with fewer than 2 appearances there is no SD, so the regressed value is the league mean with reliability near 0.

**Dashboard.** The regressed value is the main number with its sample size next to it (e.g. "112 (140 pitches)"); hover the card for the raw value. A "small sample" tag appears when Reliability < 0.5, and the red/blue intensity is multiplied by Reliability, so low-sample numbers stay closer to gray. Tables show the raw values in their own columns.

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

**Stuff+ columns.** StuffPlus is the ratio (predicted whiff probability / league average for the pitch type x 100), unchanged. StuffPlus_Scaled = 100 + 10 * z(StuffPlus) against the league-wide qualified pool (per pitch type at the pitch-type level), which puts it on exactly the same scale as Location+. StuffPlus_Scaled_Reg regresses it toward 100 for small samples, and StuffPlus_v1 / StuffPlus_Scaled_v1 keep the original model's values. The dashboard's "Stuff+" tiles, bar chart, table and Stuff+ vs. Location+ scatter show StuffPlus_Scaled_Reg.

## Dashboard

streamlit run dashboard/app.py

### Updating the deployed dashboard

The public app on Streamlit Community Cloud runs from GitHub and only sees committed files. It reads nothing from data/ (gitignored raw caches), only the CSVs in dashboard/data/.

1. Rerun the analysis scripts locally: `python -m analysis.run_all` (or the per-team run order above, ending with `python -m analysis.export_report_data`). run_all prints every dashboard/data/ file's size and flags anything over 25MB (GitHub's web upload limit). The two per-pitch files are written as .csv.gz for that reason, and the movement file keeps only the columns app.py reads.
2. Commit the updated files: `git add dashboard/data/ && git commit -m "Refresh dashboard data"`.
3. Push. Streamlit Cloud redeploys automatically.

The Pitcher Report has a fastball velocity by appearance chart: average primary-fastball velocity per game (FF, else SI, else FC), a dashed season average, a ±1 SD band, and IL stints from injury_stints.csv shaded. The pitch arsenal table, the Stuff+/Location+ bar chart and the pitch type key are all ordered by usage (most used first).

The Team Value Board tab ranks the selected team by multi-year surplus (Stuff+, risk-adjusted WAR, 2025 salary, years of control) with a WAR vs. salary scatter (point size = years of control, color = multi-year surplus). The Pitcher Report shows that pitcher's surplus by control year. Both need export_report_data.py to have been run.

The sidebar lets you navigate every MLB team in both leagues. After
`python -m analysis.run_all`, every team lists its real 2025 pitching staff,
each with real Stuff+ Proxy results. Picking a team that hasn't been run says
so. STL loads without any salaries.csv rows: pre-arb pitchers get the league
minimum, and everyone else shows n/a surplus and is listed in
data/missing_contract_report.csv. There's no sample/placeholder data mode
anymore -- everything the dashboard shows is real, or explicitly says it
hasn't been run yet.

analysis/export_report_data.py is the hand-off point between the full
analysis pipeline and the dashboard: it joins all four modules' output into
one flat CSV with a fixed column contract. It requires all four modules to
have been run league-wide first (see "Known limitations" below on the OOM
risk of that for stuff_plus_proxy.py).

## Known limitations

- A full league-wide season pull in one `load_season()` call can OOM on a memory-limited machine during pybaseball's concat step -- confirmed on an 8GB box around the 6-month mark. `load_season_monthly()` works around this (see above) and is used by every league-wide step, including stuff_plus_proxy.py with no flags.
- Stuff+ Proxy is a gradient-boosted classifier on shape features, not a full replacement for proprietary gradient-boosted Stuff+ models. Out of fold it ranks whiffs only weakly (AUC 0.51 to 0.59) and is over-confident on new pitchers for most pitch types (see "Stuff+ v2"). Extreme release points (e.g. Tyler Rogers) are extrapolated and can get implausible grades.
- IL history comes from the MLB Stats API transactions feed (analysis/injury_history.py), not manual collection anymore. The feed occasionally leaves out an activation, so when one's missing the return date is taken from the pitcher's first MLB appearance after the placement. Only pitchers you've run it for (--team/--pitcher) have injury data; everyone else gets 0 injury risk in risk_adjusted_value.py (InjuryDataPulled = False).
- Salaries come from data/salaries.csv, built by hand (not scraped) from saved FanGraphs RosterResource 2025 payroll pages for 29 teams (536 rows; PIT and LAD hand-checked; STL not included yet, so STL pitchers get only the pre-arb league-minimum default and show up in the missing-contract report). Columns: PitcherId, Pitcher, Team, Season, Salary, AAV, ContractStatus, YearsControl, PreArbYearsLeft, ArbYearsLeft, GuaranteedFuture, plus reference columns the pipeline carries but doesn't use (TeamPaid2025, Origin, Source, Notes, FanGraphsId, ServiceTime). Team is a FanGraphs code (ARI, not AZ), renamed SalariesTeam on load. contract_status.load_salaries() drops rows with a blank PitcherId and reads it as an integer. A pitcher traded midseason has one row per team; the last row is kept, and any whose rows disagree on Salary or contract fields are printed on every run. Contract status and years of control can be estimated from debut dates (contract_status.py), but EstServiceYears is SEASON minus debut year: it ignores option years, IL, partial seasons and Super Two, so hand-entered values always win.
- simple_projected_war in risk_adjusted_value.py is an uncalibrated placeholder until regressed against actual historical WAR for a training sample. It uses the same starter-based WAR anchor for relievers, so reliever WAR is overstated.
- Multi-year surplus holds ProjectedWAR flat across every control year (no aging curve yet), and the arb salary percentages, $/WAR growth, discount rate, injury shrinkage and control-year cap in config.py are assumptions, not fitted values.
- Pitchers under config.MIN_PITCHES_FOR_INCLUSION (500) pitches now get regressed values (down to config.MIN_PITCHES_TO_DISPLAY = 50) and a risk-adjusted value, but the WAR proxy has no playing-time term: a 150-pitch reliever regressed to league-average talent still projects near LEAGUE_AVG_STARTER_WAR, so small-sample surplus values are overstated even more than qualified relievers'. The k values are fit once on 2025 qualified pitchers and assumed to hold for everyone.
- Location+ still can't know the catcher's target: it measures location quality given the count, not intent. A pitcher who hits a spot the catcher set on the edge looks the same as one who missed there. Because the count is a feature, it also partly reflects getting into bad counts, not only where each pitch lands.
- The legacy Command Proxy (CommandProxyLegacy in asymmetric_upside.py) is a zone-discipline stand-in (edge%/meatball%/BB%) and has the same no-target limitation.
