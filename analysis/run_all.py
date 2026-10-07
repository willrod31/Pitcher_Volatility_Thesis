"""
Run the whole pipeline for every MLB team (or --teams PIT,LAD), then export
the dashboard data and the missing-contract report.

    python -m analysis.run_all
    python -m analysis.run_all --teams PIT,LAD

Per team, the modules run in the usual order: injury_history,
contract_status, stuff_plus_proxy, location_plus_proxy, asymmetric_upside,
volatility_discount, risk_adjusted_value; export_report_data runs once at
the end. League-wide training and grading happens ONCE, not per team:

1. Rosters: each team's statcast_team_<TEAM> cache is split out of the
   cached league monthly pulls (data_acquisition.seed_team_caches_from_league)
   instead of one Statcast pull per team.
2. Per team: injury_history and contract_status (MLB Stats API pulls,
   cached). These run for every team first, because volatility_discount and
   risk_adjusted_value grade the league using every team's IL history.
3. League, once: the league-wide Statcast season is loaded month by month
   (load_season_monthly, fits in 8GB) and shared; Stuff+ and Location+ reuse
   their cached out-of-fold scores; then upside, volatility and
   risk-adjusted value are graded league-wide.
4. Per team: each module's team view (graded vs. the league) and the
   Stuff+ dashboard exports, sliced from the shared league frame by PitcherId
   (every pitch the pitcher threw all season, any team).
5. export_report_data, then data/missing_contract_report.csv.

Each team's console output goes to logs/run_<TEAM>.txt, the league steps to
logs/run_league.txt. A failed step is logged with its traceback, the run
continues, and every failure is listed at the end.
"""
import argparse
import contextlib
import time
import traceback
from pathlib import Path

import pandas as pd

from config import END_DATE, START_DATE
from analysis import (
    asymmetric_upside, contract_status, export_report_data, injury_history, location_plus_proxy,
    risk_adjusted_value, stuff_plus_proxy, volatility_discount,
)
from analysis.data_acquisition import load_season_monthly, seed_team_caches_from_league, team_roster_ids
from analysis.utils import CACHE_DIR

# Statcast team codes (AZ, not ARI), same list as dashboard/app.py
TEAMS = [
    "AZ", "ATH", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET", "HOU", "KC", "LAA", "LAD",
    "MIA", "MIL", "MIN", "NYM", "NYY", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]
TEAM_ALIASES = {"ARI": "AZ"}   # FanGraphs / salaries.csv code -> Statcast
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
REPORT_PATH = CACHE_DIR / "missing_contract_report.csv"
REPORT_COLUMNS = ["PitcherId", "Pitcher", "Team", "Problem", "ContractStatus_est", "EstYearsControl", "Pitches2025"]
PROBLEM_NO_SALARY = "no salary"
PROBLEM_NO_CONTROL = "missing (1 yr assumed)"
PROBLEM_MISMATCH = "status mismatch"
SIZE_LIMIT_MB = 25

# One league frame for Stuff+ grading, the per-team Stuff+ exports and asymmetric_upside
LEAGUE_COLUMNS = list(dict.fromkeys(stuff_plus_proxy.SCOPE_COLUMNS + asymmetric_upside.RAW_COLUMNS))


class Runner:
    """Runs steps with stdout/stderr appended to a log file; records failures instead of stopping."""

    def __init__(self):
        self.failures: list[tuple[str, str, str]] = []

    def step(self, scope: str, name: str, fn, *args, **kwargs):
        log_path = LOG_DIR / f"run_{scope}.txt"
        start = time.time()
        with open(log_path, "a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(f"\n{'=' * 20} {name} ({scope}) {'=' * 20}")
            try:
                result = fn(*args, **kwargs)
                ok = True
            except Exception as exc:
                traceback.print_exc()
                self.failures.append((scope, name, f"{type(exc).__name__}: {exc}"))
                result, ok = None, False
        print(f"  {scope:<6} {name:<22} {'ok' if ok else 'FAILED':<7} {time.time() - start:6.1f}s", flush=True)
        return result


def check_salaries():
    """Step-1 checks: blank IDs dropped, integer ids, duplicate rows whose Salary differs."""
    salaries = contract_status.load_salaries(verbose=True)
    if salaries is None:
        return None
    print(f"salaries.csv: {len(salaries)} unique PitcherIds (dtype {salaries['PitcherId'].dtype}), "
          f"{salaries['SalariesTeam'].nunique()} teams; columns: {', '.join(salaries.columns)}")
    return salaries


def team_scope(league: pd.DataFrame, team: str) -> pd.DataFrame:
    """Every pitch thrown all season by anyone on `team`'s roster, labeled with `team`."""
    ids = team_roster_ids(team)["PitcherId"]
    return league[league["pitcher"].isin(ids)].assign(PitcherTeam=team)


def build_missing_report(pitches_by_id: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(problem rows, per pitcher x team contract resolution) for everyone on a team in the dashboard's Stuff+ table.

    Same fallbacks as risk_adjusted_value.resolve_contracts, applied to every
    roster pitcher (including those under MIN_PITCHES_TO_DISPLAY, who the
    dashboard still lists). One row per pitcher x team, so a traded pitcher
    is counted under each team; several problems are joined with "; ".
    """
    roster = pd.read_csv(stuff_plus_proxy.DASHBOARD_PATH, usecols=["PitcherId", "Pitcher", "PitcherTeam"])
    roster = roster.drop_duplicates(["PitcherId", "PitcherTeam"]).rename(columns={"PitcherTeam": "Team"})
    resolved = risk_adjusted_value.resolve_contracts(roster[["PitcherId", "Pitcher"]].drop_duplicates("PitcherId"),
                                                     contract_status.load_salaries())
    cols = ["PitcherId", "Salary", "SalarySource", "ControlSource", "StatusMismatch", "ContractStatus_est", "EstYearsControl"]
    rows = roster.merge(resolved[cols], on="PitcherId", how="left")
    rows["Pitches2025"] = rows["PitcherId"].map(pitches_by_id).fillna(0).astype(int)

    problems = pd.DataFrame({
        PROBLEM_NO_SALARY: rows["Salary"].isna(),
        PROBLEM_NO_CONTROL: rows["ControlSource"] == PROBLEM_NO_CONTROL,
        PROBLEM_MISMATCH: rows["StatusMismatch"].fillna(False).astype(bool),
    })
    rows["Problem"] = problems.apply(lambda r: "; ".join(r.index[r]), axis=1)
    report = rows[rows["Problem"] != ""].sort_values(["Pitches2025", "Team"], ascending=[False, True])
    return report[REPORT_COLUMNS].reset_index(drop=True), rows


def print_report_summary(report: pd.DataFrame, rows: pd.DataFrame, teams: list[str]):
    exploded = report.assign(Problem=report["Problem"].str.split("; ")).explode("Problem", ignore_index=True)
    counts = pd.crosstab(exploded["Team"], exploded["Problem"]).reindex(
        index=teams, columns=[PROBLEM_NO_SALARY, PROBLEM_NO_CONTROL, PROBLEM_MISMATCH], fill_value=0)
    counts["pitchers flagged"] = report.groupby("Team").size().reindex(teams, fill_value=0)
    counts.loc["TOTAL"] = counts.sum()
    print("\n── Missing contract data per team (pitcher x team rows; one pitcher can have several problems) ──")
    print(counts.to_string())
    print(f"{len(report)} flagged pitcher x team rows, {report['PitcherId'].nunique()} unique pitchers "
          f"-> {REPORT_PATH}")

    source = rows["SalarySource"].map({"hand": "salaries.csv", "league min": "league min default"}).fillna("no salary")
    by_source = pd.crosstab(rows["Team"], source).reindex(
        index=teams, columns=["salaries.csv", "league min default", "no salary"], fill_value=0)
    by_source.loc["TOTAL"] = by_source.sum()
    print("\n── Where each dashboard pitcher's 2025 Salary came from ──")
    print(by_source.to_string())


def print_file_sizes():
    print(f"\n── dashboard/data/ file sizes (limit {SIZE_LIMIT_MB}MB) ──")
    for f in sorted(export_report_data.DASHBOARD_DATA.iterdir()):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name:<34} {mb:8.2f} MB" + ("   <-- OVER LIMIT" if mb > SIZE_LIMIT_MB else ""))


def run(teams: list[str]):
    LOG_DIR.mkdir(exist_ok=True)
    for scope in teams + ["league"]:
        (LOG_DIR / f"run_{scope}.txt").write_text(f"python -m analysis.run_all, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    runner = Runner()

    print("── salaries.csv check ──")
    check_salaries()

    print(f"\n── 1. Rosters for {len(teams)} team(s) ──")
    seeded = runner.step("league", "seed team caches", seed_team_caches_from_league, teams)
    print(f"  split {len(seeded or [])} team cache(s) out of the league monthly pulls")

    print("\n── 2. IL history + contract estimates, per team ──")
    for team in teams:
        runner.step(team, "injury_history", injury_history.run, team=team)
        runner.step(team, "contract_status", contract_status.run, team=team)

    print("\n── 3. League-wide grading (once) ──")
    league = runner.step("league", "load league season", load_season_monthly, START_DATE, END_DATE, columns=LEAGUE_COLUMNS)
    if league is None:
        league = pd.DataFrame(columns=LEAGUE_COLUMNS + ["Pitcher", "PitcherTeam"])
    stuff = runner.step("league", "stuff_plus_proxy", stuff_plus_proxy.grade_league, train_df=league)
    if stuff is not None:
        runner.step("league", "stuff v1 comparison", stuff_plus_proxy.compare_to_v1, stuff[1])
    location = runner.step("league", "location_plus_proxy", location_plus_proxy.grade_league)
    upside = runner.step("league", "asymmetric_upside", asymmetric_upside.grade_league, df=league)
    volatility = runner.step("league", "volatility_discount", volatility_discount.grade_league)
    if volatility is not None:
        runner.step("league", "volatility regressions", volatility_discount.run_regressions, volatility[1])
    value = runner.step("league", "risk_adjusted_value", risk_adjusted_value.grade_league)

    print("\n── 4. Team views + dashboard exports, per team ──")
    league_pitches = pd.read_parquet(stuff_plus_proxy.LEAGUE_PITCHES_PATH) if stuff is not None else None
    for team in teams:
        if stuff is not None:
            runner.step(team, "stuff_plus_proxy", lambda t=team: stuff_plus_proxy.export_scope(
                stuff[0], stuff[1], team_scope(league, t), t, t, league_pitches))
        if location is not None:
            runner.step(team, "location_plus_proxy", location_plus_proxy.show, location, team=team)
        if upside is not None:
            runner.step(team, "asymmetric_upside", asymmetric_upside.show, upside, team)
        if volatility is not None:
            runner.step(team, "volatility_discount", volatility_discount.show, volatility[0], team)
        if value is not None:
            runner.step(team, "risk_adjusted_value", risk_adjusted_value.show, value, team)

    print("\n── 5. Dashboard export + missing-contract report ──")
    runner.step("league", "export_report_data", export_report_data.run)
    pitches_by_id = league.groupby("pitcher").size()
    league = league_pitches = None   # free memory before the report
    result = runner.step("league", "missing contract report", build_missing_report, pitches_by_id)
    if result is not None:
        report, rows = result
        report.to_csv(REPORT_PATH, index=False)
        print_report_summary(report, rows, sorted(rows["Team"].unique()))
        print(f"\n── Top 20 rows of {REPORT_PATH.name} ──")
        print(report.head(20).to_string(index=False))
    print_file_sizes()

    print(f"\nLogs: {LOG_DIR}/run_<TEAM>.txt, run_league.txt")
    if runner.failures:
        print(f"\n{len(runner.failures)} FAILED step(s):")
        for scope, name, err in runner.failures:
            print(f"  {scope:<6} {name:<22} {err}")
    else:
        print("\nNo failures.")
    return runner.failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pipeline for every MLB team, then export the dashboard data.")
    parser.add_argument("--teams", default=None, help="Comma-separated Statcast team codes, e.g. --teams PIT,LAD (default: all 30)")
    args = parser.parse_args()
    selected = [t.strip().upper() for t in args.teams.split(",")] if args.teams else TEAMS
    selected = [TEAM_ALIASES.get(t, t) for t in selected]
    unknown = sorted(set(selected) - set(TEAMS))
    if unknown:
        parser.error(f"unknown team code(s) {unknown}; use Statcast codes: {', '.join(TEAMS)}")
    run(selected)
