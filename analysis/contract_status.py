"""
Estimated contract status from MLB debut dates (MLB Stats API /people).

Fills in ContractStatus / years of control so real salaries only have to be
looked up by hand for arb and free-agent pitchers. risk_adjusted_value.py
uses these only where data/salaries.csv is blank -- hand-entered values
always win.

EstServiceYears is an APPROXIMATION: full seasons from the debut season up
to (not including) config.SEASON, i.e. SEASON - debut year. It ignores
option years, time in the minors, IL, partial seasons and Super Two, so a
pitcher who debuted in September counts the same as one who debuted on
Opening Day.

    EstServiceYears < 3   -> pre-arb
    3 to 5                -> arb
    6 or more             -> FA-eligible
    EstYearsControl       = max(0, 6 - EstServiceYears), counting SEASON as year 1
"""
import argparse
import json

import pandas as pd

from config import SEASON
from analysis.injury_history import API_BASE, _get_json, resolve_pitchers
from analysis.utils import CACHE_DIR

DEBUT_CACHE_PATH = CACHE_DIR / "mlb_debut_dates.json"
OUT_PATH = CACHE_DIR / "contract_status_est.csv"
SALARIES_PATH = CACHE_DIR / "salaries.csv"
BATCH_SIZE = 100

PRE_ARB_YEARS = 3
FULL_CONTROL_YEARS = 6

OUT_COLUMNS = [
    "PitcherId", "Pitcher", "MLBDebutDate", "EstServiceYears", "ContractStatus_est",
    "EstYearsControl", "EstPreArbYearsLeft", "EstArbYearsLeft",
]


def status_group(status) -> str | None:
    """Collapse hand-entered statuses (FA, guaranteed, veteran, ...) to pre-arb / arb / FA-eligible."""
    if pd.isna(status) or not str(status).strip():
        return None
    s = str(status).strip().lower()
    if s in ("pre-arb", "prearb", "pre arb"):
        return "pre-arb"
    if s in ("arb", "arbitration"):
        return "arb"
    return "FA-eligible"


def fetch_debut_info(mlbam_ids, force_refresh: bool = False) -> dict[int, str | None]:
    """{mlbam_id: mlbDebutDate or None}, batched against /people, cached in data/."""
    cache = json.loads(DEBUT_CACHE_PATH.read_text()) if DEBUT_CACHE_PATH.exists() else {}
    ids = sorted({int(i) for i in mlbam_ids})
    missing = ids if force_refresh else [i for i in ids if str(i) not in cache]

    for n in range(0, len(missing), BATCH_SIZE):
        batch = missing[n:n + BATCH_SIZE]
        people = _get_json(f"{API_BASE}/people", {"personIds": ",".join(map(str, batch))}).get("people", [])
        found = {p["id"]: p.get("mlbDebutDate") for p in people}
        for i in batch:
            cache[str(i)] = found.get(i)
    if missing:
        DEBUT_CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return {i: cache.get(str(i)) for i in ids}


def estimate_contract_status(pitchers: pd.DataFrame, season: int = SEASON, force_refresh: bool = False) -> pd.DataFrame:
    """One row per pitcher in `pitchers` (PitcherId, Pitcher). See module docstring for the rules."""
    pitchers = pitchers[["PitcherId", "Pitcher"]].drop_duplicates("PitcherId").copy()
    debuts = fetch_debut_info(pitchers["PitcherId"], force_refresh=force_refresh)
    pitchers["MLBDebutDate"] = pitchers["PitcherId"].map(debuts)

    debut_year = pd.to_datetime(pitchers["MLBDebutDate"]).dt.year
    service = (season - debut_year).clip(lower=0)
    pitchers["EstServiceYears"] = service.astype("Int64")

    status = pd.Series(pd.NA, index=pitchers.index, dtype="object")
    status[service < PRE_ARB_YEARS] = "pre-arb"
    status[service.between(PRE_ARB_YEARS, FULL_CONTROL_YEARS - 1)] = "arb"
    status[service >= FULL_CONTROL_YEARS] = "FA-eligible"
    pitchers["ContractStatus_est"] = status

    control = (FULL_CONTROL_YEARS - service).clip(lower=0)
    pre_arb_left = (PRE_ARB_YEARS - service).clip(lower=0)
    pitchers["EstYearsControl"] = control.astype("Int64")
    pitchers["EstPreArbYearsLeft"] = pre_arb_left.astype("Int64")
    pitchers["EstArbYearsLeft"] = (control - pre_arb_left).astype("Int64")
    return pitchers[OUT_COLUMNS].reset_index(drop=True)


def save_estimates(est: pd.DataFrame) -> pd.DataFrame:
    """Upsert by PitcherId, same as the injury/Stuff+ exports."""
    rows = est
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        existing = existing[~existing["PitcherId"].isin(est["PitcherId"])]
        rows = pd.concat([existing, est], ignore_index=True)
    rows.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(rows)} row(s) to {OUT_PATH} ({len(est)} this run)")
    return rows


def disagreements(est: pd.DataFrame, salaries: pd.DataFrame) -> pd.DataFrame:
    """Pitchers whose hand-entered ContractStatus differs from the estimate.

    FA / guaranteed / veteran etc. all count as agreeing with FA-eligible.
    """
    hand = salaries[["PitcherId", "ContractStatus"]].dropna(subset=["ContractStatus"])
    both = est.merge(hand, on="PitcherId", how="inner")
    differs = both["ContractStatus"].map(status_group) != both["ContractStatus_est"]
    return both[differs][["PitcherId", "Pitcher", "MLBDebutDate", "EstServiceYears", "ContractStatus_est", "ContractStatus"]]


def run(pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    pitchers = resolve_pitchers(pitcher, team)
    est = estimate_contract_status(pitchers, force_refresh=force_refresh)
    print(f"── Estimated contract status ({pitcher or team}, as of {SEASON}) ──")
    print(est.sort_values("EstYearsControl", ascending=False).to_string(index=False))
    save_estimates(est)

    if SALARIES_PATH.exists():
        diff = disagreements(est, pd.read_csv(SALARIES_PATH))
        print(f"\n{len(diff)} pitcher(s) where the estimate disagrees with salaries.csv ContractStatus:")
        if not diff.empty:
            print(diff.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate contract status / years of control from MLB debut dates.")
    parser.add_argument("--pitcher", default=None, help='One pitcher, e.g. --pitcher "Paul Skenes"')
    parser.add_argument("--team", default=None, help="Every pitcher on a team's Stuff+ roster, e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Re-pull debut dates instead of using the cache")
    args = parser.parse_args()
    run(pitcher=args.pitcher, team=args.team, force_refresh=args.refresh)
