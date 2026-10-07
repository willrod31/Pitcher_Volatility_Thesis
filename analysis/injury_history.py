"""
IL history from the MLB Stats API transactions feed, so injury data doesn't
have to be collected by hand anymore.

Pulls every transaction for a pitcher, turns the IL moves into one row per
stint (data/injury_stints.csv), then one row per pitcher
(data/injury_summary.csv) for volatility_discount.py.

Queries by playerId instead of teamId so time with other teams counts too.

Notes on how stints get built:
- starts at "placed ... on the N-day injured list" (retro date if given)
- a 15 -> 60 day transfer is the same stint, just updates ILType
- ends at the next activation/reinstatement by an MLB team. Older ones are
  just "activated RHP X." with no list named, those count too
- the feed sometimes skips activations (Mlodzinski 2024, Ferguson 2022 both
  pitched between two placements), so if there's no activation I use their
  first MLB appearance after going on the IL
- still nothing -> end at the last day of the regular season, OpenEnded=True
- DaysMissed is regular season days only
"""
import argparse
import json
import re
import time

import pandas as pd
import requests

from config import INJURY_LOOKBACK_SEASONS, SEASON
from analysis.data_acquisition import attach_pitcher_names, lookup_mlbam_id, team_roster_ids
from analysis.utils import GAMELOG_DIR, MLB_API_DIR, RESULTS_DIR, TRANSACTIONS_DIR

API_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_SLEEP_SECONDS = 0.5
DUPLICATE_PLACEMENT_DAYS = 3
SEASON_DATES_PATH = MLB_API_DIR / "mlb_season_dates.json"

STINTS_PATH = RESULTS_DIR / "injury_stints.csv"
SUMMARY_PATH = RESULTS_DIR / "injury_summary.csv"

# MLB team ids (these don't change when a team renames/moves). Anything else
# is a minor league team, which we skip.
MLB_TEAM_IDS = frozenset({
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 158,
})

PLACED_RE = re.compile(r"\bplaced\b.*?\bon the (\d+)-day injured list", re.IGNORECASE)
TRANSFER_RE = re.compile(r"\btransferred\b.*?\bto the (\d+)-day injured list", re.IGNORECASE)
RETRO_RE = re.compile(r"retroactive to ([A-Z][a-z]+ \d{1,2}, \d{4})")
INJURY_TEXT_RE = re.compile(r"injured list(?: retroactive to [A-Za-z]+ \d{1,2}, \d{4})?\.\s*(.*)$", re.IGNORECASE)
RETURN_RE = re.compile(r"\b(activated|reinstated)\b", re.IGNORECASE)

# checked in order, first match wins (so UCL before elbow, lat before shoulder)
BODY_PART_KEYWORDS = [
    ("elbow", r"\bucl\b|tommy john|ulnar collateral|\belbow\b"),
    ("forearm", r"\bflexor\b|\bforearm\b|\bpronator\b"),
    ("lat", r"\blats?\b|latissimus"),
    ("shoulder", r"rotator cuff|\blabrum\b|\bshoulder\b|\bscapula"),
    ("biceps/triceps", r"\bbiceps?\b|\btriceps?\b"),
    ("oblique", r"\boblique\b|\bintercostal\b|\brib\b|\bside\b"),
    ("back", r"\bback\b|\blumbar\b|\bspine\b|\bthoracic\b"),
    ("neck", r"\bneck\b|\bcervical\b|\btrapezius\b"),
    ("hip/groin", r"\bhip\b|\bgroin\b|\badductor\b|\bgluteal\b|\bglute\b"),
    ("hamstring", r"\bhamstring\b"),
    ("quad", r"\bquad(riceps)?\b"),
    ("knee", r"\bknee\b|\bmeniscus\b|\bacl\b|\bpatellar\b"),
    ("calf", r"\bcalf\b|\bachilles\b"),
    ("ankle/foot", r"\bankle\b|\bfoot\b|\btoe\b|\bheel\b|\bplantar\b"),
    ("hand/wrist", r"\bwrist\b|\bhand\b|\bfinger\b|\bthumb\b|\bblister\b|\bnail\b"),
    ("head", r"\bconcussion\b|\bhead\b|\bface\b|\bjaw\b|\beye\b"),
    ("illness", r"\bil+ness\b|\bcovid\b|\bvirus\b|\bviral\b|\binfection\b|\bflu\b"),
]
ARM_BODY_PARTS = {"elbow", "forearm", "lat", "shoulder"}

STINT_COLUMNS = [
    "PitcherId", "Pitcher", "Season", "StintId", "StartDate", "EndDate", "DaysMissed", "ILType",
    "Injury", "BodyPart", "IsArmInjury", "OpenEnded", "SourceDescription",
]
SUMMARY_COLUMNS = [
    "PitcherId", "Pitcher", "ILStints", "TotalDaysMissed", "DaysMissedLastSeason",
    "ArmILStints", "Had60DayIL", "MostRecentInjury", "MostRecentILEnd",
]


def _get_json(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return resp.json()


def default_seasons() -> tuple[int, int]:
    return SEASON - INJURY_LOOKBACK_SEASONS, SEASON


def fetch_transactions(mlbam_id: int, start_season: int | None = None, end_season: int | None = None, force_refresh: bool = False) -> list[dict]:
    """All transactions for one player, cached in data/."""
    default_start, default_end = default_seasons()
    start_season = start_season or default_start
    end_season = end_season or default_end

    raw_path = TRANSACTIONS_DIR / f"transactions_{mlbam_id}_{start_season}_{end_season}.json"
    if raw_path.exists() and not force_refresh:
        return json.loads(raw_path.read_text())

    data = _get_json(f"{API_BASE}/transactions", {
        "playerId": mlbam_id,
        "startDate": f"{start_season}-01-01",
        "endDate": f"{end_season}-12-31",
    })
    transactions = data.get("transactions", [])
    raw_path.write_text(json.dumps(transactions))
    return transactions


def season_dates(season: int, force_refresh: bool = False) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Opening day and last regular season day.

    Not using regularSeasonStartDate for opening day since that's the Tokyo
    series in 2025 (3/18), using the first day with a real slate of games.
    """
    cache = json.loads(SEASON_DATES_PATH.read_text()) if SEASON_DATES_PATH.exists() else {}
    key = str(season)
    if key not in cache or force_refresh:
        info = _get_json(f"{API_BASE}/seasons/{season}", {"sportId": 1})["seasons"][0]
        first_listed = pd.Timestamp(info["regularSeasonStartDate"])
        schedule = _get_json(f"{API_BASE}/schedule", {
            "sportId": 1, "gameType": "R",
            "startDate": first_listed.strftime("%Y-%m-%d"),
            "endDate": (first_listed + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        })
        full_slate = [d["date"] for d in schedule.get("dates", []) if d.get("totalGames", 0) >= 5]
        cache[key] = {
            "opening_day": full_slate[0] if full_slate else info["regularSeasonStartDate"],
            "season_end": info["regularSeasonEndDate"],
        }
        SEASON_DATES_PATH.write_text(json.dumps(cache, indent=1))
    return pd.Timestamp(cache[key]["opening_day"]), pd.Timestamp(cache[key]["season_end"])


def fetch_appearance_dates(mlbam_id: int, season: int, force_refresh: bool = False) -> list[pd.Timestamp]:
    """Dates they pitched in an MLB regular season game that year."""
    raw_path = GAMELOG_DIR / f"gamelog_{mlbam_id}_{season}.json"
    if raw_path.exists() and not force_refresh:
        dates = json.loads(raw_path.read_text())
    else:
        data = _get_json(f"{API_BASE}/people/{mlbam_id}/stats", {
            "stats": "gameLog", "group": "pitching", "season": season, "sportId": 1,
        })
        splits = data["stats"][0]["splits"] if data.get("stats") else []
        dates = sorted({s["date"] for s in splits if s.get("gameType") == "R"})
        raw_path.write_text(json.dumps(dates))
    return [pd.Timestamp(d) for d in dates]


def first_appearance_after(mlbam_id: int, start: pd.Timestamp, before: pd.Timestamp) -> pd.Timestamp | None:
    """First MLB appearance after going on the IL (if they pitched, they were off it)."""
    for season in range(start.year, before.year + 1):
        for date in fetch_appearance_dates(mlbam_id, season):
            if start < date < before:
                return date
    return None


def classify_body_part(injury: str) -> str:
    text = injury.lower()
    for part, pattern in BODY_PART_KEYWORDS:
        if re.search(pattern, text):
            return part
    return "other"


def _is_mlb_club(txn: dict) -> bool:
    return any(txn.get(side, {}).get("id") in MLB_TEAM_IDS for side in ("toTeam", "fromTeam"))


def _txn_date(txn: dict) -> pd.Timestamp:
    return pd.Timestamp(txn.get("effectiveDate") or txn["date"])


def _injury_text(description: str) -> str:
    match = INJURY_TEXT_RE.search(description)
    return match.group(1).strip() if match else ""


def _is_return(txn: dict) -> bool:
    """Activated/reinstated by an MLB team, or contract selected."""
    desc = txn.get("description", "")
    if txn.get("typeCode") == "SE":
        return True
    if txn.get("typeCode") != "SC" or not RETURN_RE.search(desc):
        return False
    # skip paternity/bereavement/restricted list, but plain "activated RHP X." counts
    return "injured list" in desc.lower() or " list" not in desc.lower()


def parse_il_stints(transactions: list[dict], pitcher_name: str | None = None, infer_missing_returns: bool = True) -> pd.DataFrame:
    """One row per IL stint (split into one row per season if it crosses an offseason).

    infer_missing_returns=False skips the game log lookup.
    """
    txns = {t["id"]: t for t in transactions if "person" in t and t.get("date")}.values()
    txns = [t for t in txns if _is_mlb_club(t)]
    txns = sorted(txns, key=lambda t: (t["date"], t["id"]))

    stints = []
    open_stint = None

    def close(stint, end_date, observed_return):
        if not observed_return and infer_missing_returns:
            before = end_date if end_date is not None else season_dates(stint["start"].year)[1] + pd.Timedelta(days=1)
            returned = first_appearance_after(stint["person"]["id"], stint["start"], before)
            if returned is not None:
                stint["descriptions"].append(f"(no activation listed; return inferred from first MLB appearance {returned.date()})")
                end_date, observed_return = returned, True
        stint["end"] = end_date
        stint["open_ended"] = not observed_return
        stints.append(stint)

    for txn in txns:
        desc = txn.get("description", "")
        if txn.get("typeCode") == "SC" and (placed := PLACED_RE.search(desc)):
            retro = RETRO_RE.search(desc)
            start = pd.Timestamp(pd.to_datetime(retro.group(1), format="%B %d, %Y")) if retro else _txn_date(txn)
            if open_stint is not None and abs((start - open_stint["start"]).days) <= DUPLICATE_PLACEMENT_DAYS:
                # same stint posted twice (ex. placed 3/25, then placed retroactive to 3/25 on 3/28)
                open_stint["il_type"] = int(placed.group(1))
                open_stint["injury"] = _injury_text(desc) or open_stint["injury"]
                open_stint["descriptions"].append(desc)
                continue
            if open_stint is not None:
                # new placement with no activation in between = feed missed the activation
                close(open_stint, max(start, open_stint["start"]), observed_return=False)
            open_stint = {
                "person": txn["person"], "start": start, "il_type": int(placed.group(1)),
                "injury": _injury_text(desc), "descriptions": [desc],
            }
        elif txn.get("typeCode") == "SC" and (transfer := TRANSFER_RE.search(desc)):
            if open_stint is None:
                # placement was before the lookback window
                open_stint = {"person": txn["person"], "start": _txn_date(txn), "il_type": None, "injury": "", "descriptions": []}
            open_stint["il_type"] = int(transfer.group(1))
            open_stint["injury"] = open_stint["injury"] or _injury_text(desc)
            open_stint["descriptions"].append(desc)
        elif open_stint is not None and _is_return(txn) and _txn_date(txn) >= open_stint["start"]:
            open_stint["descriptions"].append(desc)
            close(open_stint, _txn_date(txn), observed_return=True)
            open_stint = None

    if open_stint is not None:
        close(open_stint, None, observed_return=False)

    rows = []
    for n, stint in enumerate(stints, start=1):
        rows.extend(_split_by_season(stint, n, pitcher_name))
    return pd.DataFrame(rows, columns=STINT_COLUMNS)


def _split_by_season(stint: dict, stint_number: int, pitcher_name: str | None) -> list[dict]:
    """Regular season days only, one row per season."""
    start = stint["start"]
    end = stint["end"]
    first_season = start.year
    last_season = end.year if end is not None else first_season

    body_part = classify_body_part(stint["injury"])
    rows = []
    for season in range(first_season, last_season + 1):
        opening_day, season_end = season_dates(season)
        row_start = start if season == first_season else opening_day
        is_last = season == last_season
        row_end = min(end, season_end) if (is_last and end is not None) else season_end
        if season != first_season and row_end < opening_day:
            continue  # back before opening day, didn't miss anything that season
        days = max(0, (min(row_end, season_end) - max(row_start, opening_day)).days)
        rows.append({
            "PitcherId": stint["person"]["id"],
            "Pitcher": pitcher_name or stint["person"]["fullName"],
            "Season": season,
            "StintId": f"{stint['person']['id']}-{stint_number}",
            "StartDate": row_start.date(),
            "EndDate": row_end.date(),
            "DaysMissed": days,
            "ILType": stint["il_type"],
            "Injury": stint["injury"],
            "BodyPart": body_part,
            "IsArmInjury": body_part in ARM_BODY_PARTS,
            "OpenEnded": stint["open_ended"] and is_last,
            "SourceDescription": " | ".join(stint["descriptions"]),
        })
    return rows


def summarize_injuries(stints: pd.DataFrame, pitchers: pd.DataFrame, last_season: int = SEASON) -> pd.DataFrame:
    """One row per pitcher. Pitchers with no IL stints get 0s (not missing)."""
    stints = stints.sort_values(["PitcherId", "StartDate"])
    grouped = stints.groupby("PitcherId")
    per_stint = stints.drop_duplicates("StintId")

    summary = pitchers[["PitcherId", "Pitcher"]].drop_duplicates("PitcherId").set_index("PitcherId")
    summary["ILStints"] = per_stint.groupby("PitcherId").size()
    summary["TotalDaysMissed"] = grouped["DaysMissed"].sum()
    summary["DaysMissedLastSeason"] = stints[stints["Season"] == last_season].groupby("PitcherId")["DaysMissed"].sum()
    summary["ArmILStints"] = per_stint[per_stint["IsArmInjury"]].groupby("PitcherId").size()
    summary["Had60DayIL"] = grouped["ILType"].apply(lambda s: (s == 60).any())
    summary["MostRecentInjury"] = grouped["Injury"].last()
    summary["MostRecentILEnd"] = grouped["EndDate"].last()

    count_cols = ["ILStints", "TotalDaysMissed", "DaysMissedLastSeason", "ArmILStints"]
    summary[count_cols] = summary[count_cols].fillna(0).astype(int)
    summary["Had60DayIL"] = summary["Had60DayIL"].fillna(False).astype(bool)
    summary["MostRecentInjury"] = summary["MostRecentInjury"].fillna("")
    return summary.reset_index()[SUMMARY_COLUMNS]


def build_injury_history(pitchers: pd.DataFrame, start_season: int | None = None, end_season: int | None = None, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (stints, summary) for every pitcher in `pitchers`."""
    default_start, default_end = default_seasons()
    start_season = start_season or default_start
    end_season = end_season or default_end

    frames = []
    for pitcher_id, name in pitchers[["PitcherId", "Pitcher"]].itertuples(index=False):
        transactions = fetch_transactions(int(pitcher_id), start_season, end_season, force_refresh=force_refresh)
        frames.append(parse_il_stints(transactions, pitcher_name=name))
    stints = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STINT_COLUMNS)
    stints = stints[stints["Season"].between(start_season, end_season)]
    summary = summarize_injuries(stints, pitchers, last_season=end_season)
    return stints, summary


def _upsert_csv(rows: pd.DataFrame, path, pitcher_ids) -> pd.DataFrame:
    """Same upsert as the Stuff+ exports, only replaces these pitchers' rows."""
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing["PitcherId"].isin(pitcher_ids)]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(path, index=False)
    return rows


def save_injury_history(stints: pd.DataFrame, summary: pd.DataFrame):
    pitcher_ids = summary["PitcherId"].unique()
    all_stints = _upsert_csv(stints, STINTS_PATH, pitcher_ids)
    all_summary = _upsert_csv(summary, SUMMARY_PATH, pitcher_ids)
    print(f"Saved {len(all_stints)} stint row(s) to {STINTS_PATH} ({len(stints)} this run)")
    print(f"Saved {len(all_summary)} pitcher row(s) to {SUMMARY_PATH} ({len(summary)} this run)")


def resolve_pitchers(pitcher: str | None = None, team: str | None = None) -> pd.DataFrame:
    """PitcherId/Pitcher for --pitcher or --team (same roster as Stuff+)."""
    if pitcher and team:
        raise ValueError("Pass --pitcher or --team, not both")
    if team:
        return team_roster_ids(team)
    if pitcher:
        pitcher_id = lookup_mlbam_id(pitcher)
        named = attach_pitcher_names(pd.DataFrame({"pitcher": [pitcher_id]}))
        return pd.DataFrame({"PitcherId": [pitcher_id], "Pitcher": [named["Pitcher"].iloc[0] or pitcher]})
    raise ValueError("Pass --pitcher or --team")


def run(pitcher: str | None = None, team: str | None = None, start_season: int | None = None, end_season: int | None = None, force_refresh: bool = False):
    pitchers = resolve_pitchers(pitcher, team)
    print(f"Pulling IL transactions for {len(pitchers)} pitcher(s)...")
    stints, summary = build_injury_history(pitchers, start_season, end_season, force_refresh=force_refresh)

    print(f"\n── IL stints ({pitcher or team}) ──")
    print(stints.drop(columns=["SourceDescription"]).to_string(index=False))
    print(f"\n── IL summary ({pitcher or team}) ──")
    print(summary.sort_values("TotalDaysMissed", ascending=False).to_string(index=False))
    save_injury_history(stints, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull IL stint history from the MLB Stats API transactions feed.")
    parser.add_argument("--pitcher", default=None, help='One pitcher, e.g. --pitcher "Paul Skenes"')
    parser.add_argument("--team", default=None, help="Every pitcher on a team's Stuff+ roster, e.g. --team PIT")
    parser.add_argument("--start-season", type=int, default=None, help=f"Default: config.SEASON - {INJURY_LOOKBACK_SEASONS}")
    parser.add_argument("--end-season", type=int, default=None, help="Default: config.SEASON")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using cached transactions")
    args = parser.parse_args()
    run(pitcher=args.pitcher, team=args.team, start_season=args.start_season, end_season=args.end_season, force_refresh=args.refresh)
