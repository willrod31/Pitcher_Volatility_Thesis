"""
Pulls and caches public Statcast data via pybaseball, and applies the
cleaning/engineering steps every downstream script builds on top of.

This is the slow, network-dependent step -- every analysis script (Stuff+
Proxy and the rest) works off the parquet files this writes, so re-running
the pipeline for tweaks doesn't re-hit Statcast every time.
"""
import functools
import shutil

import numpy as np
import pandas as pd

import pybaseball as pb

from config import END_DATE, MIN_PITCHES_FOR_INCLUSION, START_DATE
from analysis.utils import CACHE_DIR

# Non pitch types that show up in the raw feed and aren't real pitches.
NON_PITCH_TYPES = {"PO", "IN", "EP", "FA", "UN", "SC"}


def fetch_statcast(start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Pull league-wide pitch-level Statcast data for a date range, cached to disk."""
    raw_path = CACHE_DIR / f"statcast_raw_{start_date}_{end_date}.parquet"
    if raw_path.exists() and not force_refresh:
        return pd.read_parquet(raw_path)

    pb.cache.enable()
    df = pb.statcast(start_dt=start_date, end_dt=end_date)
    df.to_parquet(raw_path, index=False)
    return df


def fetch_statcast_pitcher(pitcher_name: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Pull one pitcher's Statcast data directly, cached to disk.

    Hits Baseball Savant's per-player endpoint instead of the league-wide
    one, so a full season for one pitcher is a few thousand rows instead of
    a few million -- cheap enough to always pull in full, and the only
    practical way to score one pitcher's whole season in a memory-limited
    environment where a league-wide pull for the same range would OOM.
    """
    slug = pitcher_name.strip().lower().replace(" ", "_")
    raw_path = CACHE_DIR / f"statcast_pitcher_{slug}_{start_date}_{end_date}.parquet"
    if raw_path.exists() and not force_refresh:
        return pd.read_parquet(raw_path)

    mlbam_id = lookup_mlbam_id(pitcher_name)

    pb.cache.enable()
    df = pb.statcast_pitcher(start_date, end_date, mlbam_id)
    df.to_parquet(raw_path, index=False)
    return df


def lookup_mlbam_id(pitcher_name: str) -> int:
    """Resolve a "First Last" name to an MLBAM id (most recently active match wins)."""
    first, last = pitcher_name.strip().split(" ", 1)
    lookup = pb.playerid_lookup(last, first)
    if lookup.empty:
        raise ValueError(f"No player found for name '{pitcher_name}'")
    return int(lookup.sort_values("mlb_played_last", ascending=False)["key_mlbam"].iloc[0])


def load_pitcher_season(pitcher_name: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Same cleaning pipeline as load_season(), scoped to one pitcher.

    Names/teams are attached straight off this pitcher's own rows -- no
    league-wide reverse lookup needed when we already know who they are.
    """
    raw = fetch_statcast_pitcher(pitcher_name, start_date, end_date, force_refresh=force_refresh)
    if raw.empty:
        raise ValueError(f"No Statcast pitches found for '{pitcher_name}' between {start_date} and {end_date}")
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def fetch_statcast_team(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Pull one team's own-pitching Statcast data directly, cached to disk.

    pybaseball's `team` filter on statcast() already returns only the pitches
    where that team is pitching (confirmed empirically, not just games they
    played in), so a full season for one team's staff is ~20-25k rows --
    small enough to never hit the league-wide OOM ceiling, and gets every
    pitcher on the staff in one request instead of one per pitcher.
    """
    raw_path = CACHE_DIR / f"statcast_team_{team}_{start_date}_{end_date}.parquet"
    if raw_path.exists() and not force_refresh:
        return pd.read_parquet(raw_path)

    pb.cache.enable()
    df = pb.statcast(start_dt=start_date, end_dt=end_date, team=team)
    df.to_parquet(raw_path, index=False)
    return df


def load_team_season(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Same cleaning pipeline as load_season(), scoped to one team's staff.

    Only covers pitches actually thrown while pitching for `team` -- a
    pitcher traded mid-range shows up with a partial season. See
    load_team_roster_full_seasons() for every pitcher's complete season
    across whichever teams they were actually on.
    """
    raw = fetch_statcast_team(team, start_date, end_date, force_refresh=force_refresh)
    if raw.empty:
        raise ValueError(f"No Statcast pitches found for team '{team}' between {start_date} and {end_date}")
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def seed_team_caches_from_league(teams, start_date: str = START_DATE, end_date: str = END_DATE) -> list[str]:
    """Write fetch_statcast_team()'s cache for each of `teams` that has none, from the league monthly caches.

    A team cache is the team's own-pitching rows, which is exactly the
    league feed's rows where that team is fielding (confirmed identical for
    PIT: same 22,590 rows, same 31 pitchers). Splitting the cached league
    months avoids one pb.statcast(team=...) pull per team. Months are
    processed one at a time and split into per-team part files, so memory
    stays at one raw month. Returns the teams it wrote.
    """
    todo = [t for t in teams if not (CACHE_DIR / f"statcast_team_{t}_{start_date}_{end_date}.parquet").exists()]
    if not todo:
        return []
    parts_dir = CACHE_DIR / "_team_parts"
    parts_dir.mkdir(exist_ok=True)
    try:
        for n, (chunk_start, chunk_end) in enumerate(_month_chunks(start_date, end_date)):
            raw = fetch_statcast(chunk_start, chunk_end)
            fielding = np.where(raw["inning_topbot"] == "Top", raw["home_team"], raw["away_team"])
            for team in todo:
                raw[fielding == team].to_parquet(parts_dir / f"{team}_{n}.parquet", index=False)
            del raw
        for team in todo:
            parts = sorted(parts_dir.glob(f"{team}_*.parquet"))
            df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
            df.to_parquet(CACHE_DIR / f"statcast_team_{team}_{start_date}_{end_date}.parquet", index=False)
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
    return todo


def team_roster_ids(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """PitcherId + Pitcher for everyone who threw for `team` at any point in this range.

    Memoized per process (run_all.py asks for the same roster several times per team).
    """
    if force_refresh:
        _team_roster_ids.cache_clear()
    return _team_roster_ids(team, start_date, end_date, force_refresh).copy()


@functools.lru_cache(maxsize=None)
def _team_roster_ids(team: str, start_date: str, end_date: str, force_refresh: bool) -> pd.DataFrame:
    cached = CACHE_DIR / f"statcast_team_{team}_{start_date}_{end_date}.parquet"
    if cached.exists() and not force_refresh:
        raw = pd.read_parquet(cached, columns=["pitcher"])   # only the ids are needed
    else:
        raw = fetch_statcast_team(team, start_date, end_date, force_refresh=force_refresh)
    named = attach_pitcher_names(raw[["pitcher"]].drop_duplicates())
    named = named.dropna(subset=["Pitcher"]).rename(columns={"pitcher": "PitcherId"})
    named["PitcherId"] = named["PitcherId"].astype(int)
    return named.sort_values("Pitcher").reset_index(drop=True)


def team_roster(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> list[str]:
    """Names of every pitcher who threw for `team` at any point in this range."""
    return team_roster_ids(team, start_date, end_date, force_refresh=force_refresh)["Pitcher"].tolist()


def load_team_roster_full_seasons(team: str, start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Every pitcher who threw for `team`, each with their FULL season's pitches.

    Unlike load_team_season(), a pitcher traded to/from `team` mid-range is
    included with everything they threw all season, not just their time on
    this roster -- one load_pitcher_season() pull per name on the roster.
    PitcherTeam is overridden to `team` for every row, since the point is
    "how this current roster grades out using each player's whole season,"
    not per-pitch team attribution.
    """
    roster = team_roster(team, start_date, end_date, force_refresh=force_refresh)
    frames = []
    for name in roster:
        player_df = load_pitcher_season(name, start_date, end_date, force_refresh=force_refresh)
        player_df = player_df.copy()
        player_df["PitcherTeam"] = team
        frames.append(player_df)
    return pd.concat(frames, ignore_index=True)


def attach_pitcher_names(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve pitcher MLBAM IDs to real names via pybaseball's ID lookup.

    Uses the numeric `pitcher` id rather than the `player_name` column,
    since which player that column names depends on how the query was run --
    the id is unambiguous.
    """
    df = df.copy()
    ids = df["pitcher"].dropna().unique().astype(int).tolist()
    lookup = pb.playerid_reverse_lookup(ids, key_type="mlbam")
    lookup = lookup.assign(
        Pitcher=lookup["name_first"].str.title() + " " + lookup["name_last"].str.title()
    )
    name_map = dict(zip(lookup["key_mlbam"], lookup["Pitcher"]))
    df["Pitcher"] = df["pitcher"].map(name_map)
    return df


def attach_pitcher_team(df: pd.DataFrame) -> pd.DataFrame:
    """Whichever team is fielding for a given pitch is the pitcher's team.

    Computed per pitch (not looked up once per player) so a pitcher traded
    mid-season is credited correctly on both sides of the trade.
    """
    df = df.copy()
    df["PitcherTeam"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    return df


def clean_pitch_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to realistic pitch values and engineer break columns.

    pfx_x/pfx_z come back from Statcast in feet, so *12 converts to the
    inches convention (ivb/hb) the rest of the project uses.
    """
    df = df.copy()
    df["ivb"] = df["pfx_z"] * 12
    df["hb"] = df["pfx_x"] * 12

    df = df[df["release_speed"].between(60, 105)]
    df = df[df["release_spin_rate"].between(1000, 3600)]
    df = df[df["release_extension"].between(3, 9)]
    df = df[df["ivb"].between(-25, 30)]
    df = df[df["hb"].between(-25, 25)]
    df = df[df["pitch_type"].notna()]
    df = df[~df["pitch_type"].isin(NON_PITCH_TYPES)]
    return df


def load_season(start_date: str = START_DATE, end_date: str = END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Convenience wrapper: fetch, name, team-attach, and clean a full season.

    Does NOT filter by MIN_PITCHES_FOR_INCLUSION -- that's a per-pitcher
    threshold, applied by callers after they've decided the unit of
    aggregation (e.g. total pitches vs. pitches of a given type).
    """
    raw = fetch_statcast(start_date, end_date, force_refresh=force_refresh)
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def _month_chunks(start_date: str, end_date: str):
    """Split a date range into calendar-month (start, end) string pairs."""
    chunk_start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.offsets.MonthEnd(0), end)
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        chunk_start = chunk_end + pd.Timedelta(days=1)


def load_season_monthly(start_date: str = START_DATE, end_date: str = END_DATE, columns: list[str] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Same cleaning pipeline as load_season(), fetched one calendar month at a time.

    A full 6-month league-wide statcast() call OOMs in a memory-limited
    environment during pybaseball's internal per-day concat (confirmed: peak
    ~7GB+ on an 8GB box, see README "Known limitations"). Fetching month by
    month keeps that internal concat bounded to ~120k rows (~3.3GB peak,
    confirmed safe) per call -- each month still hits fetch_statcast()'s
    normal per-range cache, so a month already pulled for another purpose is
    reused instantly.

    Pass `columns` (a subset of the ~119 raw Statcast columns) to drop
    everything else right after each month's fetch, before accumulating
    across months -- keeps the final combined frame small regardless of how
    many months it spans. Must include whatever attach_pitcher_names/
    attach_pitcher_team/clean_pitch_physics need: pitcher, home_team,
    away_team, inning_topbot, pfx_z, pfx_x, release_speed,
    release_spin_rate, release_extension, pitch_type.
    """
    chunks = []
    for chunk_start, chunk_end in _month_chunks(start_date, end_date):
        raw = fetch_statcast(chunk_start, chunk_end, force_refresh=force_refresh)
        if columns is not None:
            raw = raw[columns]
        chunks.append(raw)

    raw = pd.concat(chunks, ignore_index=True)
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_pitch_physics(df)
    return df


def filter_qualified(df: pd.DataFrame, min_pitches: int = MIN_PITCHES_FOR_INCLUSION) -> pd.DataFrame:
    """Drop pitchers below the season pitch-count threshold in config.py.

    Counted by MLBAM id, not name, so two pitchers who share a name aren't merged.
    """
    counts = df.groupby("pitcher").size()
    qualified = counts[counts >= min_pitches].index
    return df[df["pitcher"].isin(qualified)]


def primary_team(df: pd.DataFrame) -> pd.Series:
    """PitcherTeam each pitcher threw the most pitches for, indexed by MLBAM id.

    League-wide grading is per pitcher, not per pitcher-team, so a traded
    pitcher gets one row -- labeled with the team he threw most for.
    """
    counts = df.groupby(["pitcher", "PitcherTeam"]).size().rename("n").reset_index()
    top = counts.sort_values("n", ascending=False).drop_duplicates("pitcher")
    return top.set_index("pitcher")["PitcherTeam"]


def filter_to_team(graded: pd.DataFrame, team: str) -> pd.DataFrame:
    """Keep only pitchers on `team`'s roster (same roster as stuff_plus_proxy.py --team).

    Includes pitchers traded to/from the team, graded on their full season,
    and relabels PitcherTeam to `team` -- the same convention as the Stuff+
    dashboard export.
    """
    roster_ids = set(team_roster_ids(team)["PitcherId"])
    return graded[graded["PitcherId"].isin(roster_ids)].assign(PitcherTeam=team)
