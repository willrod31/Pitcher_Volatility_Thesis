"""
Pulls and caches public Statcast data via pybaseball, and applies the
cleaning/engineering steps every downstream script builds on top of.

This is the slow, network-dependent step -- every analysis script (Stuff+
Proxy and the rest) works off the parquet files this writes, so re-running
the pipeline for tweaks doesn't re-hit Statcast every time.
"""
import numpy as np
import pandas as pd

import pybaseball as pb

from config import END_DATE, MIN_PITCHES_FOR_INCLUSION, START_DATE
from src.utils import CACHE_DIR

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


def filter_qualified(df: pd.DataFrame, min_pitches: int = MIN_PITCHES_FOR_INCLUSION) -> pd.DataFrame:
    """Drop pitchers below the season pitch-count threshold in config.py."""
    counts = df.groupby("Pitcher").size()
    qualified = counts[counts >= min_pitches].index
    return df[df["Pitcher"].isin(qualified)]
