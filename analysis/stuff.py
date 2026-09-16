"""
Stuff+ pitch-quality model built on public Statcast data via pybaseball.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import pybaseball as pb

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

SUMMARY_PATH = CACHE_DIR / "stuff_plus_summary.parquet"
PITCHES_PATH = CACHE_DIR / "stuff_plus_pitches.parquet"

# Model features -- same shape features the original Trackman-based script
# used (RelSpeed, InducedVertBreak, HorzBreak, SpinRate, SpinAxis, Extension,
# RelHeight, RelSide, EffectiveVelo), renamed to Statcast's column names.
# ivb/hb are engineered below from Statcast's pfx_z/pfx_x (feet -> inches).
FEATURES = [
    "release_speed",
    "ivb",
    "hb",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_z",
    "release_pos_x",
    "effective_speed",
]

WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {"foul", "foul_tip", "hit_into_play"}

# Non pitch types that show up in the raw feed and aren't real pitches.
NON_PITCH_TYPES = {"PO", "IN", "EP", "FA", "UN", "SC"}

MIN_PITCHES_PER_TYPE = 50      # min swings needed to train a pitch-type model
MIN_SAMPLE_FOR_SUMMARY = 25    # min pitches to keep a Pitcher/PitchType row


def fetch_statcast(start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
    """Pull league-wide pitch-level Statcast data for a date range, cached to disk.

    This is the slow, network-dependent step -- everything downstream works
    off the parquet file this writes, so re-running the pipeline for
    tweaks (feature list, thresholds) doesn't re-hit Statcast every time.
    """
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
    mid-range is credited correctly on both sides of the trade.
    """
    df = df.copy()
    df["PitcherTeam"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to realistic pitch values and engineer break/label columns.

    Same bounds as the original Trackman-based version, just applied to the
    Statcast equivalents (pfx_x/pfx_z come back in feet, so *12 for inches).
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

    df["Whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    df["SwingOrNot"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    return df


def build_stuff_plus(raw: pd.DataFrame):
    """Train one whiff-probability model per pitch type and score every swing.

    Returns (per_pitch, summary):
      per_pitch -- one row per scored swing, with its StuffPlus value
      summary   -- one row per Pitcher x PitcherTeam x PitchType, averaged
    """
    df = attach_pitcher_names(raw)
    df = attach_pitcher_team(df)
    df = clean_and_engineer(df)

    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type", "Pitcher"]).copy()
    print(f"{len(swings)} swings with complete data")

    results = []
    for pitch_type, group in swings.groupby("pitch_type"):
        if len(group) < MIN_PITCHES_PER_TYPE:
            continue

        X = group[FEATURES]
        y = group["Whiff"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_scaled, y)

        group = group.copy()
        group["PredWhiffProb"] = model.predict_proba(X_scaled)[:, 1]
        mean_prob = group["PredWhiffProb"].mean()
        group["StuffPlus"] = (group["PredWhiffProb"] / mean_prob) * 100
        group["PitchType"] = pitch_type

        results.append(group[[
            "Pitcher", "PitcherTeam", "PitchType",
            "release_speed", "ivb", "hb", "release_spin_rate", "StuffPlus",
        ]])
        print(f"{pitch_type}: avg Stuff+ = {group['StuffPlus'].mean():.1f}")

    if not results:
        raise ValueError(
            "No pitch type had enough swings to train a model. "
            "Try a wider date range or lower MIN_PITCHES_PER_TYPE."
        )

    per_pitch = pd.concat(results, ignore_index=True)

    summary = (
        per_pitch.groupby(["Pitcher", "PitcherTeam", "PitchType"])
        .agg(StuffPlus=("StuffPlus", "mean"), Pitches=("StuffPlus", "count"))
        .round({"StuffPlus": 1})
        .reset_index()
        .sort_values("StuffPlus", ascending=False)
    )
    summary = summary[summary["Pitches"] >= MIN_SAMPLE_FOR_SUMMARY]

    return per_pitch, summary


def run(start_date: str, end_date: str, force_refresh: bool = False):
    print(f"Pulling Statcast data {start_date} to {end_date}...")
    raw = fetch_statcast(start_date, end_date, force_refresh=force_refresh)
    print(f"Loaded {len(raw)} raw pitches")

    per_pitch, summary = build_stuff_plus(raw)

    per_pitch.to_parquet(PITCHES_PATH, index=False)
    summary.to_parquet(SUMMARY_PATH, index=False)

    print("\n── Top 20 Pitches by Stuff+ ──")
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved summary ({len(summary)} rows) to {SUMMARY_PATH}")
    print(f"Saved per-pitch data ({len(per_pitch)} rows) to {PITCHES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Stuff+ dataset from Statcast via pybaseball.")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(args.start, args.end, force_refresh=args.refresh)