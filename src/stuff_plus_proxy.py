"""
Stuff+ pitch-quality model built on public Statcast data via pybaseball.

Stuff+ and Location+ are proprietary (FanGraphs/PitcherList), so this trains
its own proxy: one whiff-probability model per pitch type on raw pitch
physics (velocity, movement, spin, extension), scaled to a 100-average scale
the same way the proprietary metrics are.
"""
import argparse

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.data_acquisition import CACHE_DIR, END_DATE, START_DATE, load_season

SUMMARY_PATH = CACHE_DIR / "stuff_plus_summary.parquet"
PITCHES_PATH = CACHE_DIR / "stuff_plus_pitches.parquet"

# Model features -- same shape features the original Trackman-based script
# used (RelSpeed, InducedVertBreak, HorzBreak, SpinRate, SpinAxis, Extension,
# RelHeight, RelSide, EffectiveVelo), renamed to Statcast's column names.
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

MIN_PITCHES_PER_TYPE = 50      # min swings needed to train a pitch-type model
MIN_SAMPLE_FOR_SUMMARY = 25    # min pitches to keep a Pitcher/PitchType row


def label_swings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    df["SwingOrNot"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    return df


def build_stuff_plus(df: pd.DataFrame):
    """Train one whiff-probability model per pitch type and score every swing.

    Returns (per_pitch, summary):
      per_pitch -- one row per scored swing, with its StuffPlus value
      summary   -- one row per Pitcher x PitcherTeam x PitchType, averaged
    """
    df = label_swings(df)
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


def pitcher_level_stuff_plus(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse the PitchType-level summary to one pitch-count-weighted Stuff+ per pitcher."""
    weighted = summary.assign(Weighted=summary["StuffPlus"] * summary["Pitches"])
    out = (
        weighted.groupby(["Pitcher", "PitcherTeam"])
        .agg(WeightedSum=("Weighted", "sum"), Pitches=("Pitches", "sum"))
        .reset_index()
    )
    out["StuffPlus"] = (out["WeightedSum"] / out["Pitches"]).round(1)
    return out.drop(columns="WeightedSum")


def run(start_date: str, end_date: str, force_refresh: bool = False):
    print(f"Pulling Statcast data {start_date} to {end_date}...")
    df = load_season(start_date, end_date, force_refresh=force_refresh)
    print(f"Loaded {len(df)} cleaned pitches")

    per_pitch, summary = build_stuff_plus(df)

    per_pitch.to_parquet(PITCHES_PATH, index=False)
    summary.to_parquet(SUMMARY_PATH, index=False)

    print("\n── Top 20 Pitches by Stuff+ ──")
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved summary ({len(summary)} rows) to {SUMMARY_PATH}")
    print(f"Saved per-pitch data ({len(per_pitch)} rows) to {PITCHES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Stuff+ dataset from Statcast via pybaseball.")
    parser.add_argument("--start", default=START_DATE, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="End date, YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(args.start, args.end, force_refresh=args.refresh)
