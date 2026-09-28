"""
Stuff+ pitch-quality model built on public Statcast data via pybaseball.

Stuff+ and Location+ are proprietary (FanGraphs/PitcherList), so this trains
its own proxy: one whiff-probability model per pitch type on raw pitch
physics (velocity, movement, spin, extension), scaled to a 100-average scale
the same way the proprietary metrics are.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from analysis.data_acquisition import CACHE_DIR, END_DATE, START_DATE, load_pitcher_season, load_season, load_season_monthly, load_team_roster_full_seasons, primary_team
from analysis.utils import scale_100
from config import MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, SHOULDER_HEIGHT_FT

SUMMARY_PATH = CACHE_DIR / "stuff_plus_summary.parquet"
PITCHES_PATH = CACHE_DIR / "stuff_plus_pitches.parquet"

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "stuff_plus_pitch_types.csv"
DASHBOARD_PITCHES_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "stuff_plus_pitches.csv"
# ALL pitches (not just scored swings) for the pitch movement chart
MOVEMENT_PITCHES_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "movement_pitches.csv"
MOVEMENT_COLUMNS = [
    "PitcherId", "Pitcher", "PitcherTeam", "PitchType", "release_speed", "ivb", "hb",
    "p_throws", "arm_angle", "ArmAngleSource", "release_pos_x", "release_pos_z", "release_extension",
]

# League-wide sample the model is trained on when scoring a single pitcher or
# team: the whole season (config.START_DATE-END_DATE), pulled one calendar
# month at a time via load_season_monthly() -- a single league-wide
# statcast() call for 6 months OOMs in a memory-limited environment during
# pybaseball's internal concat (confirmed: peak ~7GB+ on an 8GB box); one
# month at a time stays bounded (~120k rows, ~3.3GB peak, confirmed safe),
# and RAW_TRAIN_COLUMNS drops the ~100 unused Statcast columns per month
# before accumulating, so the combined 6-month frame stays small too.
TRAIN_START = START_DATE
TRAIN_END = END_DATE
RAW_TRAIN_COLUMNS = [
    "pitcher", "home_team", "away_team", "inning_topbot",
    "release_speed", "release_spin_rate", "release_extension",
    "spin_axis", "release_pos_z", "release_pos_x", "effective_speed",
    "pfx_z", "pfx_x", "pitch_type", "description",
    # not model features -- carried along for the dashboard movement chart only
    "p_throws", "arm_angle",
]

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


def fit_stuff_plus_models(df: pd.DataFrame) -> dict:
    """Train one whiff-probability model per pitch type on a pool of swings.

    Returns {pitch_type: (model, scaler, baseline_whiff_prob)}.
    baseline_whiff_prob is this pool's average predicted whiff probability
    for that pitch type -- the value a StuffPlus of exactly 100 means.
    """
    df = label_swings(df)
    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type"]).copy()
    print(f"{len(swings)} training swings with complete data")

    models = {}
    for pitch_type, group in swings.groupby("pitch_type"):
        if len(group) < MIN_PITCHES_PER_TYPE:
            continue

        X = group[FEATURES]
        y = group["Whiff"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_scaled, y)

        baseline = model.predict_proba(X_scaled)[:, 1].mean()
        models[pitch_type] = (model, scaler, baseline)
        print(f"{pitch_type}: trained on {len(group)} swings, baseline whiff prob = {baseline:.1%}")

    if not models:
        raise ValueError(
            "No pitch type had enough swings to train a model. "
            "Try a wider date range or lower MIN_PITCHES_PER_TYPE."
        )
    return models


def score_stuff_plus(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Score every swing in df against pre-fit per-pitch-type models.

    Returns one row per scored swing, with its StuffPlus value (100 =
    the training pool's average whiff probability for that pitch type).
    """
    df = label_swings(df)
    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type", "Pitcher"]).copy()

    results = []
    for pitch_type, group in swings.groupby("pitch_type"):
        if pitch_type not in models:
            continue
        model, scaler, baseline = models[pitch_type]

        group = group.copy()
        X_scaled = scaler.transform(group[FEATURES])
        group["PredWhiffProb"] = model.predict_proba(X_scaled)[:, 1]
        group["StuffPlus"] = (group["PredWhiffProb"] / baseline) * 100
        group["PitchType"] = pitch_type

        results.append(group[[
            "Pitcher", "pitcher", "PitcherTeam", "PitchType",
            "release_speed", "ivb", "hb", "release_spin_rate", "StuffPlus",
        ]].rename(columns={"pitcher": "PitcherId"}))

    if not results:
        raise ValueError("None of this data's pitch types had a trained model to score against.")

    return pd.concat(results, ignore_index=True)


def summarize_stuff_plus(per_pitch: pd.DataFrame) -> pd.DataFrame:
    """Collapse scored swings to one row per Pitcher x PitcherTeam x PitchType."""
    summary = (
        per_pitch.groupby(["Pitcher", "PitcherId", "PitcherTeam", "PitchType"])
        .agg(StuffPlus=("StuffPlus", "mean"), Pitches=("StuffPlus", "count"))
        .round({"StuffPlus": 1})
        .reset_index()
        .sort_values("StuffPlus", ascending=False)
    )
    return summary[summary["Pitches"] >= MIN_SAMPLE_FOR_SUMMARY]


def build_stuff_plus(df: pd.DataFrame):
    """Fit and score the same pool of data (the original, single-pool behavior).

    Returns (per_pitch, summary). For scoring a single pitcher against a
    separately-trained league baseline, use fit_stuff_plus_models() +
    score_stuff_plus() directly instead -- see run()'s --pitcher path.
    """
    models = fit_stuff_plus_models(df)
    per_pitch = score_stuff_plus(df, models)
    summary = summarize_stuff_plus(per_pitch)
    return per_pitch, summary


def pitcher_level_stuff_plus(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse the PitchType-level summary to one pitch-count-weighted Stuff+ per pitcher."""
    weighted = summary.assign(Weighted=summary["StuffPlus"] * summary["Pitches"])
    out = (
        weighted.groupby(["Pitcher", "PitcherId", "PitcherTeam"])
        .agg(WeightedSum=("Weighted", "sum"), Pitches=("Pitches", "sum"))
        .reset_index()
    )
    out["StuffPlus"] = (out["WeightedSum"] / out["Pitches"]).round(1)
    return out.drop(columns="WeightedSum")


def league_scale_pool(league_df: pd.DataFrame, league_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The league-wide qualified pool StuffPlus_Scaled is graded against.

    Returns (pitcher pool, pitcher x pitch type pool) of ratio StuffPlus values:
    pitchers with MIN_PITCHES_FOR_INCLUSION pitches thrown, and pitcher x pitch
    type rows with MIN_PITCHES_PER_TYPE_FOR_SCALE pitches of that type thrown.
    One row per PitcherId (traded pitchers aren't split by team).
    """
    thrown = league_df.groupby("pitcher").size()
    thrown_by_type = league_df.groupby(["pitcher", "pitch_type"]).size()

    by_type = league_summary.groupby(["PitcherId", "PitchType"]).apply(
        lambda g: (g["StuffPlus"] * g["Pitches"]).sum() / g["Pitches"].sum(), include_groups=False,
    ).rename("StuffPlus").reset_index()
    by_type_thrown = thrown_by_type.reindex(pd.MultiIndex.from_frame(by_type[["PitcherId", "PitchType"]])).to_numpy()
    type_pool = by_type[by_type_thrown >= MIN_PITCHES_PER_TYPE_FOR_SCALE]

    weighted = league_summary.assign(Weighted=league_summary["StuffPlus"] * league_summary["Pitches"])
    per_pitcher = weighted.groupby("PitcherId")[["Weighted", "Pitches"]].sum()
    per_pitcher["StuffPlus"] = per_pitcher["Weighted"] / per_pitcher["Pitches"]
    per_pitcher = per_pitcher.reset_index()
    pitcher_pool = per_pitcher[thrown.reindex(per_pitcher["PitcherId"]).to_numpy() >= MIN_PITCHES_FOR_INCLUSION]
    return pitcher_pool, type_pool


def add_scaled_stuff_plus(summary: pd.DataFrame, pitcher_level: pd.DataFrame, pitcher_pool: pd.DataFrame, type_pool: pd.DataFrame):
    """ADD StuffPlus_Scaled = 100 + 10 * z(StuffPlus), z vs. the league-wide qualified pool.

    The ratio StuffPlus column is left exactly as it is. Pitch-type rows are
    z-scored within their own pitch type. Same scale as Location+.
    """
    summary = summary.copy()
    summary["StuffPlus_Scaled"] = float("nan")
    for pitch_type, rows in summary.groupby("PitchType"):
        pool = type_pool.loc[type_pool["PitchType"] == pitch_type, "StuffPlus"]
        if len(pool) >= 2:
            summary.loc[rows.index, "StuffPlus_Scaled"] = scale_100(rows["StuffPlus"], pool).round(1)
    pitcher_level = pitcher_level.copy()
    pitcher_level["StuffPlus_Scaled"] = scale_100(pitcher_level["StuffPlus"], pitcher_pool["StuffPlus"]).round(1)
    return summary, pitcher_level


def save_for_dashboard(summary: pd.DataFrame, pitcher_level: pd.DataFrame):
    """Upsert this run's per-pitch-type Stuff+ rows into the dashboard's CSV.

    Stuff+ here is still normalized against a separately-fit league-wide
    model (see run()'s --pitcher/--team path), so results stay meaningful
    even though only a subset of pitchers is written out at a time -- 100 is
    a league average, not this subset's own.

    Upserted by whichever Pitchers appear in `summary`, so re-running for a
    pitcher or team replaces just their old rows, and everyone else already
    in the file is left alone.
    """
    rows = summary.merge(
        pitcher_level[["Pitcher", "StuffPlus", "StuffPlus_Scaled"]].rename(
            columns={"StuffPlus": "OverallStuffPlus", "StuffPlus_Scaled": "OverallStuffPlus_Scaled"}), on="Pitcher"
    )
    rows = rows[["Pitcher", "PitcherId", "PitcherTeam", "PitchType", "Pitches", "StuffPlus", "StuffPlus_Scaled",
                 "OverallStuffPlus", "OverallStuffPlus_Scaled"]]
    pitchers_in_run = rows["Pitcher"].unique()

    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DASHBOARD_PATH.exists():
        existing = pd.read_csv(DASHBOARD_PATH)
        existing = existing[~existing["Pitcher"].isin(pitchers_in_run)]
        rows = pd.concat([existing, rows], ignore_index=True)

    rows.to_csv(DASHBOARD_PATH, index=False)
    print(f"Saved {len(rows)} row(s) to {DASHBOARD_PATH} for the dashboard ({len(pitchers_in_run)} pitcher(s) this run)")


def save_pitches_for_dashboard(per_pitch: pd.DataFrame):
    """Upsert this run's scored swings (for the movement scatter chart)."""
    rows = per_pitch[["Pitcher", "PitcherId", "PitcherTeam", "PitchType", "release_speed", "ivb", "hb", "StuffPlus"]]
    pitchers_in_run = rows["Pitcher"].unique()

    DASHBOARD_PITCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DASHBOARD_PITCHES_PATH.exists():
        existing = pd.read_csv(DASHBOARD_PITCHES_PATH)
        existing = existing[~existing["Pitcher"].isin(pitchers_in_run)]
        rows = pd.concat([existing, rows], ignore_index=True)

    rows.to_csv(DASHBOARD_PITCHES_PATH, index=False)
    print(f"Saved {len(rows)} pitch row(s) to {DASHBOARD_PITCHES_PATH} for the dashboard")


def estimate_arm_angle(df: pd.DataFrame) -> pd.Series:
    """APPROXIMATE arm angle (degrees above horizontal) from release point.

    degrees(atan2(release height - SHOULDER_HEIGHT_FT, |release side|)). Only
    used when Statcast has no arm_angle for a pitcher at all.
    """
    return np.degrees(np.arctan2(df["release_pos_z"] - SHOULDER_HEIGHT_FT, df["release_pos_x"].abs()))


def save_movement_for_dashboard(scope_df: pd.DataFrame):
    """Upsert every pitch (not just swings) for this run's pitchers, for the movement chart.

    arm_angle is Statcast's when the pitcher has any (ArmAngleSource =
    "Statcast"; the odd null pitch stays null); a pitcher with none at all
    gets the release-point estimate on every pitch ("release-point estimate").
    """
    rows = scope_df.rename(columns={"pitcher": "PitcherId", "pitch_type": "PitchType"}).copy()
    if "arm_angle" not in rows:
        rows["arm_angle"] = np.nan
    rows["arm_angle"] = rows["arm_angle"].astype(float)
    has_statcast = rows.groupby("PitcherId")["arm_angle"].transform(lambda s: s.notna().any())
    rows["ArmAngleSource"] = np.where(has_statcast, "Statcast", "release-point estimate")
    rows.loc[~has_statcast, "arm_angle"] = estimate_arm_angle(rows.loc[~has_statcast])
    rows = rows[MOVEMENT_COLUMNS]

    MOVEMENT_PITCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MOVEMENT_PITCHES_PATH.exists():
        existing = pd.read_csv(MOVEMENT_PITCHES_PATH)
        existing = existing[~existing["PitcherId"].isin(rows["PitcherId"].unique())]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(MOVEMENT_PITCHES_PATH, index=False)
    print(f"Saved {len(rows)} pitch row(s) to {MOVEMENT_PITCHES_PATH} for the movement chart")


def run(start_date: str, end_date: str, pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    if pitcher and team:
        raise ValueError("Pass --pitcher or --team, not both")

    if pitcher or team:
        # Fit on the full season, pulled one month at a time (see module
        # docstring), then score this pitcher's or team's own full-range
        # data against it -- pulling their full range league-wide instead
        # would OOM the same way a single-call full-season pull does.
        print(f"Pulling a league-wide training sample {TRAIN_START} to {TRAIN_END} (month by month)...")
        train_df = load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS, force_refresh=force_refresh)
        print(f"Loaded {len(train_df)} cleaned training pitches")
        models = fit_stuff_plus_models(train_df)

        if pitcher:
            print(f"\nPulling {pitcher}'s Statcast data {start_date} to {end_date}...")
            scope_df = load_pitcher_season(pitcher, start_date, end_date, force_refresh=force_refresh)
            print(f"Loaded {len(scope_df)} cleaned pitches for {pitcher}")
        else:
            print(f"\nFinding {team}'s pitching roster and each pitcher's full season "
                  f"(including time with other teams if traded)...")
            scope_df = load_team_roster_full_seasons(team, start_date, end_date, force_refresh=force_refresh)
            print(f"Loaded {len(scope_df)} cleaned pitches across {team}'s roster's full seasons")

        per_pitch = score_stuff_plus(scope_df, models)
        summary = summarize_stuff_plus(per_pitch)

        # League-wide pool for StuffPlus_Scaled: score the whole training season
        # with the same models (one row per pitcher, not per pitcher-team).
        print("\nScoring the league-wide pool for StuffPlus_Scaled...")
        league_scoring = train_df.assign(PitcherTeam=train_df["pitcher"].map(primary_team(train_df)))
        league_summary = summarize_stuff_plus(score_stuff_plus(league_scoring, models))
        pitcher_pool, type_pool = league_scale_pool(train_df, league_summary)
    else:
        print(f"Pulling league-wide Statcast data {start_date} to {end_date}...")
        df = load_season(start_date, end_date, force_refresh=force_refresh)
        print(f"Loaded {len(df)} cleaned pitches")
        per_pitch, summary = build_stuff_plus(df)
        pitcher_pool, type_pool = league_scale_pool(df, summary)

    summary, _ = add_scaled_stuff_plus(summary, pitcher_level_stuff_plus(summary), pitcher_pool, type_pool)
    per_pitch.to_parquet(PITCHES_PATH, index=False)
    summary.to_parquet(SUMMARY_PATH, index=False)

    print("\n── Top 20 Pitches by Stuff+ ──")
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved summary ({len(summary)} rows) to {SUMMARY_PATH}")
    print(f"Saved per-pitch data ({len(per_pitch)} rows) to {PITCHES_PATH}")

    if pitcher or team:
        pitcher_level = pitcher_level_stuff_plus(summary)
        _, pitcher_level = add_scaled_stuff_plus(summary, pitcher_level, pitcher_pool, type_pool)
        if pitcher_level.empty:
            scope = pitcher or team
            raise ValueError(f"'{scope}' not found in the Stuff+ summary (not enough qualifying pitches this range)")
        print(f"\n── {pitcher or team} pitching staff ──")
        print(pitcher_level.sort_values("StuffPlus", ascending=False).to_string(index=False))
        save_for_dashboard(summary, pitcher_level)
        save_pitches_for_dashboard(per_pitch)
        save_movement_for_dashboard(scope_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Stuff+ dataset from Statcast via pybaseball.")
    parser.add_argument("--start", default=START_DATE, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="End date, YYYY-MM-DD")
    parser.add_argument("--pitcher", default=None, help="Also print/export one pitcher's result for the dashboard, e.g. --pitcher \"Paul Skenes\"")
    parser.add_argument("--team", default=None, help="Also print/export a whole team's pitching staff for the dashboard, e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(args.start, args.end, pitcher=args.pitcher, team=args.team, force_refresh=args.refresh)
