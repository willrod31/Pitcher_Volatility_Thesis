"""
Stuff+ pitch-quality model built on public Statcast data via pybaseball.

Stuff+ and Location+ are proprietary (FanGraphs/PitcherList), so this trains
its own proxy: one whiff-probability model per pitch type on raw pitch
physics (velocity, movement, spin, extension), scaled to a 100-average scale
the same way the proprietary metrics are.

v2 (current):
- Handedness: hb, release_pos_x and spin_axis are in the catcher's view, so
  a LHP slider and a RHP slider have opposite signs. to_rhp_frame() mirrors
  left-handers into the right-hander frame before fitting AND scoring (the
  saved raw hb/release columns used by the movement chart are untouched).
- Trained on swings (the model is P(whiff | swing, pitch shape)), but every
  pitch of that type is scored, so a pitcher's grade uses his whole arsenal
  sample, not just the pitches that happened to be swung at.
- Out-of-fold: 5-fold GroupKFold by pitcher, so a pitcher's own pitches
  never train the model that grades him (same as location_plus_proxy.py).
  Out-of-fold AUC / log loss vs. a constant-whiff-rate baseline per pitch
  type go to data/stuff_plus_validation.csv.

v1 (the original: both hands in one frame, swings only, in-sample) is kept
as StuffPlus_v1 / StuffPlus_Scaled_v1 for comparison.

StuffPlus = predicted whiff probability / league average for the pitch type
x 100 (ratio). StuffPlus_Scaled = 100 + 10 z vs. the league-wide qualified
pool. StuffPlus_Scaled_Reg regresses small samples toward 100 using the
stabilization point k from analysis/stabilization.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analysis.data_acquisition import END_DATE, START_DATE, pitchers_only, load_pitcher_season, load_season_monthly, load_team_roster_full_seasons, primary_team
from analysis.stabilization import load_k, regress, reliability
from analysis.utils import STUFF_PLUS_DIR, scale_100
from config import (
    MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, MIN_PITCHES_PER_TYPE_TO_DISPLAY, MIN_PITCHES_TO_DISPLAY, SHOULDER_HEIGHT_FT,
)

# League-wide, graded vs. the qualified pool (asymmetric_upside.py and the dashboard read these)
LEAGUE_PITCHES_PATH = STUFF_PLUS_DIR / "stuff_plus_league_pitches.parquet"   # every pitch, out-of-fold score (cached, slow step)
LEAGUE_SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_league_summary.parquet"   # pitcher x pitch type
LEAGUE_PITCHER_PATH = STUFF_PLUS_DIR / "stuff_plus_league_pitcher.parquet"   # pitcher
V1_SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_v1_league_summary.parquet"    # original model's scores, for StuffPlus_v1
VALIDATION_PATH = STUFF_PLUS_DIR / "stuff_plus_validation.csv"
V1_COMPARISON_PATH = STUFF_PLUS_DIR / "stuff_plus_v1_comparison.csv"
# This run's --pitcher/--team scope
SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_summary.parquet"
PITCHES_PATH = STUFF_PLUS_DIR / "stuff_plus_pitches.parquet"

DASHBOARD_DATA = Path(__file__).resolve().parent.parent / "dashboard" / "data"
DASHBOARD_PATH = DASHBOARD_DATA / "stuff_plus_pitch_types.csv"
# The two per-pitch exports are gzipped: league-wide they're 30-60MB as plain CSV,
# over the 25MB GitHub web limit (pandas reads/writes .csv.gz transparently).
DASHBOARD_PITCHES_PATH = DASHBOARD_DATA / "stuff_plus_pitches.csv.gz"
# ALL pitches (not just scored swings) for the pitch movement chart -- only the columns app.py reads
MOVEMENT_PITCHES_PATH = DASHBOARD_DATA / "movement_pitches.csv.gz"
MOVEMENT_COLUMNS = [
    "PitcherId", "PitchType", "release_speed", "ivb", "hb", "p_throws", "arm_angle", "ArmAngleSource",
]
# Per-appearance primary fastball velocity, for the velocity-by-appearance chart
VELOCITY_PATH = DASHBOARD_DATA / "velocity_by_game.csv"
PRIMARY_FASTBALL_ORDER = ["FF", "SI", "FC"]
MIN_PITCHES_FOR_PRIMARY_FASTBALL = 20   # season total; fewer and the next fastball in the order is used

# League-wide training sample: the whole season (config.START_DATE-END_DATE),
# pulled one calendar month at a time via load_season_monthly() -- a single
# league-wide statcast() call for 6 months OOMs in a memory-limited
# environment during pybaseball's internal concat (confirmed: peak ~7GB+ on
# an 8GB box); one month at a time stays bounded (~120k rows, ~3.3GB peak),
# and RAW_TRAIN_COLUMNS drops the ~100 unused Statcast columns per month.
TRAIN_START = START_DATE
TRAIN_END = END_DATE
RAW_TRAIN_COLUMNS = [
    "pitcher", "home_team", "away_team", "inning_topbot",
    "release_speed", "release_spin_rate", "release_extension",
    "spin_axis", "release_pos_z", "release_pos_x", "effective_speed",
    "pfx_z", "pfx_x", "pitch_type", "description",
    "p_throws",   # handedness mirroring (to_rhp_frame)
    "arm_angle",  # not a model feature -- dashboard movement chart only
]
# Training columns plus what the per-team velocity chart needs, so one
# league-wide load can grade AND be sliced into each team's scope (run_all.py).
SCOPE_COLUMNS = RAW_TRAIN_COLUMNS + ["game_pk", "game_date"]

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

MIN_PITCHES_PER_TYPE = 50      # min league swings needed to train a pitch-type model
N_FOLDS = 5
MIN_SAMPLE_FOR_SUMMARY_V1 = 25  # v1 only: min scored swings to keep a Pitcher/PitchType row


def label_swings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    df["SwingOrNot"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    return df


def to_rhp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror left-handers into the right-hander frame (model features only).

    hb, release_pos_x and spin_axis are measured from the catcher's view, so
    the same pitch shape has the opposite sign for a LHP. Used by both fit
    and score; callers keep their own un-mirrored copy for anything saved.
    spin_axis: 360 - axis, taken mod 360 so an axis of 0 stays 0.
    """
    df = df.copy()
    lefty = df["p_throws"] == "L"
    df.loc[lefty, "hb"] = -df.loc[lefty, "hb"]
    df.loc[lefty, "release_pos_x"] = -df.loc[lefty, "release_pos_x"]
    df.loc[lefty, "spin_axis"] = (360 - df.loc[lefty, "spin_axis"]) % 360
    return df


def new_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)


# ── v2: out-of-fold, every pitch, handedness-mirrored ────────────────────────

def fit_and_score_oof(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold PredWhiffProb for every pitch, plus per-pitch-type validation.

    Per pitch type: folds are GroupKFold by pitcher over ALL pitches of that
    type; each fold's model is trained on the OTHER folds' swings only, then
    scores every pitch (swing or not) of the held-out pitchers. Validation
    (AUC, log loss) uses the held-out swings, against a baseline that always
    predicts the training folds' whiff rate. Pitch types with fewer than
    MIN_PITCHES_PER_TYPE league swings, or fewer than N_FOLDS pitchers, are
    not scored.
    """
    df = label_swings(df).dropna(subset=FEATURES + ["pitch_type", "p_throws"])
    X_all = to_rhp_frame(df)[FEATURES]
    pred = pd.Series(np.nan, index=df.index)
    validation = []

    for pitch_type, group in df.groupby("pitch_type"):
        swing = group["SwingOrNot"].to_numpy() == 1
        n_pitchers = group["pitcher"].nunique()
        if swing.sum() < MIN_PITCHES_PER_TYPE or n_pitchers < N_FOLDS:
            print(f"{pitch_type}: skipped ({swing.sum()} swings, {n_pitchers} pitchers)")
            continue
        X = X_all.loc[group.index].to_numpy()
        y = group["Whiff"].to_numpy()
        type_pred = np.full(len(group), np.nan)
        baseline = np.full(len(group), np.nan)
        for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(X, groups=group["pitcher"].to_numpy()):
            train_swings = train_idx[swing[train_idx]]
            model = new_model().fit(X[train_swings], y[train_swings])
            type_pred[test_idx] = model.predict_proba(X[test_idx])[:, 1]
            baseline[test_idx] = y[train_swings].mean()
        pred.loc[group.index] = type_pred

        y_sw, p_sw, b_sw = y[swing], type_pred[swing], baseline[swing]
        row = {
            "PitchType": pitch_type, "Pitchers": n_pitchers, "Pitches": len(group), "Swings": int(swing.sum()),
            "WhiffRate": y_sw.mean(), "AUC": roc_auc_score(y_sw, p_sw),
            "LogLoss": log_loss(y_sw, p_sw), "BaselineLogLoss": log_loss(y_sw, b_sw),
        }
        row["LogLossImprovementPct"] = 100 * (1 - row["LogLoss"] / row["BaselineLogLoss"])
        validation.append(row)
        print(f"{pitch_type}: {row['Swings']} swings, OOF AUC {row['AUC']:.3f}, "
              f"log loss {row['LogLoss']:.4f} vs baseline {row['BaselineLogLoss']:.4f}")

    df["PredWhiffProb"] = pred
    scored = df[df["PredWhiffProb"].notna()].copy()
    # 100 = the league's average predicted whiff probability for that pitch type, over every pitch
    scored["StuffPlus"] = 100 * scored["PredWhiffProb"] / scored.groupby("pitch_type")["PredWhiffProb"].transform("mean")
    return scored, pd.DataFrame(validation)


def build_league_scores(train_df: pd.DataFrame | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Every league pitch with its out-of-fold Stuff+, cached (the slow step). Writes the validation table."""
    if LEAGUE_PITCHES_PATH.exists() and not force_refresh:
        # pitchers_only: the cache may predate the position filter
        return pitchers_only(pd.read_parquet(LEAGUE_PITCHES_PATH), "PitcherId")

    if train_df is None:
        print(f"Pulling league-wide Statcast {TRAIN_START} to {TRAIN_END} (month by month)...")
        train_df = load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS)
    print("Fitting out-of-fold Stuff+ models (5-fold GroupKFold by pitcher, handedness-mirrored)...")
    scored, validation = fit_and_score_oof(train_df)
    scored["PitcherTeam"] = scored["pitcher"].map(primary_team(train_df))

    validation.to_csv(VALIDATION_PATH, index=False)
    print_validation(validation)

    out = scored.rename(columns={"pitcher": "PitcherId", "pitch_type": "PitchType", "SwingOrNot": "Swing"})[[
        "PitcherId", "Pitcher", "PitcherTeam", "PitchType", "p_throws", "Swing", "Whiff", "PredWhiffProb", "StuffPlus",
        "release_speed", "ivb", "hb", "release_spin_rate",
    ]]
    out.to_parquet(LEAGUE_PITCHES_PATH, index=False)
    return out


def print_validation(validation: pd.DataFrame):
    print("\n── Stuff+ out-of-fold validation (held-out pitchers' swings) ──")
    print(validation.round({"WhiffRate": 3, "AUC": 3, "LogLoss": 4, "BaselineLogLoss": 4, "LogLossImprovementPct": 1}).to_string(index=False))
    print("AUC > 0.5 and LogLoss < BaselineLogLoss (a constant whiff rate) = the shape features carry signal.")
    print(f"Saved to {VALIDATION_PATH}")


def summarize_stuff_plus(per_pitch: pd.DataFrame) -> pd.DataFrame:
    """Collapse scored pitches to one row per Pitcher x PitcherTeam x PitchType (no minimum -- see add_regressed)."""
    return (
        per_pitch.groupby(["Pitcher", "PitcherId", "PitcherTeam", "PitchType"])
        .agg(StuffPlus=("StuffPlus", "mean"), Pitches=("StuffPlus", "size"), Swings=("Swing", "sum"))
        .round({"StuffPlus": 1})
        .reset_index()
        .sort_values("StuffPlus", ascending=False)
    )


def pitcher_level_stuff_plus(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse the PitchType-level summary to one pitch-count-weighted Stuff+ per pitcher."""
    weighted = summary.assign(Weighted=summary["StuffPlus"] * summary["Pitches"])
    agg = {"WeightedSum": ("Weighted", "sum"), "Pitches": ("Pitches", "sum")}
    if "Swings" in summary:
        agg["Swings"] = ("Swings", "sum")
    out = weighted.groupby(["Pitcher", "PitcherId", "PitcherTeam"]).agg(**agg).reset_index()
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


def add_regressed(summary: pd.DataFrame, pitcher_level: pd.DataFrame, thrown: pd.Series):
    """StuffPlus_Scaled_Reg / StuffPlus_Reliability, regressed toward 100 (the qualified pool's mean).

    n = scored pitches (per pitch type, or overall); k from data/stabilization.csv
    (per pitch type, falling back to the overall k for a type with none).
    Blank (NaN) for pitchers under config.MIN_PITCHES_TO_DISPLAY pitches thrown
    and pitch types under config.MIN_PITCHES_PER_TYPE_TO_DISPLAY -- the raw
    columns are always kept as they are.
    """
    summary = summary.copy()
    k = summary["PitchType"].map(lambda p: load_k(f"StuffPlus_Scaled[{p}]", fallback="StuffPlus_Scaled"))
    summary["StuffPlus_Reliability"] = reliability(summary["Pitches"], k).round(3)
    summary["StuffPlus_Scaled_Reg"] = regress(summary["StuffPlus_Scaled"], summary["Pitches"], k, 100).round(1)
    hidden = (summary["Pitches"] < MIN_PITCHES_PER_TYPE_TO_DISPLAY) | (summary["PitcherId"].map(thrown) < MIN_PITCHES_TO_DISPLAY)
    summary.loc[hidden, "StuffPlus_Scaled_Reg"] = np.nan

    pitcher_level = pitcher_level.copy()
    pitcher_level["PitchesThrown"] = pitcher_level["PitcherId"].map(thrown)
    pitcher_level["Qualified"] = pitcher_level["PitchesThrown"] >= MIN_PITCHES_FOR_INCLUSION
    k = load_k("StuffPlus_Scaled")
    pitcher_level["StuffPlus_Reliability"] = reliability(pitcher_level["Pitches"], k).round(3)
    pitcher_level["StuffPlus_Scaled_Reg"] = regress(pitcher_level["StuffPlus_Scaled"], pitcher_level["Pitches"], k, 100).round(1)
    pitcher_level.loc[pitcher_level["PitchesThrown"] < MIN_PITCHES_TO_DISPLAY, "StuffPlus_Scaled_Reg"] = np.nan
    return summary, pitcher_level


# ── v1: the original model, kept only for StuffPlus_v1 ──────────────────────

def fit_stuff_plus_models_v1(df: pd.DataFrame) -> dict:
    """v1: one model per pitch type on all swings, both hands in the catcher's frame. {type: (model, scaler, baseline)}."""
    df = label_swings(df)
    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type"]).copy()
    models = {}
    for pitch_type, group in swings.groupby("pitch_type"):
        if len(group) < MIN_PITCHES_PER_TYPE:
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(group[FEATURES])
        model = new_model().fit(X_scaled, group["Whiff"])
        models[pitch_type] = (model, scaler, model.predict_proba(X_scaled)[:, 1].mean())
    return models


def score_stuff_plus_v1(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """v1: score only the swings, in-sample."""
    df = label_swings(df)
    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type", "Pitcher"])
    results = []
    for pitch_type, group in swings.groupby("pitch_type"):
        if pitch_type not in models:
            continue
        model, scaler, baseline = models[pitch_type]
        pred = model.predict_proba(scaler.transform(group[FEATURES]))[:, 1]
        results.append(pd.DataFrame({
            "Pitcher": group["Pitcher"], "PitcherId": group["pitcher"], "PitcherTeam": group["PitcherTeam"],
            "PitchType": pitch_type, "StuffPlus": pred / baseline * 100,
        }))
    return pd.concat(results, ignore_index=True)


def league_v1_summary(train_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """v1 per pitcher x pitch type ratio StuffPlus (25+ scored swings), cached."""
    if V1_SUMMARY_PATH.exists() and not force_refresh:
        return pd.read_parquet(V1_SUMMARY_PATH)
    print("Re-fitting the v1 Stuff+ model (for StuffPlus_v1)...")
    scoring = train_df.assign(PitcherTeam=train_df["pitcher"].map(primary_team(train_df)))
    per_pitch = score_stuff_plus_v1(scoring, fit_stuff_plus_models_v1(train_df))
    summary = (
        per_pitch.groupby(["Pitcher", "PitcherId", "PitcherTeam", "PitchType"])
        .agg(StuffPlus=("StuffPlus", "mean"), Pitches=("StuffPlus", "size"))
        .round({"StuffPlus": 1}).reset_index()
    )
    summary = summary[summary["Pitches"] >= MIN_SAMPLE_FOR_SUMMARY_V1]
    summary.to_parquet(V1_SUMMARY_PATH, index=False)
    return summary


def v1_tables(train_df: pd.DataFrame, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitcher x type, pitcher) v1 StuffPlus_v1 / StuffPlus_Scaled_v1, graded exactly as before."""
    summary = league_v1_summary(train_df, force_refresh)
    pitcher_pool, type_pool = league_scale_pool(train_df, summary)
    summary, pitcher = add_scaled_stuff_plus(summary, pitcher_level_stuff_plus(summary), pitcher_pool, type_pool)
    rename = {"StuffPlus": "StuffPlus_v1", "StuffPlus_Scaled": "StuffPlus_Scaled_v1"}
    return (summary[["PitcherId", "PitchType", "StuffPlus", "StuffPlus_Scaled"]].rename(columns=rename),
            pitcher[["PitcherId", "StuffPlus", "StuffPlus_Scaled"]].rename(columns=rename))


# ── League grading ───────────────────────────────────────────────────────────

def grade_league(force_refresh: bool = False, train_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """League-wide (pitcher x pitch type, pitcher) Stuff+ tables, saved; plus the training frame.

    One row per PitcherId (traded pitchers labeled with their primary team).
    Pass `train_df` (load_season_monthly with at least RAW_TRAIN_COLUMNS) to
    reuse a frame that's already loaded.
    """
    if train_df is None:
        print(f"Pulling a league-wide training sample {TRAIN_START} to {TRAIN_END} (month by month)...")
        train_df = load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS)
    print(f"Loaded {len(train_df)} cleaned training pitches")

    league_pitches = build_league_scores(train_df, force_refresh=force_refresh)
    summary = summarize_stuff_plus(league_pitches)
    pitcher_pool, type_pool = league_scale_pool(train_df, summary)
    summary, pitcher = add_scaled_stuff_plus(summary, pitcher_level_stuff_plus(summary), pitcher_pool, type_pool)
    thrown = train_df.groupby("pitcher").size()
    summary, pitcher = add_regressed(summary, pitcher, thrown)

    v1_type, v1_pitcher = v1_tables(train_df, force_refresh)
    summary = summary.merge(v1_type, on=["PitcherId", "PitchType"], how="left")
    pitcher = pitcher.merge(v1_pitcher, on="PitcherId", how="left")
    pitcher["p_throws"] = pitcher["PitcherId"].map(train_df.drop_duplicates("pitcher").set_index("pitcher")["p_throws"])

    summary.to_parquet(LEAGUE_SUMMARY_PATH, index=False)
    pitcher.to_parquet(LEAGUE_PITCHER_PATH, index=False)
    print(f"Saved {len(summary)} pitcher x pitch type rows to {LEAGUE_SUMMARY_PATH}")
    print(f"Saved {len(pitcher)} pitcher rows ({pitcher['Qualified'].sum()} qualified) to {LEAGUE_PITCHER_PATH}")
    return summary, pitcher, train_df


def compare_to_v1(pitcher: pd.DataFrame) -> pd.DataFrame:
    """Correlation of v1 vs. new Stuff+ (qualified pitchers), overall / RHP / LHP, and the 10 biggest movers."""
    q = pitcher[pitcher["Qualified"]].dropna(subset=["StuffPlus", "StuffPlus_v1"]).copy()
    q["ScaledChange"] = q["StuffPlus_Scaled"] - q["StuffPlus_Scaled_v1"]
    print("\n── Stuff+ v1 vs. new (qualified pitchers) ──")
    for label, rows in [("All", q), ("RHP", q[q["p_throws"] == "R"]), ("LHP", q[q["p_throws"] == "L"])]:
        print(f"{label}: n = {len(rows)}, corr(StuffPlus_v1, StuffPlus) = {rows['StuffPlus_v1'].corr(rows['StuffPlus']):.3f}, "
              f"corr(scaled) = {rows['StuffPlus_Scaled_v1'].corr(rows['StuffPlus_Scaled']):.3f}, "
              f"mean |scaled change| = {rows['ScaledChange'].abs().mean():.1f}")
    movers = q.reindex(q["ScaledChange"].abs().sort_values(ascending=False).index).head(10)
    print("\n10 biggest changes in StuffPlus_Scaled:")
    print(movers[["Pitcher", "PitcherTeam", "p_throws", "Pitches", "StuffPlus_v1", "StuffPlus",
                  "StuffPlus_Scaled_v1", "StuffPlus_Scaled", "ScaledChange"]].round(1).to_string(index=False))
    q.to_csv(V1_COMPARISON_PATH, index=False)
    print(f"Saved the full comparison to {V1_COMPARISON_PATH}")
    return q


# ── Dashboard exports ────────────────────────────────────────────────────────

def save_for_dashboard(summary: pd.DataFrame, pitcher_level: pd.DataFrame):
    """Upsert this run's per-pitch-type Stuff+ rows (plus each pitcher's overall) into the dashboard's CSV.

    Graded against the league-wide qualified pool, so 100 is a league
    average, not this subset's own. Upserted by PitcherId x PitcherTeam, so a
    traded pitcher keeps a row under each team he was run for, and re-running
    a team replaces just that team's rows for those pitchers.
    """
    overall = pitcher_level[[
        "PitcherId", "PitchesThrown", "Pitches", "Swings", "StuffPlus", "StuffPlus_Scaled", "StuffPlus_Scaled_Reg",
        "StuffPlus_Reliability", "StuffPlus_v1", "StuffPlus_Scaled_v1",
    ]].rename(columns=lambda c: c if c in ("PitcherId", "PitchesThrown") else f"Overall{c}")
    rows = summary.merge(overall, on="PitcherId")[[
        "Pitcher", "PitcherId", "PitcherTeam", "PitchType", "Pitches", "Swings", "StuffPlus", "StuffPlus_Scaled",
        "StuffPlus_Scaled_Reg", "StuffPlus_Reliability", "StuffPlus_v1", "StuffPlus_Scaled_v1",
        "PitchesThrown", "OverallPitches", "OverallSwings", "OverallStuffPlus", "OverallStuffPlus_Scaled",
        "OverallStuffPlus_Scaled_Reg", "OverallStuffPlus_Reliability", "OverallStuffPlus_v1", "OverallStuffPlus_Scaled_v1",
    ]]
    upsert_csv(rows, DASHBOARD_PATH, keys=["PitcherId", "PitcherTeam"])


def upsert_csv(rows: pd.DataFrame, path: Path, keys: list[str] | None = None):
    """Replace this run's `keys` (default PitcherId) in `path`, keep everyone else."""
    keys = keys or ["PitcherId"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if set(keys) <= set(existing.columns):
            replaced = pd.MultiIndex.from_frame(existing[keys]).isin(pd.MultiIndex.from_frame(rows[keys].drop_duplicates()))
            existing = existing[~replaced]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(path, index=False)
    print(f"Saved {len(rows)} row(s) to {path}")


def purge_non_pitchers_from_dashboard():
    """Drop position players' rows (left by older runs) from every upserted dashboard CSV."""
    for path in [DASHBOARD_PATH, DASHBOARD_PITCHES_PATH, MOVEMENT_PITCHES_PATH, VELOCITY_PATH]:
        if path.exists():
            rows = pd.read_csv(path)
            kept = pitchers_only(rows, "PitcherId")
            if len(kept) < len(rows):
                kept.to_csv(path, index=False)
                print(f"Removed {len(rows) - len(kept)} non-pitcher row(s) from {path.name}")


def save_pitches_for_dashboard(per_pitch: pd.DataFrame):
    """Upsert this run's scored pitches (per-pitch StuffPlus)."""
    rows = per_pitch[["Pitcher", "PitcherId", "PitcherTeam", "PitchType", "release_speed", "ivb", "hb", "StuffPlus"]]
    upsert_csv(rows.round({"ivb": 2, "hb": 2, "StuffPlus": 1}), DASHBOARD_PITCHES_PATH)


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
    Raw catcher's-view hb/release_pos_x -- the chart does its own arm-side flip.
    """
    rows = scope_df.rename(columns={"pitcher": "PitcherId", "pitch_type": "PitchType"}).copy()
    if "arm_angle" not in rows:
        rows["arm_angle"] = np.nan
    rows["arm_angle"] = rows["arm_angle"].astype(float)
    has_statcast = rows.groupby("PitcherId")["arm_angle"].transform(lambda s: s.notna().any())
    rows["ArmAngleSource"] = np.where(has_statcast, "Statcast", "release-point estimate")
    rows.loc[~has_statcast, "arm_angle"] = estimate_arm_angle(rows.loc[~has_statcast])
    upsert_csv(rows[MOVEMENT_COLUMNS].round({"ivb": 2, "hb": 2, "arm_angle": 1}), MOVEMENT_PITCHES_PATH)


def primary_fastball(pitch_types: pd.Series) -> str | None:
    """FF, else SI, else FC -- the first with MIN_PITCHES_FOR_PRIMARY_FASTBALL pitches; else the most-thrown of the three."""
    counts = pitch_types.value_counts()
    for pitch_type in PRIMARY_FASTBALL_ORDER:
        if counts.get(pitch_type, 0) >= MIN_PITCHES_FOR_PRIMARY_FASTBALL:
            return pitch_type
    thrown = counts.reindex(PRIMARY_FASTBALL_ORDER).dropna()
    return thrown.idxmax() if not thrown.empty else None


def save_velocity_for_dashboard(scope_df: pd.DataFrame):
    """Upsert one row per pitcher x appearance: primary fastball's average velocity, pitches thrown, opponent."""
    rows = []
    for pitcher_id, pitches in scope_df.groupby("pitcher"):
        fastball = primary_fastball(pitches["pitch_type"])
        if fastball is None:
            continue
        opponent = np.where(pitches["inning_topbot"] == "Top", pitches["away_team"], pitches["home_team"])
        games = pitches.assign(Opponent=opponent, IsFastball=pitches["pitch_type"] == fastball)
        per_game = games.groupby(["game_pk", "game_date"]).agg(
            Opponent=("Opponent", "first"), PitchesThrown=("pitch_type", "size"),
            FastballPitches=("IsFastball", "sum"),
        )
        per_game["AvgVelo"] = games[games["IsFastball"]].groupby(["game_pk", "game_date"])["release_speed"].mean()
        per_game = per_game.dropna(subset=["AvgVelo"]).reset_index()
        per_game.insert(0, "PitcherId", pitcher_id)
        per_game.insert(1, "PitchType", fastball)
        rows.append(per_game)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out["game_date"] = pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
        out["AvgVelo"] = out["AvgVelo"].round(2)
        upsert_csv(out.sort_values(["PitcherId", "game_date"]), VELOCITY_PATH)


def run(start_date: str, end_date: str, pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    if pitcher and team:
        raise ValueError("Pass --pitcher or --team, not both")

    summary, pitcher_level, train_df = grade_league(force_refresh=force_refresh)
    compare_to_v1(pitcher_level)
    if VALIDATION_PATH.exists():
        print_validation(pd.read_csv(VALIDATION_PATH))

    if not (pitcher or team):
        print("\n── Top 20 qualified pitchers by Stuff+ (league-wide) ──")
        print(pitcher_level[pitcher_level["Qualified"]].sort_values("StuffPlus_Scaled", ascending=False).head(20)
              .drop(columns="PitcherId").to_string(index=False))
        return

    # Grades are league-wide and out-of-fold; the scope pull decides WHO is shown
    # and feeds the movement / velocity charts.
    del train_df
    if pitcher:
        print(f"\nPulling {pitcher}'s Statcast data {start_date} to {end_date}...")
        scope_df = load_pitcher_season(pitcher, start_date, end_date, force_refresh=force_refresh)
        label = scope_df["PitcherTeam"].value_counts().idxmax()
    else:
        print(f"\nFinding {team}'s pitching roster and each pitcher's full season "
              f"(including time with other teams if traded)...")
        scope_df = load_team_roster_full_seasons(team, start_date, end_date, force_refresh=force_refresh)
        label = team
    export_scope(summary, pitcher_level, scope_df, label, pitcher or team)


def export_scope(summary: pd.DataFrame, pitcher_level: pd.DataFrame, scope_df: pd.DataFrame, label: str, title: str,
                 league_pitches: pd.DataFrame | None = None):
    """Print the scope's staff (graded vs. league) and upsert its rows into every dashboard CSV.

    scope_df = every pitch the scope's pitchers threw (cleaned Statcast);
    label = the PitcherTeam written for them. league_pitches defaults to
    the cached LEAGUE_PITCHES_PATH (pass it in to avoid re-reading it per team).
    """
    print(f"Loaded {len(scope_df)} cleaned pitches")
    ids = scope_df["pitcher"].unique()

    scope_summary = summary[summary["PitcherId"].isin(ids)].assign(PitcherTeam=label)
    scope_pitcher = pitcher_level[pitcher_level["PitcherId"].isin(ids)].assign(PitcherTeam=label)
    per_pitch = league_pitches if league_pitches is not None else pd.read_parquet(LEAGUE_PITCHES_PATH)
    per_pitch = per_pitch[per_pitch["PitcherId"].isin(ids)].assign(PitcherTeam=label)
    if scope_pitcher.empty:
        raise ValueError(f"'{title}' not found in the league Stuff+ table")
    scope_summary.to_parquet(SUMMARY_PATH, index=False)
    per_pitch.to_parquet(PITCHES_PATH, index=False)

    print(f"\n── {title} pitching staff (graded vs. league) ──")
    print(scope_pitcher.sort_values("StuffPlus_Scaled_Reg", ascending=False).drop(columns="PitcherId").to_string(index=False))
    save_for_dashboard(scope_summary, scope_pitcher)
    save_pitches_for_dashboard(per_pitch)
    save_movement_for_dashboard(scope_df)
    save_velocity_for_dashboard(scope_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Stuff+ dataset from Statcast via pybaseball.")
    parser.add_argument("--start", default=START_DATE, help="Start date for the --pitcher/--team pull (charts), YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="End date for the --pitcher/--team pull (charts), YYYY-MM-DD")
    parser.add_argument("--pitcher", default=None, help="Also print/export one pitcher's result for the dashboard, e.g. --pitcher \"Paul Skenes\"")
    parser.add_argument("--team", default=None, help="Also print/export a whole team's pitching staff for the dashboard, e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Refit the models (out-of-fold and v1) instead of using cached scores")
    args = parser.parse_args()
    run(args.start, args.end, pitcher=args.pitcher, team=args.team, force_refresh=args.refresh)
