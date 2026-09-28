"""
Location+ proxy: command measured from pitch location and count ONLY.

Mirrors stuff_plus_proxy.py from the other side -- Stuff+ sees only pitch
shape (velo, movement, spin, extension, release), Location+ sees only where
the pitch ended up and the count it was thrown in. Pitch shape is never a
feature here, so Asymmetric Upside = z(Stuff+) - z(Location+) separates
"the stuff is good" from "the location is the problem".

Target (per pitch, pitcher's point of view, lower = better for the pitcher):
delta_run_exp, except on balls in play, where it's replaced by an EXPECTED
run value -- a league regression of delta_run_exp on xwOBA
(estimated_woba_using_speedangle) + balls + strikes -- so batted-ball luck
isn't credited or charged to command.

Features: norm_x (plate_x / half plate width, flipped for LHB so positive is
always the same side of the batter), norm_z (height relative to the batter's
own zone, same normalization as asymmetric_upside.classify_zone_location),
balls, strikes, same_hand.

One HistGradientBoostingRegressor per pitch group (fastball / breaking /
offspeed), trained on every pitch league-wide. Predictions are out-of-fold
(5-fold GroupKFold by pitcher), so a pitcher's own pitches never train the
model that scores him.

LocationPlus = 100 + 10 * z(-mean predicted run value), z against the
league-wide qualified pool (config.MIN_PITCHES_FOR_INCLUSION pitches; per
pitch type, config.MIN_PITCHES_PER_TYPE_FOR_SCALE and z within that pitch
type). 100 = league average, 10 points = 1 SD, higher = better.

Location+ still can't see the catcher's target: it grades location quality
given the count, not whether the pitcher hit his spot.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import END_DATE, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, START_DATE
from analysis.data_acquisition import filter_to_team, load_season_monthly, lookup_mlbam_id, primary_team
from analysis.utils import CACHE_DIR, scale_100

PITCHES_PATH = CACHE_DIR / "location_plus_pitches.parquet"
SUMMARY_PATH = CACHE_DIR / "location_plus_summary.parquet"
PITCHER_PATH = CACHE_DIR / "location_plus_pitcher.parquet"
RELIABILITY_PATH = CACHE_DIR / "command_reliability.csv"
DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "location_plus_pitch_types.csv"

# release_speed/spin/extension/pfx are loaded ONLY because load_season_monthly's
# shared cleaning step filters on them (same pitch pool as Stuff+) -- they are
# never model features.
RAW_COLUMNS = [
    "pitcher", "pitch_type", "plate_x", "plate_z", "sz_top", "sz_bot",
    "balls", "strikes", "stand", "p_throws", "type", "description", "events",
    "delta_run_exp", "estimated_woba_using_speedangle",
    "game_pk", "game_date", "at_bat_number", "pitch_number",
    "home_team", "away_team", "inning_topbot",
    "release_speed", "release_spin_rate", "release_extension", "pfx_x", "pfx_z",
]

PITCH_GROUPS = {
    "Fastball": ["FF", "SI", "FC"],
    "Breaking": ["SL", "ST", "CU", "KC", "SV"],
    "Offspeed": ["CH", "FS", "FO", "SC"],
}
PITCH_TO_GROUP = {p: g for g, types in PITCH_GROUPS.items() for p in types}

FEATURES = ["norm_x", "norm_z", "balls", "strikes", "same_hand"]
N_FOLDS = 5
PLATE_HALF_WIDTH_FT = 17 / 2 / 12

STRIKE_OR_CONTACT = {
    "called_strike", "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "foul_bunt", "missed_bunt", "bunt_foul_tip", "hit_into_play",
}


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PitchGroup"] = df["pitch_type"].map(PITCH_TO_GROUP)
    df = df.dropna(subset=["PitchGroup", "plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes", "stand", "p_throws", "delta_run_exp"])
    df = df[df["sz_top"] > df["sz_bot"]]

    side = np.where(df["stand"] == "L", -1.0, 1.0)
    df["norm_x"] = df["plate_x"] / PLATE_HALF_WIDTH_FT * side
    sz_mid = (df["sz_top"] + df["sz_bot"]) / 2
    sz_half = (df["sz_top"] - df["sz_bot"]) / 2
    df["norm_z"] = (df["plate_z"] - sz_mid) / sz_half
    df["same_hand"] = (df["p_throws"] == df["stand"]).astype(int)
    df["balls"] = df["balls"].astype(int)
    df["strikes"] = df["strikes"].astype(int)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """RunValueTarget: delta_run_exp, with balls in play replaced by an xwOBA-based expected value."""
    df = df.copy()
    df["delta_run_exp"] = df["delta_run_exp"].astype(float)
    df["RunValueTarget"] = df["delta_run_exp"]

    bip = df["type"] == "X"
    xwoba = df["estimated_woba_using_speedangle"].astype(float)
    fit_rows = bip & xwoba.notna()
    reg_cols = ["xwoba", "balls", "strikes"]
    X_fit = pd.DataFrame({"xwoba": xwoba[fit_rows], "balls": df.loc[fit_rows, "balls"], "strikes": df.loc[fit_rows, "strikes"]})
    reg = LinearRegression().fit(X_fit[reg_cols], df.loc[fit_rows, "delta_run_exp"])
    print(f"BIP expected-run-value regression: n = {fit_rows.sum()}, R^2 = {reg.score(X_fit[reg_cols], df.loc[fit_rows, 'delta_run_exp']):.3f}")

    # BIP without an xwOBA (rare tracking gaps) get the league BIP-average xwOBA
    X_bip = pd.DataFrame({
        "xwoba": xwoba[bip].fillna(xwoba[fit_rows].mean()),
        "balls": df.loc[bip, "balls"], "strikes": df.loc[bip, "strikes"],
    })
    df.loc[bip, "RunValueTarget"] = reg.predict(X_bip[reg_cols])
    return df


def out_of_fold_predictions(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """PredRunValue for every pitch from a model that never saw that pitcher; plus fold R^2 per group."""
    preds = pd.Series(np.nan, index=df.index)
    fold_rows = []
    for group, g in df.groupby("PitchGroup"):
        X, y, pitchers = g[FEATURES].to_numpy(), g["RunValueTarget"].to_numpy(), g["pitcher"].to_numpy()
        for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, pitchers), start=1):
            model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, min_samples_leaf=200, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            fold_pred = model.predict(X[test_idx])
            preds.loc[g.index[test_idx]] = fold_pred
            fold_rows.append({"PitchGroup": group, "Fold": fold, "TestPitches": len(test_idx), "R2": r2_score(y[test_idx], fold_pred)})
    return preds, pd.DataFrame(fold_rows)


def build_scored_pitches(force_refresh: bool = False) -> pd.DataFrame:
    """League-wide pitches with out-of-fold PredRunValue, cached (the slow step)."""
    if PITCHES_PATH.exists() and not force_refresh:
        print(f"Using cached out-of-fold predictions ({PITCHES_PATH.name}); --refresh to retrain")
        return pd.read_parquet(PITCHES_PATH)

    print(f"Loading league-wide Statcast {START_DATE} to {END_DATE} (month by month)...")
    # force_refresh only retrains the models; the monthly Statcast pulls stay cached
    df = load_season_monthly(START_DATE, END_DATE, columns=RAW_COLUMNS)
    df["PitcherTeam"] = df["pitcher"].map(primary_team(df))
    df = add_target(add_location_features(df))
    print(f"{len(df)} pitches in the three pitch groups")

    df["PredRunValue"], folds = out_of_fold_predictions(df)
    print("\n── Out-of-fold R^2 by pitch group (GroupKFold by pitcher) ──")
    print(folds.pivot(index="PitchGroup", columns="Fold", values="R2").assign(
        Mean=lambda t: t.mean(axis=1)).round(4).to_string())
    print("(low is expected: single-pitch run value is mostly noise)")

    keep = [
        "pitcher", "Pitcher", "PitcherTeam", "pitch_type", "PitchGroup", "game_pk", "game_date",
        "at_bat_number", "pitch_number", "plate_x", "plate_z", "sz_top", "sz_bot",
        "balls", "strikes", "description", "events", "RunValueTarget", "PredRunValue",
    ]
    out = df[keep].rename(columns={"pitcher": "PitcherId"})
    out.to_parquet(PITCHES_PATH, index=False)
    return out


def summarize_location_plus(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitcher x pitch type, pitcher) tables with LocationPlus on the 100 + 10z scale."""
    pitcher = scored.groupby("PitcherId").agg(
        Pitcher=("Pitcher", "first"), PitcherTeam=("PitcherTeam", "first"),
        Pitches=("PredRunValue", "size"), MeanPredRunValue=("PredRunValue", "mean"),
    ).reset_index()
    pitcher["Qualified"] = pitcher["Pitches"] >= MIN_PITCHES_FOR_INCLUSION
    pool = -pitcher.loc[pitcher["Qualified"], "MeanPredRunValue"]
    pitcher["LocationPlus"] = scale_100(-pitcher["MeanPredRunValue"], pool).round(1)

    by_type = scored.groupby(["PitcherId", "pitch_type"]).agg(
        Pitcher=("Pitcher", "first"), PitcherTeam=("PitcherTeam", "first"),
        Pitches=("PredRunValue", "size"), MeanPredRunValue=("PredRunValue", "mean"),
    ).reset_index().rename(columns={"pitch_type": "PitchType"})
    by_type["Qualified"] = by_type["Pitches"] >= MIN_PITCHES_PER_TYPE_FOR_SCALE
    by_type["LocationPlus"] = np.nan
    for pitch_type, rows in by_type.groupby("PitchType"):
        pool = -rows.loc[rows["Qualified"], "MeanPredRunValue"]
        if len(pool) >= 2:
            by_type.loc[rows.index, "LocationPlus"] = scale_100(-rows["MeanPredRunValue"], pool).round(1)
    return by_type, pitcher


def save_for_dashboard(by_type: pd.DataFrame, pitcher: pd.DataFrame):
    """League-wide per-pitch-type Location+ plus each pitcher's overall number, keyed on PitcherId."""
    rows = by_type.merge(
        pitcher[["PitcherId", "LocationPlus", "Qualified"]].rename(
            columns={"LocationPlus": "OverallLocationPlus", "Qualified": "OverallQualified"}),
        on="PitcherId",
    )[["PitcherId", "Pitcher", "PitchType", "Pitches", "LocationPlus", "Qualified", "OverallLocationPlus", "OverallQualified"]]
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(DASHBOARD_PATH, index=False)
    print(f"Saved {len(rows)} row(s) to {DASHBOARD_PATH}")


# ── Reliability (thesis methods) ─────────────────────────────────────────────

def _legacy_command_parts(pitches: pd.DataFrame) -> pd.DataFrame:
    """EdgePct / MeatballPct per PitcherId, same definitions as asymmetric_upside.classify_zone_location."""
    from analysis.asymmetric_upside import EDGE_BAND, MEATBALL_MAX

    sz_mid = (pitches["sz_top"] + pitches["sz_bot"]) / 2
    sz_half = (pitches["sz_top"] - pitches["sz_bot"]) / 2
    dist = np.maximum(pitches["plate_x"].abs() / PLATE_HALF_WIDTH_FT, (pitches["plate_z"] - sz_mid).abs() / sz_half)
    return pd.DataFrame({
        "PitcherId": pitches["PitcherId"],
        "Edge": dist.between(*EDGE_BAND),
        "Meatball": dist < MEATBALL_MAX,
    }).groupby("PitcherId").agg(EdgePct=("Edge", "mean"), MeatballPct=("Meatball", "mean"))


def _pa_metrics(pitches: pd.DataFrame) -> pd.DataFrame:
    """BBPct, FirstPitchStrikePct, ThreeBallPct per PitcherId, one row per plate appearance first."""
    flags = pd.DataFrame({
        "PitcherId": pitches["PitcherId"], "game_pk": pitches["game_pk"], "at_bat_number": pitches["at_bat_number"],
        "Walk": pitches["events"] == "walk",
        "Ended": pitches["events"].notna(),
        "ReachedThreeBalls": pitches["balls"] == 3,
    })
    pa = flags.groupby(["PitcherId", "game_pk", "at_bat_number"])[["Walk", "Ended", "ReachedThreeBalls"]].max()
    first = pitches[(pitches["balls"] == 0) & (pitches["strikes"] == 0)]
    first = first.drop_duplicates(["PitcherId", "game_pk", "at_bat_number"]).set_index(["PitcherId", "game_pk", "at_bat_number"])
    pa["FirstPitchStrike"] = first["description"].isin(STRIKE_OR_CONTACT).reindex(pa.index)

    per = pa.reset_index().groupby("PitcherId")
    return pd.DataFrame({
        "BBPct": per["Walk"].sum() / per["Ended"].sum(),
        "FirstPitchStrikePct": per["FirstPitchStrike"].mean(),
        "ThreeBallPct": per["ReachedThreeBalls"].mean(),
    })


def command_metrics(pitch_half: pd.DataFrame, pa_half: pd.DataFrame) -> pd.DataFrame:
    """Every command metric for one half. Pitch-level metrics use pitch_half, PA-level ones pa_half."""
    from analysis.utils import zscore

    out = pd.DataFrame({"LocationPlus": -pitch_half.groupby("PitcherId")["PredRunValue"].mean()})
    out["LocationPlus"] = scale_100(out["LocationPlus"])
    out = out.join(_legacy_command_parts(pitch_half)).join(_pa_metrics(pa_half))
    out["CommandProxy"] = zscore(out["EdgePct"]) - zscore(out["MeatballPct"]) - zscore(out["BBPct"])
    return out


RELIABILITY_METRICS = ["LocationPlus", "CommandProxy", "EdgePct", "MeatballPct", "BBPct", "FirstPitchStrikePct", "ThreeBallPct"]


def reliability_test(scored: pd.DataFrame) -> pd.DataFrame:
    """Odd/even split-half reliability of each command metric, plus its correlation with Stuff+.

    Pitch-level metrics (LocationPlus, EdgePct, MeatballPct) split each
    qualified pitcher's pitches into odd and even by order thrown. PA-level
    metrics (BBPct, FirstPitchStrikePct, ThreeBallPct) split plate
    appearances odd/even the same way, since they're defined per PA.
    CommandProxy combines the two (edge/meatball from pitch halves, BB% from
    PA halves), z-scored within each half like the original.
    SpearmanBrown = 2r / (1 + r): estimated full-season reliability.
    CorrWithStuffPlus uses the full season (lower = more separate from stuff).
    """
    counts = scored.groupby("PitcherId").size()
    qualified = counts[counts >= MIN_PITCHES_FOR_INCLUSION].index
    pitches = scored[scored["PitcherId"].isin(qualified)].sort_values(
        ["PitcherId", "game_date", "game_pk", "at_bat_number", "pitch_number"]).copy()

    pitches["PitchOrder"] = pitches.groupby("PitcherId").cumcount()
    pa_order = pitches.drop_duplicates(["PitcherId", "game_pk", "at_bat_number"])[["PitcherId", "game_pk", "at_bat_number"]]
    pa_order["PAOrder"] = pa_order.groupby("PitcherId").cumcount()
    pitches = pitches.merge(pa_order, on=["PitcherId", "game_pk", "at_bat_number"])

    odd = command_metrics(pitches[pitches["PitchOrder"] % 2 == 1], pitches[pitches["PAOrder"] % 2 == 1])
    even = command_metrics(pitches[pitches["PitchOrder"] % 2 == 0], pitches[pitches["PAOrder"] % 2 == 0])
    full = command_metrics(pitches, pitches)

    from analysis.asymmetric_upside import LEAGUE_STUFF_PATH, pitcher_stuff_plus
    stuff = None
    if LEAGUE_STUFF_PATH.exists():
        stuff = pitcher_stuff_plus(pd.read_parquet(LEAGUE_STUFF_PATH)).set_index("PitcherId")["StuffPlus"]
    else:
        print(f"No {LEAGUE_STUFF_PATH.name} yet -- run python -m analysis.asymmetric_upside for the Stuff+ correlations")

    rows = []
    for metric in RELIABILITY_METRICS:
        both = pd.concat([odd[metric].rename("odd"), even[metric].rename("even")], axis=1).dropna()
        r = both["odd"].corr(both["even"])
        stuff_r = full[metric].corr(stuff.reindex(full.index)) if stuff is not None else np.nan
        rows.append({
            "Metric": metric, "Pitchers": len(both), "SplitHalfR": r,
            "SpearmanBrown": 2 * r / (1 + r), "CorrWithStuffPlus": stuff_r,
        })
    table = pd.DataFrame(rows)
    table.to_csv(RELIABILITY_PATH, index=False)
    return table


def run(pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    if pitcher and team:
        raise ValueError("Pass --pitcher or --team, not both")

    scored = build_scored_pitches(force_refresh=force_refresh)
    by_type, pitcher_level = summarize_location_plus(scored)
    by_type.to_parquet(SUMMARY_PATH, index=False)
    pitcher_level.to_parquet(PITCHER_PATH, index=False)
    print(f"Saved {len(by_type)} pitcher x pitch type rows to {SUMMARY_PATH}")
    print(f"Saved {len(pitcher_level)} pitcher rows ({pitcher_level['Qualified'].sum()} qualified) to {PITCHER_PATH}")
    save_for_dashboard(by_type, pitcher_level)

    print("\n── Command reliability (odd/even split halves, qualified pitchers) ──")
    print(reliability_test(scored).round(3).to_string(index=False))
    print(f"Saved to {RELIABILITY_PATH}")

    # grading above is always league-wide; --pitcher/--team only filter what's shown
    if team:
        shown, title = filter_to_team(pitcher_level, team), f"{team}, graded vs. league"
    elif pitcher:
        shown, title = pitcher_level[pitcher_level["PitcherId"] == lookup_mlbam_id(pitcher)], f"{pitcher}, graded vs. league"
    else:
        shown, title = pitcher_level[pitcher_level["Qualified"]].sort_values("LocationPlus", ascending=False).head(20), "top 20 league-wide"
    print(f"\n── Location+ ({title}) ──")
    print(shown.drop(columns="PitcherId").sort_values("LocationPlus", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Location+ proxy: command from pitch location and count only.")
    parser.add_argument("--pitcher", default=None, help='Only print one pitcher (graded vs. the league), e.g. --pitcher "Paul Skenes"')
    parser.add_argument("--team", default=None, help="Only print a team's staff (graded vs. the league), e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Retrain the out-of-fold models instead of using cached predictions")
    args = parser.parse_args()
    run(pitcher=args.pitcher, team=args.team, force_refresh=args.refresh)
