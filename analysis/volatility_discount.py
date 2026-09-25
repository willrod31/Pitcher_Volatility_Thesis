"""
Volatility Discount Metric.

Computes per-start variance in release point and velocity, then regresses
next-start performance on current (trailing) volatility to test whether
volatility actually predicts decline -- as opposed to just being something
the market appears to penalize.

Also pulls in IL history from injury_history.py (run that first) to see if
volatility predicts going on the IL, not just a bad next start.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from config import SEASON
from analysis.data_acquisition import filter_qualified, load_season
from analysis.injury_history import STINTS_PATH, SUMMARY_PATH
from analysis.utils import CACHE_DIR, percentile_rank, zscore

OUT_PATH = CACHE_DIR / "volatility_discount.parquet"

MIN_PITCHES_PER_START = 20   # drops relief cameos / openers from the "start" unit
MIN_STARTS = 8                # need enough starts for variance to mean anything
TRAILING_MIN_STARTS = 3       # min prior starts before trailing volatility is computed
IL_WINDOW_DAYS = 30           # "went on the IL soon after this start" window


def build_start_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level data to one row per Pitcher x start (game_pk)."""
    starts = df.rename(columns={"pitcher": "PitcherId"}).groupby(["Pitcher", "PitcherId", "PitcherTeam", "game_pk", "game_date"]).agg(
        Pitches=("release_speed", "size"),
        MeanVelo=("release_speed", "mean"),
        MeanRelX=("release_pos_x", "mean"),
        MeanRelZ=("release_pos_z", "mean"),
        RunValue=("delta_run_exp", lambda s: -s.sum()),  # runs prevented; higher = better start
    ).reset_index()
    starts = starts[starts["Pitches"] >= MIN_PITCHES_PER_START]
    return starts.sort_values(["Pitcher", "game_date"])


def compute_season_volatility(starts: pd.DataFrame) -> pd.DataFrame:
    """Season-long start-to-start volatility -- the number the dashboard shows."""
    counts = starts.groupby("Pitcher").size()
    qualified = counts[counts >= MIN_STARTS].index
    starts = starts[starts["Pitcher"].isin(qualified)]

    per_pitcher = starts.groupby("Pitcher").agg(
        PitcherId=("PitcherId", "first"),
        PitcherTeam=("PitcherTeam", "first"),
        Starts=("game_pk", "size"),
        VeloVolatility=("MeanVelo", "std"),
        RelXVolatility=("MeanRelX", "std"),
        RelZVolatility=("MeanRelZ", "std"),
    ).reset_index()

    per_pitcher["ReleaseVolatility"] = np.sqrt(
        per_pitcher["RelXVolatility"] ** 2 + per_pitcher["RelZVolatility"] ** 2
    )
    per_pitcher["VolatilityScore"] = zscore(per_pitcher["VeloVolatility"]) + zscore(per_pitcher["ReleaseVolatility"])
    per_pitcher["VolatilityPercentile"] = percentile_rank(per_pitcher["VolatilityScore"]).round(1)
    return per_pitcher


def compute_trailing_volatility(starts: pd.DataFrame) -> pd.DataFrame:
    """Expanding (prior-starts-only) volatility as of each start, paired with the NEXT start's result.

    Uses an expanding window rather than a fixed 30-day one so it works off
    start count (consistent across pitchers regardless of missed time);
    swap in a date-based rolling window here if pitcher-specific rest gaps
    matter more than start count.
    """
    starts = starts.sort_values(["Pitcher", "game_date"]).copy()
    grouped = starts.groupby("Pitcher")

    starts["TrailingVeloVolatility"] = grouped["MeanVelo"].transform(
        lambda s: s.expanding(min_periods=TRAILING_MIN_STARTS).std().shift(1)
    )
    rel_x_std = grouped["MeanRelX"].transform(lambda s: s.expanding(min_periods=TRAILING_MIN_STARTS).std().shift(1))
    rel_z_std = grouped["MeanRelZ"].transform(lambda s: s.expanding(min_periods=TRAILING_MIN_STARTS).std().shift(1))
    starts["TrailingReleaseVolatility"] = np.sqrt(rel_x_std ** 2 + rel_z_std ** 2)

    starts["NextRunValue"] = grouped["RunValue"].shift(-1)
    return starts


def run_predictive_regression(starts_with_trailing: pd.DataFrame) -> LinearRegression:
    """Does trailing volatility predict the NEXT start's run value?

    Pooled OLS across all pitcher-starts (not per-pitcher) -- exploratory,
    not causal: confirms whether the market's volatility penalty is
    justified by actual future performance, or is overreacting to noise.
    """
    reg_cols = ["TrailingVeloVolatility", "TrailingReleaseVolatility", "NextRunValue"]
    data = starts_with_trailing.dropna(subset=reg_cols)

    X = data[["TrailingVeloVolatility", "TrailingReleaseVolatility"]]
    y = data["NextRunValue"]

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    print("── Volatility -> Next-Start Performance Regression ──")
    print(f"n = {len(data)} pitcher-starts")
    print(f"R^2 = {r2:.4f}")
    print(f"coef TrailingVeloVolatility    = {model.coef_[0]:.4f}")
    print(f"coef TrailingReleaseVolatility = {model.coef_[1]:.4f}")
    print(
        "(negative coefficients would mean more trailing volatility -> worse "
        "next start, i.e. volatility is predictive, not just penalized)"
    )
    return model


def attach_injury_history(season_volatility: pd.DataFrame) -> pd.DataFrame:
    """Join IL history on PitcherId. NaN = not pulled yet, 0 = no IL time."""
    if not SUMMARY_PATH.exists():
        print(f"No {SUMMARY_PATH.name} yet -- run python -m analysis.injury_history --team XXX first")
        return season_volatility
    summary = pd.read_csv(SUMMARY_PATH)
    cols = ["PitcherId", "TotalDaysMissed", "DaysMissedLastSeason", "ArmILStints", "Had60DayIL"]
    return season_volatility.merge(summary[cols], on="PitcherId", how="left")


def add_il_outcomes(starts: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Flag starts followed by an IL stint within IL_WINDOW_DAYS.

    PriorILDays/PriorArmILStints only count seasons before SEASON. The summary
    totals include this season, so using them would include the stints we're
    trying to predict.
    """
    starts = starts.copy()
    stints = stints.copy()
    stints["StartDate"] = pd.to_datetime(stints["StartDate"])
    # a stint split across two seasons is still one IL trip
    placements = stints.drop_duplicates("StintId")

    covered = set(pd.read_csv(SUMMARY_PATH)["PitcherId"]) if SUMMARY_PATH.exists() else set()
    game_dates = pd.to_datetime(starts["game_date"])

    went_on_il = []
    for pid, date in zip(starts["PitcherId"], game_dates):
        if pid not in covered:
            went_on_il.append(np.nan)
            continue
        il_starts = placements.loc[placements["PitcherId"] == pid, "StartDate"]
        hit = ((il_starts > date) & (il_starts <= date + pd.Timedelta(days=IL_WINDOW_DAYS))).any()
        went_on_il.append(float(hit))
    starts[f"ILWithin{IL_WINDOW_DAYS}Days"] = went_on_il

    prior = stints[stints["Season"] < SEASON]
    prior_days = prior.groupby("PitcherId")["DaysMissed"].sum()
    prior_arm = prior[prior["IsArmInjury"]].drop_duplicates("StintId").groupby("PitcherId").size()
    in_covered = starts["PitcherId"].isin(covered)
    starts["PriorILDays"] = np.where(in_covered, starts["PitcherId"].map(prior_days).fillna(0), np.nan)
    starts["PriorArmILStints"] = np.where(in_covered, starts["PitcherId"].map(prior_arm).fillna(0), np.nan)
    return starts


def run_injury_regression(starts_with_outcomes: pd.DataFrame):
    """Logistic regression: trailing volatility -> IL stint in the next IL_WINDOW_DAYS.

    Prior IL history as controls. Exploratory, sample is small until more
    teams have injury data pulled.
    """
    target = f"ILWithin{IL_WINDOW_DAYS}Days"
    features = ["TrailingVeloVolatility", "TrailingReleaseVolatility", "PriorILDays", "PriorArmILStints"]
    data = starts_with_outcomes.dropna(subset=features + [target])

    print(f"\n── Volatility -> IL Within {IL_WINDOW_DAYS} Days (Logistic) ──")
    print(f"n = {len(data)} pitcher-starts, {int(data[target].sum())} followed by an IL stint")
    if data[target].nunique() < 2:
        print("Not enough IL outcomes to fit yet -- pull injury history for more teams.")
        return None

    X = (data[features] - data[features].mean()) / data[features].std()  # standardize so coefs are comparable
    model = LogisticRegression().fit(X, data[target])
    for name, coef in zip(features, model.coef_[0]):
        print(f"coef {name:<26} = {coef:.4f}")
    print("(positive volatility coefficients = more volatile pitchers are likelier to get hurt soon after)")
    return model


def run(force_refresh: bool = False):
    df = load_season(force_refresh=force_refresh)
    df = filter_qualified(df)

    starts = build_start_level(df)

    season_volatility = compute_season_volatility(starts)
    season_volatility = attach_injury_history(season_volatility)
    season_volatility.to_parquet(OUT_PATH, index=False)

    print("── Top 20 Most Volatile Pitchers ──")
    print(
        season_volatility.sort_values("VolatilityPercentile", ascending=False)
        .head(20).round(2).to_string(index=False)
    )
    print(f"\nSaved {len(season_volatility)} rows to {OUT_PATH}")

    trailing = compute_trailing_volatility(starts)
    run_predictive_regression(trailing)

    if STINTS_PATH.exists():
        trailing = add_il_outcomes(trailing, pd.read_csv(STINTS_PATH))
        run_injury_regression(trailing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility Discount Metric.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(force_refresh=args.refresh)
