"""
Volatility Discount Metric.

Computes per-start variance in release point and velocity, then regresses
next-start performance on current (trailing) volatility to test whether
volatility actually predicts decline -- as opposed to just being something
the market appears to penalize.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_acquisition import filter_qualified, load_season
from src.utils import CACHE_DIR, percentile_rank, zscore

OUT_PATH = CACHE_DIR / "volatility_discount.parquet"

MIN_PITCHES_PER_START = 20   # drops relief cameos / openers from the "start" unit
MIN_STARTS = 8                # need enough starts for variance to mean anything
TRAILING_MIN_STARTS = 3       # min prior starts before trailing volatility is computed


def build_start_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level data to one row per Pitcher x start (game_pk)."""
    starts = df.groupby(["Pitcher", "PitcherTeam", "game_pk", "game_date"]).agg(
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


def run(force_refresh: bool = False):
    df = load_season(force_refresh=force_refresh)
    df = filter_qualified(df)

    starts = build_start_level(df)

    season_volatility = compute_season_volatility(starts)
    season_volatility.to_parquet(OUT_PATH, index=False)

    print("── Top 20 Most Volatile Pitchers ──")
    print(
        season_volatility.sort_values("VolatilityPercentile", ascending=False)
        .head(20).round(2).to_string(index=False)
    )
    print(f"\nSaved {len(season_volatility)} rows to {OUT_PATH}")

    trailing = compute_trailing_volatility(starts)
    run_predictive_regression(trailing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility Discount Metric.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(force_refresh=args.refresh)
