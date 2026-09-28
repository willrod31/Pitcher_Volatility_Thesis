"""Shared helpers used across the analysis modules."""
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a series to mean 0, std 1. Constant series map to all-0."""
    std = series.std()
    if not std:
        return series * 0
    return (series - series.mean()) / std


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank a series to [0, 100], higher value = higher percentile."""
    return series.rank(pct=True) * 100


def scale_100(values: pd.Series, pool: pd.Series | None = None) -> pd.Series:
    """100 + 10 * z, with the mean/SD taken from `pool` (default: `values` itself).

    Puts a metric on the Stuff+/Location+ scale: 100 = pool average, 10 points
    = 1 standard deviation. Pass the league-wide qualified pool as `pool` so
    a subset (one team) is still graded against the league.
    """
    pool = values if pool is None else pool
    std = pool.std()
    if not std:
        return values * 0 + 100
    return 100 + 10 * (values - pool.mean()) / std


def zscore_vs(values: pd.Series, pool: pd.Series) -> pd.Series:
    """z of `values` using `pool`'s mean and SD (e.g. the qualified pool), so small samples don't move the scale."""
    std = pool.std()
    if not std:
        return values * 0
    return (values - pool.mean()) / std


def pct_vs_pool(values: pd.Series, pool: pd.Series) -> pd.Series:
    """Share of `pool` at or below each value, 0-100 (equals percentile_rank for pool members, barring ties)."""
    ranked = np.sort(pool.dropna().to_numpy())
    if not len(ranked):
        return values * np.nan
    pct = 100 * np.searchsorted(ranked, values.to_numpy(), side="right") / len(ranked)
    return pd.Series(np.where(values.isna(), np.nan, pct), index=values.index)
