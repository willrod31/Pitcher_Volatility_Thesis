"""Shared helpers used across the analysis modules."""
from pathlib import Path

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
