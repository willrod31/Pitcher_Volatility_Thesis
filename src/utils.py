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
