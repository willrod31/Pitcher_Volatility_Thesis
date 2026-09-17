"""
Risk-Adjusted Value Projection.

Combines asymmetric_upside.py and volatility_discount.py into a projected WAR proxy, shrunk toward
replacement level as volatility increases. Ranks pitchers by surplus value
(risk-adjusted projected value minus contract cost).

simple_projected_war is an UNCALIBRATED PLACEHOLDER: it converts overall
talent (Stuff+ and Command Proxy, z-scored and summed) into WAR using a
fixed WAR-per-standard-deviation constant (config.WAR_PER_TALENT_Z), not a
regression against actual historical WAR. Treat it as a rank-ordering
device, not a real WAR projection, until it's regressed against a real
training sample -- see README "Known limitations".

Contract cost isn't available via pybaseball (see README) -- this looks for
a manually-collected data/salaries.csv with columns [Pitcher, Salary] (in
dollars) and merges it in if present; otherwise Salary/Surplus are left NaN.
"""
import argparse

import pandas as pd

from config import (
    DOLLARS_PER_WAR,
    LEAGUE_AVG_STARTER_WAR,
    REPLACEMENT_LEVEL_WAR,
    VOLATILITY_SHRINKAGE,
    WAR_PER_TALENT_Z,
)
from src.asymmetric_upside import OUT_PATH as UPSIDE_PATH
from src.volatility_discount import OUT_PATH as VOLATILITY_PATH
from src.utils import CACHE_DIR, zscore

OUT_PATH = CACHE_DIR / "risk_adjusted_value.parquet"
SALARIES_PATH = CACHE_DIR / "salaries.csv"


def load_salaries() -> pd.DataFrame | None:
    if not SALARIES_PATH.exists():
        print(
            f"No {SALARIES_PATH} found -- Salary/Surplus will be left blank. "
            "Add a CSV with columns [Pitcher, Salary] (dollars) to fill them in."
        )
        return None
    return pd.read_csv(SALARIES_PATH)


def compute_projected_war(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    talent_z = zscore(merged["StuffPlus"]) + zscore(merged["CommandProxy"])
    merged["simple_projected_war"] = LEAGUE_AVG_STARTER_WAR + WAR_PER_TALENT_Z * talent_z
    return merged


def apply_volatility_shrinkage(merged: pd.DataFrame) -> pd.DataFrame:
    """Pull projected WAR back toward replacement level as volatility rises.

    A pitcher at the 0th volatility percentile keeps their full projected
    upside; one at the 100th percentile has VOLATILITY_SHRINKAGE (default
    60%) of their upside over replacement pulled back.
    """
    merged = merged.copy()
    shrink_frac = VOLATILITY_SHRINKAGE * (merged["VolatilityPercentile"].fillna(50) / 100)
    upside_over_replacement = merged["simple_projected_war"] - REPLACEMENT_LEVEL_WAR
    merged["risk_adjusted_war"] = REPLACEMENT_LEVEL_WAR + upside_over_replacement * (1 - shrink_frac)
    return merged


def run():
    if not UPSIDE_PATH.exists():
        raise FileNotFoundError(f"{UPSIDE_PATH} not found -- run `python -m src.asymmetric_upside` first.")
    if not VOLATILITY_PATH.exists():
        raise FileNotFoundError(f"{VOLATILITY_PATH} not found -- run `python -m src.volatility_discount` first.")

    upside = pd.read_parquet(UPSIDE_PATH)
    volatility = pd.read_parquet(VOLATILITY_PATH)

    merged = upside.merge(
        volatility[["Pitcher", "VolatilityPercentile"]], on="Pitcher", how="left"
    )

    merged = compute_projected_war(merged)
    merged = apply_volatility_shrinkage(merged)

    salaries = load_salaries()
    if salaries is not None:
        merged = merged.merge(salaries, on="Pitcher", how="left")
    else:
        merged["Salary"] = pd.NA

    merged["surplus_value"] = merged["risk_adjusted_war"] * DOLLARS_PER_WAR - merged["Salary"]

    merged = merged.sort_values("surplus_value", ascending=False, na_position="last")
    merged.to_parquet(OUT_PATH, index=False)

    print("── Top 20 by Risk-Adjusted Surplus Value ──")
    cols = [
        "Pitcher", "PitcherTeam", "StuffPlus", "CommandProxy",
        "VolatilityPercentile", "simple_projected_war", "risk_adjusted_war",
        "Salary", "surplus_value",
    ]
    print(merged[cols].head(20).round(2).to_string(index=False))
    print(f"\nSaved {len(merged)} rows to {OUT_PATH}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Risk-Adjusted Value Projection.").parse_args()
    run()
