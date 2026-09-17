"""
Hand-off point between analysis and dashboard.

Joins stuff_plus_proxy.py, asymmetric_upside.py, volatility_discount.py, and
risk_adjusted_value.py output into one flat CSV with a fixed column
contract. dashboard/app.py reads this file (or
dashboard/generate_sample_data.py's placeholder version) and never needs to
change when real data replaces it, as long as this contract stays the same.

Column contract:
    Pitcher, Team, StuffPlus, CommandProxy, VolatilityPercentile,
    RiskAdjWAR, Salary_M, Surplus_M, AsymmetricUpsideIndex,
    PitchMixInefficient, MostUsedPitch, BestPitch
"""
import argparse
from pathlib import Path

import pandas as pd

from src.risk_adjusted_value import OUT_PATH as RISK_ADJUSTED_PATH

OUT_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "pitcher_report_data.csv"

COLUMN_CONTRACT = [
    "Pitcher", "Team", "StuffPlus", "CommandProxy", "VolatilityPercentile",
    "RiskAdjWAR", "Salary_M", "Surplus_M", "AsymmetricUpsideIndex",
    "PitchMixInefficient", "MostUsedPitch", "BestPitch",
]


def run():
    if not RISK_ADJUSTED_PATH.exists():
        raise FileNotFoundError(f"{RISK_ADJUSTED_PATH} not found -- run `python -m src.risk_adjusted_value` first.")

    df = pd.read_parquet(RISK_ADJUSTED_PATH)

    out = pd.DataFrame({
        "Pitcher": df["Pitcher"],
        "Team": df["PitcherTeam"],
        "StuffPlus": df["StuffPlus"].round(1),
        "CommandProxy": df["CommandProxy"].round(2),
        "VolatilityPercentile": df["VolatilityPercentile"].round(1),
        "RiskAdjWAR": df["risk_adjusted_war"].round(2),
        "Salary_M": (df["Salary"] / 1_000_000).round(2),
        "Surplus_M": (df["surplus_value"] / 1_000_000).round(2),
        "AsymmetricUpsideIndex": df["AsymmetricUpsideIndex"].round(2),
        "PitchMixInefficient": df["PitchMixInefficient"],
        "MostUsedPitch": df["MostUsedPitch"],
        "BestPitch": df["BestPitch"],
    })[COLUMN_CONTRACT]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Join all analysis output into the dashboard's report CSV.").parse_args()
    run()
