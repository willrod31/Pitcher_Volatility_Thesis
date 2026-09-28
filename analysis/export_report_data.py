"""
Hand-off point between the full analysis pipeline and the dashboard.

Joins stuff_plus_proxy.py, asymmetric_upside.py, volatility_discount.py, and
risk_adjusted_value.py output into one flat CSV with a fixed column
contract, for the eventual multi-pitcher League Mispricing Board.

Written league-wide (every graded pitcher, keyed on PitcherId); the
dashboard joins it to its own Stuff+ rows on PitcherId and filters by team.
Also copies data/surplus_by_year.csv next to it for the per-pitcher chart.

Column contract:
    PitcherId, Pitcher, Team, StuffPlus, StuffPlus_Scaled, LocationPlus,
    CommandProxyLegacy, EdgePct, MeatballPct, BBPct, VolatilityPath,
    VolatilityPercentile, TotalDaysMissed, ArmILStints, InjuryRiskScore,
    RiskAdjWAR, ContractStatus, Salary_M, YearsControl, ControlYearsUsed,
    ControlSource, SurplusCurrentSeason_M, MultiYearSurplus_M,
    SurplusPerControlYear_M, AsymmetricUpsideIndex, PitchMixInefficient,
    MostUsedPitch, BestPitch
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis.risk_adjusted_value import OUT_PATH as RISK_ADJUSTED_PATH, SURPLUS_BY_YEAR_PATH

DASHBOARD_DATA = Path(__file__).resolve().parent.parent / "dashboard" / "data"
OUT_PATH = DASHBOARD_DATA / "pitcher_report_data.csv"
SURPLUS_OUT_PATH = DASHBOARD_DATA / "surplus_by_year.csv"

COLUMN_CONTRACT = [
    "PitcherId", "Pitcher", "Team", "StuffPlus", "StuffPlus_Scaled", "LocationPlus",
    "CommandProxyLegacy", "EdgePct", "MeatballPct", "BBPct", "VolatilityPath",
    "VolatilityPercentile", "TotalDaysMissed", "ArmILStints", "InjuryRiskScore",
    "RiskAdjWAR", "ContractStatus", "Salary_M", "YearsControl", "ControlYearsUsed",
    "ControlSource", "SurplusCurrentSeason_M", "MultiYearSurplus_M",
    "SurplusPerControlYear_M", "AsymmetricUpsideIndex", "PitchMixInefficient",
    "MostUsedPitch", "BestPitch",
]


def millions(series: pd.Series) -> pd.Series:
    return (series / 1_000_000).round(2)


def run():
    if not RISK_ADJUSTED_PATH.exists():
        raise FileNotFoundError(f"{RISK_ADJUSTED_PATH} not found -- run `python -m analysis.risk_adjusted_value` first.")

    df = pd.read_parquet(RISK_ADJUSTED_PATH)

    out = pd.DataFrame({
        "PitcherId": df["PitcherId"],
        "Pitcher": df["Pitcher"],
        "Team": df["PitcherTeam"],
        "StuffPlus": df["StuffPlus"].round(1),
        "StuffPlus_Scaled": df["StuffPlus_Scaled"].round(1),
        "LocationPlus": df["LocationPlus"].round(1),
        "CommandProxyLegacy": df["CommandProxyLegacy"].round(2),
        "EdgePct": df["EdgePct"].round(3),
        "MeatballPct": df["MeatballPct"].round(3),
        "BBPct": df["BBPct"].round(3),
        "VolatilityPath": df["VolatilityPath"],
        "VolatilityPercentile": df["VolatilityPercentile"].round(1),
        "TotalDaysMissed": df["TotalDaysMissed"],
        "ArmILStints": df["ArmILStints"],
        "InjuryRiskScore": df["InjuryRiskScore"].round(2),
        "RiskAdjWAR": df["risk_adjusted_war"].round(2),
        "ContractStatus": df["ContractStatus"],
        "Salary_M": millions(df["Salary"]),
        "YearsControl": df["YearsControl"],
        "ControlYearsUsed": df["ControlYearsUsed"],
        "ControlSource": df["ControlSource"],
        "SurplusCurrentSeason_M": millions(df["SurplusCurrentSeason"]),
        "MultiYearSurplus_M": millions(df["MultiYearSurplus"]),
        "SurplusPerControlYear_M": millions(df["SurplusPerControlYear"]),
        "AsymmetricUpsideIndex": df["AsymmetricUpsideIndex"].round(2),
        "PitchMixInefficient": df["PitchMixInefficient"],
        "MostUsedPitch": df["MostUsedPitch"],
        "BestPitch": df["BestPitch"],
    })[COLUMN_CONTRACT]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUT_PATH}")

    if SURPLUS_BY_YEAR_PATH.exists():
        by_year = pd.read_csv(SURPLUS_BY_YEAR_PATH)
        by_year.to_csv(SURPLUS_OUT_PATH, index=False)
        print(f"Saved {len(by_year)} rows to {SURPLUS_OUT_PATH}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Join all analysis output into the dashboard's report CSV.").parse_args()
    run()
