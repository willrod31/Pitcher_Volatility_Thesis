"""
Hand-off point between the full analysis pipeline and the dashboard.

Joins stuff_plus_proxy.py, asymmetric_upside.py, volatility_discount.py, and
risk_adjusted_value.py output into one flat CSV with a fixed column
contract, for the eventual multi-pitcher League Mispricing Board.

Written league-wide (every graded pitcher, keyed on PitcherId); the
dashboard joins it to its own Stuff+ rows on PitcherId and filters by team.
Also copies data/surplus_by_year.csv and data/injury_stints.csv next to it
for the per-pitcher charts.

Raw and regressed (*_Reg) values are both exported, with each metric's
sample size and Reliability = n / (n + k). Qualified marks the league pool
(config.MIN_PITCHES_FOR_INCLUSION pitches) the dashboard takes league
means / SDs / percentiles from.

Column contract:
    PitcherId, Pitcher, Team, StuffPlus, StuffPlus_Scaled, LocationPlus,
    CommandProxyLegacy, EdgePct, MeatballPct, BBPct, VolatilityPath,
    VolatilityPercentile, TotalDaysMissed, ArmILStints, InjuryRiskScore,
    RiskAdjWAR, ContractStatus, Salary_M, YearsControl, ControlYearsUsed,
    ControlSource, SurplusCurrentSeason_M, MultiYearSurplus_M,
    SurplusPerControlYear_M, AsymmetricUpsideIndex, PitchMixInefficient,
    MostUsedPitch, BestPitch,
    Qualified, Pitches, PA, Appearances, StuffPlus_v1, StuffPlus_Scaled_v1,
    StuffPlus_Scaled_Reg, StuffPlus_Reliability, LocationPlus_Reg, LocationPlus_Reliability,
    EdgePct_Reg, MeatballPct_Reg, BBPct_Reg, BBPct_Reliability, CommandProxyLegacy_Reg,
    VolatilityScore, VolatilityScore_Reg, VolatilityReliability, VolatilityPercentile_Reg,
    AsymmetricUpsideIndex_Reg
"""
import argparse
from pathlib import Path

import pandas as pd

from analysis.injury_history import STINTS_PATH
from analysis.risk_adjusted_value import OUT_PATH as RISK_ADJUSTED_PATH, SURPLUS_BY_YEAR_PATH

DASHBOARD_DATA = Path(__file__).resolve().parent.parent / "dashboard" / "data"
OUT_PATH = DASHBOARD_DATA / "pitcher_report_data.csv"
SURPLUS_OUT_PATH = DASHBOARD_DATA / "surplus_by_year.csv"
STINTS_OUT_PATH = DASHBOARD_DATA / "injury_stints.csv"

COLUMN_CONTRACT = [
    "PitcherId", "Pitcher", "Team", "StuffPlus", "StuffPlus_Scaled", "LocationPlus",
    "CommandProxyLegacy", "EdgePct", "MeatballPct", "BBPct", "VolatilityPath",
    "VolatilityPercentile", "TotalDaysMissed", "ArmILStints", "InjuryRiskScore",
    "RiskAdjWAR", "ContractStatus", "Salary_M", "YearsControl", "ControlYearsUsed",
    "ControlSource", "SurplusCurrentSeason_M", "MultiYearSurplus_M",
    "SurplusPerControlYear_M", "AsymmetricUpsideIndex", "PitchMixInefficient",
    "MostUsedPitch", "BestPitch",
    "Qualified", "Pitches", "PA", "Appearances", "StuffPlus_v1", "StuffPlus_Scaled_v1",
    "StuffPlus_Scaled_Reg", "StuffPlus_Reliability", "LocationPlus_Reg", "LocationPlus_Reliability",
    "EdgePct_Reg", "MeatballPct_Reg", "BBPct_Reg", "BBPct_Reliability", "CommandProxyLegacy_Reg",
    "VolatilityScore", "VolatilityScore_Reg", "VolatilityReliability", "VolatilityPercentile_Reg",
    "AsymmetricUpsideIndex_Reg",
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
        "Qualified": df["Qualified"],
        "Pitches": df["Pitches"],
        "PA": df["PA"],
        "Appearances": df["Appearances"],
        "StuffPlus_v1": df["StuffPlus_v1"].round(1),
        "StuffPlus_Scaled_v1": df["StuffPlus_Scaled_v1"].round(1),
        "StuffPlus_Scaled_Reg": df["StuffPlus_Scaled_Reg"].round(1),
        "StuffPlus_Reliability": df["StuffPlus_Reliability"].round(3),
        "LocationPlus_Reg": df["LocationPlus_Reg"].round(1),
        "LocationPlus_Reliability": df["LocationPlus_Reliability"].round(3),
        "EdgePct_Reg": df["EdgePct_Reg"].round(3),
        "MeatballPct_Reg": df["MeatballPct_Reg"].round(3),
        "BBPct_Reg": df["BBPct_Reg"].round(3),
        "BBPct_Reliability": df["BBPct_Reliability"].round(3),
        "CommandProxyLegacy_Reg": df["CommandProxyLegacy_Reg"].round(2),
        "VolatilityScore": df["VolatilityScore"].round(2),
        "VolatilityScore_Reg": df["VolatilityScore_Reg"].round(2),
        "VolatilityReliability": df["VolatilityReliability"].round(3),
        "VolatilityPercentile_Reg": df["VolatilityPercentile_Reg"].round(1),
        "AsymmetricUpsideIndex_Reg": df["AsymmetricUpsideIndex_Reg"].round(2),
    })[COLUMN_CONTRACT]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUT_PATH}")

    if SURPLUS_BY_YEAR_PATH.exists():
        by_year = pd.read_csv(SURPLUS_BY_YEAR_PATH)
        by_year.to_csv(SURPLUS_OUT_PATH, index=False)
        print(f"Saved {len(by_year)} rows to {SURPLUS_OUT_PATH}")

    if STINTS_PATH.exists():
        stints = pd.read_csv(STINTS_PATH)
        stints.to_csv(STINTS_OUT_PATH, index=False)
        print(f"Saved {len(stints)} rows to {STINTS_OUT_PATH}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Join all analysis output into the dashboard's report CSV.").parse_args()
    run()
