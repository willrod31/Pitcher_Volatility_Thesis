"""
Placeholder data for building/demoing the dashboard UI before real Statcast
data has been run through Modules 1-3.

Writes the exact same column contract as src/export_report_data.py, so
app.py needs no changes when real data replaces this file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "data" / "pitcher_report_data.csv"

TEAMS = ["Team A", "Team B", "Team C", "Team D", "Team E", "Team F"]
PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI"]

N_PITCHERS = 30


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    stuff_plus = rng.normal(100, 12, N_PITCHERS).round(1)
    command_proxy = rng.normal(0, 1, N_PITCHERS).round(2)
    volatility_pctile = rng.uniform(0, 100, N_PITCHERS).round(1)

    talent_z = (stuff_plus - stuff_plus.mean()) / stuff_plus.std() + command_proxy
    projected_war = 2.0 + 1.5 * talent_z
    shrink = 0.6 * (volatility_pctile / 100)
    risk_adj_war = (projected_war * (1 - shrink)).round(2)

    salary_m = rng.uniform(1, 35, N_PITCHERS).round(2)
    surplus_m = (risk_adj_war * 9.5 - salary_m).round(2)

    most_used = rng.choice(PITCH_TYPES, N_PITCHERS)
    best = rng.choice(PITCH_TYPES, N_PITCHERS)

    df = pd.DataFrame({
        "Pitcher": [f"Pitcher {i + 1}" for i in range(N_PITCHERS)],
        "Team": rng.choice(TEAMS, N_PITCHERS),
        "StuffPlus": stuff_plus,
        "CommandProxy": command_proxy,
        "VolatilityPercentile": volatility_pctile,
        "RiskAdjWAR": risk_adj_war,
        "Salary_M": salary_m,
        "Surplus_M": surplus_m,
        "AsymmetricUpsideIndex": (
            (stuff_plus - stuff_plus.mean()) / stuff_plus.std() - command_proxy
        ).round(2),
        "PitchMixInefficient": most_used != best,
        "MostUsedPitch": most_used,
        "BestPitch": best,
    })
    return df.sort_values("Surplus_M", ascending=False)


if __name__ == "__main__":
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} placeholder rows to {OUT_PATH}")
