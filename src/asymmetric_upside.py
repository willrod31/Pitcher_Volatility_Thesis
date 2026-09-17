"""
Asymmetric Upside Index.

Flags pitchers with elite Stuff+ Proxy but poor surface results explained by
command, not shape: Index = z(Stuff+ Proxy) - z(Command Proxy).

Also flags pitch-mix inefficiency: a pitcher whose best-performing pitch
(by Stuff+) isn't their most-thrown pitch is leaving upside on the table by
usage alone, independent of command.

Command Proxy note: Location+ is proprietary, and Statcast has no intended
target, so command can't be measured directly. This proxy instead scores
*zone discipline*: how often a pitcher works the edge of the zone (skill)
vs. the heart of the plate (mistake), plus walk rate. It's a stand-in for
command, not a measurement of it -- see README "Known limitations".
"""
import argparse

import numpy as np
import pandas as pd

from src.data_acquisition import filter_qualified, load_season
from src.stuff_plus_proxy import SUMMARY_PATH, pitcher_level_stuff_plus
from src.utils import CACHE_DIR, zscore

OUT_PATH = CACHE_DIR / "asymmetric_upside.parquet"

PLATE_HALF_WIDTH_FT = 17 / 2 / 12  # rulebook plate width, in feet, halved

EDGE_BAND = (0.8, 1.2)   # normalized zone-distance band counted as "edge"
MEATBALL_MAX = 0.4       # normalized zone-distance below this counted as "meatball"


def classify_zone_location(df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalized Chebyshev distance from the center of the zone.

    0 = dead center, 1 = rulebook zone boundary, >1 = outside the zone.
    Vertical bounds use each pitch's own sz_top/sz_bot (per-batter stance).
    """
    df = df.copy()
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"])

    sz_mid = (df["sz_top"] + df["sz_bot"]) / 2
    sz_half = (df["sz_top"] - df["sz_bot"]) / 2

    norm_x = df["plate_x"].abs() / PLATE_HALF_WIDTH_FT
    norm_z = (df["plate_z"] - sz_mid).abs() / sz_half

    df["ZoneDist"] = np.maximum(norm_x, norm_z)
    df["EdgeZone"] = df["ZoneDist"].between(*EDGE_BAND)
    df["Meatball"] = df["ZoneDist"] < MEATBALL_MAX
    return df


def compute_command_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher Command Proxy: edge-hitting and avoiding the middle of the plate, less walks."""
    located = classify_zone_location(df)

    per_pitcher = located.groupby("Pitcher").agg(
        Pitches=("ZoneDist", "size"),
        EdgePct=("EdgeZone", "mean"),
        MeatballPct=("Meatball", "mean"),
    )

    walks = df.groupby("Pitcher")["events"].apply(lambda s: (s == "walk").sum())
    batters_faced = df.groupby("Pitcher")["events"].apply(lambda s: s.notna().sum())
    per_pitcher["BBPct"] = (walks / batters_faced).reindex(per_pitcher.index)

    per_pitcher["CommandProxy"] = (
        zscore(per_pitcher["EdgePct"])
        - zscore(per_pitcher["MeatballPct"])
        - zscore(per_pitcher["BBPct"])
    )
    return per_pitcher.reset_index()


def compute_pitch_mix(df: pd.DataFrame, stuff_summary: pd.DataFrame) -> pd.DataFrame:
    """Best-performing pitch (by Stuff+) vs. most-thrown pitch, per pitcher."""
    usage = (
        df.groupby(["Pitcher", "pitch_type"]).size()
        .rename("Thrown").reset_index()
    )
    most_used = usage.loc[usage.groupby("Pitcher")["Thrown"].idxmax()][["Pitcher", "pitch_type"]]
    most_used = most_used.rename(columns={"pitch_type": "MostUsedPitch"})

    best = stuff_summary.loc[stuff_summary.groupby("Pitcher")["StuffPlus"].idxmax()][["Pitcher", "PitchType"]]
    best = best.rename(columns={"PitchType": "BestPitch"})

    mix = most_used.merge(best, on="Pitcher", how="inner")
    mix["PitchMixInefficient"] = mix["MostUsedPitch"] != mix["BestPitch"]
    return mix


def run(force_refresh: bool = False):
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found -- run `python -m src.stuff_plus_proxy` first."
        )
    stuff_summary = pd.read_parquet(SUMMARY_PATH)
    stuff_pitcher = pitcher_level_stuff_plus(stuff_summary)

    df = load_season(force_refresh=force_refresh)
    df = filter_qualified(df)

    command = compute_command_proxy(df)
    mix = compute_pitch_mix(df, stuff_summary)

    out = (
        stuff_pitcher[["Pitcher", "PitcherTeam", "StuffPlus"]]
        .merge(command[["Pitcher", "EdgePct", "MeatballPct", "BBPct", "CommandProxy"]], on="Pitcher", how="inner")
        .merge(mix[["Pitcher", "MostUsedPitch", "BestPitch", "PitchMixInefficient"]], on="Pitcher", how="left")
    )
    out["AsymmetricUpsideIndex"] = zscore(out["StuffPlus"]) - zscore(out["CommandProxy"])
    out = out.sort_values("AsymmetricUpsideIndex", ascending=False)

    out.to_parquet(OUT_PATH, index=False)

    print("── Top 20 Asymmetric Upside Candidates ──")
    print(out.head(20).round(2).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asymmetric Upside Index.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(force_refresh=args.refresh)
