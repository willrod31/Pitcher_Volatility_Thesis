"""
Asymmetric Upside Index.

Flags pitchers with elite Stuff+ but poor results explained by command, not
shape: Index = z(Stuff+) - z(Location+) by default (config.COMMAND_METRIC =
"location_plus"; run analysis/location_plus_proxy.py first). Location+ uses
only location and count, Stuff+ only pitch shape, so the two don't overlap.
config.COMMAND_METRIC = "legacy" switches back to the original Command Proxy,
which is always kept as CommandProxyLegacy for comparison.

Also flags pitch-mix inefficiency: a pitcher whose best-performing pitch
(by Stuff+) isn't their most-thrown pitch is leaving upside on the table by
usage alone, independent of command.

Legacy Command Proxy note: Location+ (the proprietary one) is proprietary, and Statcast has no intended
target, so command can't be measured directly. This proxy instead scores
*zone discipline*: how often a pitcher works the edge of the zone (skill)
vs. the heart of the plate (mistake), plus walk rate. It's a stand-in for
command, not a measurement of it -- see README "Known limitations".

Grading is always league-wide: Stuff+, Command Proxy and the index are
z-scored against every qualified MLB pitcher, pulled one month at a time via
load_season_monthly() (a single full-season pull OOMs on an 8GB machine).
--team only filters the printed output afterwards, so a Pirates pitcher is
compared to the league, not to the rest of the Pirates staff.
"""
import argparse

import numpy as np
import pandas as pd

from analysis.data_acquisition import filter_qualified, filter_to_team, load_season_monthly, primary_team
from analysis.location_plus_proxy import PITCHER_PATH as LOCATION_PITCHER_PATH
from analysis.stuff_plus_proxy import (
    RAW_TRAIN_COLUMNS,
    add_scaled_stuff_plus,
    fit_stuff_plus_models,
    league_scale_pool,
    score_stuff_plus,
    summarize_stuff_plus,
)
from analysis.utils import CACHE_DIR, zscore
from config import COMMAND_METRIC, END_DATE, START_DATE

if COMMAND_METRIC not in ("location_plus", "legacy"):
    raise ValueError(f'config.COMMAND_METRIC must be "location_plus" or "legacy", not {COMMAND_METRIC!r}')
# The command column every downstream module uses (higher = better command either way).
COMMAND_COLUMN = "LocationPlus" if COMMAND_METRIC == "location_plus" else "CommandProxyLegacy"

OUT_PATH = CACHE_DIR / "asymmetric_upside.parquet"
# League-wide per-pitch-type Stuff+ (same model/training data as
# stuff_plus_proxy.py, scored for every pitcher instead of one team).
LEAGUE_STUFF_PATH = CACHE_DIR / "stuff_plus_league_summary.parquet"

# Stuff+ training columns plus what the Command Proxy needs.
RAW_COLUMNS = RAW_TRAIN_COLUMNS + ["plate_x", "plate_z", "sz_top", "sz_bot", "events"]

PLATE_HALF_WIDTH_FT = 17 / 2 / 12  # rulebook plate width, in feet, halved

EDGE_BAND = (0.8, 1.2)   # normalized zone-distance band counted as "edge"
MEATBALL_MAX = 0.4       # normalized zone-distance below this counted as "meatball"


def classify_zone_location(df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalized Chebyshev distance from the center of the zone.

    0 = dead center, 1 = rulebook zone boundary, >1 = outside the zone.
    Vertical bounds use each pitch's own sz_top/sz_bot (per-batter stance).
    """
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"])

    sz_mid = (df["sz_top"] + df["sz_bot"]) / 2
    sz_half = (df["sz_top"] - df["sz_bot"]) / 2

    norm_x = df["plate_x"].abs() / PLATE_HALF_WIDTH_FT
    norm_z = (df["plate_z"] - sz_mid).abs() / sz_half

    zone_dist = np.maximum(norm_x, norm_z)
    return pd.DataFrame({
        "PitcherId": df["pitcher"],
        "ZoneDist": zone_dist,
        "EdgeZone": zone_dist.between(*EDGE_BAND),
        "Meatball": zone_dist < MEATBALL_MAX,
    })


def compute_command_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher Command Proxy: edge-hitting and avoiding the middle of the plate, less walks."""
    located = classify_zone_location(df)

    per_pitcher = located.groupby("PitcherId").agg(
        Pitches=("ZoneDist", "size"),
        EdgePct=("EdgeZone", "mean"),
        MeatballPct=("Meatball", "mean"),
    )

    events = df[["pitcher", "events"]].rename(columns={"pitcher": "PitcherId"})
    walks = events.assign(BB=events["events"] == "walk").groupby("PitcherId")["BB"].sum()
    batters_faced = events.groupby("PitcherId")["events"].count()
    per_pitcher["BBPct"] = (walks / batters_faced).reindex(per_pitcher.index)

    per_pitcher["CommandProxy"] = (
        zscore(per_pitcher["EdgePct"])
        - zscore(per_pitcher["MeatballPct"])
        - zscore(per_pitcher["BBPct"])
    )
    return per_pitcher.reset_index()


def league_stuff_plus(df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Per-pitch-type Stuff+ for every pitcher in df, cached.

    Uses stuff_plus_proxy.py's own fit/score functions on the same full
    season it trains on, so the numbers match what --team/--pitcher produce
    for the dashboard -- just scored for the whole league at once.
    """
    if LEAGUE_STUFF_PATH.exists() and not force_refresh:
        return pd.read_parquet(LEAGUE_STUFF_PATH)

    models = fit_stuff_plus_models(df)
    # One row per pitcher (not per pitcher-team) so traded pitchers aren't split.
    scoring = df.assign(PitcherTeam=df["pitcher"].map(primary_team(df)))
    summary = summarize_stuff_plus(score_stuff_plus(scoring, models))
    summary.to_parquet(LEAGUE_STUFF_PATH, index=False)
    return summary


def pitcher_stuff_plus(stuff_summary: pd.DataFrame) -> pd.DataFrame:
    """Pitch-count-weighted overall Stuff+ per PitcherId."""
    weighted = stuff_summary.assign(Weighted=stuff_summary["StuffPlus"] * stuff_summary["Pitches"])
    out = weighted.groupby("PitcherId").agg(
        Pitcher=("Pitcher", "first"),
        PitcherTeam=("PitcherTeam", "first"),
        WeightedSum=("Weighted", "sum"),
        StuffPitches=("Pitches", "sum"),
    ).reset_index()
    out["StuffPlus"] = (out["WeightedSum"] / out["StuffPitches"]).round(1)
    return out.drop(columns="WeightedSum")


def compute_pitch_mix(df: pd.DataFrame, stuff_summary: pd.DataFrame) -> pd.DataFrame:
    """Best-performing pitch (by Stuff+) vs. most-thrown pitch, per pitcher."""
    usage = (
        df.groupby(["pitcher", "pitch_type"]).size()
        .rename("Thrown").reset_index()
        .rename(columns={"pitcher": "PitcherId"})
    )
    most_used = usage.loc[usage.groupby("PitcherId")["Thrown"].idxmax()][["PitcherId", "pitch_type"]]
    most_used = most_used.rename(columns={"pitch_type": "MostUsedPitch"})

    best = stuff_summary.loc[stuff_summary.groupby("PitcherId")["StuffPlus"].idxmax()][["PitcherId", "PitchType"]]
    best = best.rename(columns={"PitchType": "BestPitch"})

    mix = most_used.merge(best, on="PitcherId", how="inner")
    mix["PitchMixInefficient"] = mix["MostUsedPitch"] != mix["BestPitch"]
    return mix


def run(team: str | None = None, force_refresh: bool = False):
    if not LOCATION_PITCHER_PATH.exists():
        raise FileNotFoundError(f"{LOCATION_PITCHER_PATH} not found -- run `python -m analysis.location_plus_proxy` first.")
    location = pd.read_parquet(LOCATION_PITCHER_PATH)[["PitcherId", "LocationPlus"]]

    print(f"Loading league-wide Statcast {START_DATE} to {END_DATE} (month by month)...")
    df = load_season_monthly(START_DATE, END_DATE, columns=RAW_COLUMNS, force_refresh=force_refresh)
    # Stuff+ is fit on the unfiltered pool, exactly like stuff_plus_proxy.py.
    stuff_summary = league_stuff_plus(df, force_refresh=force_refresh)
    pitcher_pool, type_pool = league_scale_pool(df, stuff_summary)

    df = filter_qualified(df)
    print(f"{df['pitcher'].nunique()} qualified pitchers league-wide")
    stuff_pitcher = pitcher_stuff_plus(stuff_summary)
    _, stuff_pitcher = add_scaled_stuff_plus(stuff_summary, stuff_pitcher, pitcher_pool, type_pool)
    command = compute_command_proxy(df)
    mix = compute_pitch_mix(df, stuff_summary)

    out = (
        stuff_pitcher[["PitcherId", "Pitcher", "PitcherTeam", "StuffPlus", "StuffPlus_Scaled"]]
        .merge(location, on="PitcherId", how="left")
        .merge(command[["PitcherId", "EdgePct", "MeatballPct", "BBPct", "CommandProxy"]]
               .rename(columns={"CommandProxy": "CommandProxyLegacy"}), on="PitcherId", how="inner")
        .merge(mix[["PitcherId", "MostUsedPitch", "BestPitch", "PitchMixInefficient"]], on="PitcherId", how="left")
    )
    out["CommandMetric"] = COMMAND_METRIC
    out["AsymmetricUpsideIndex"] = zscore(out["StuffPlus"]) - zscore(out[COMMAND_COLUMN])
    out = out.sort_values("AsymmetricUpsideIndex", ascending=False)

    # Always save the league-wide table -- downstream z-scores need the full pool.
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(out)} league-wide rows to {OUT_PATH}")

    shown = filter_to_team(out, team) if team else out.head(20)
    print(f"\n── Asymmetric Upside = z(StuffPlus) - z({COMMAND_COLUMN}) ({team + ', graded vs. league' if team else 'top 20 league-wide'}) ──")
    print(shown.drop(columns="PitcherId").round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asymmetric Upside Index.")
    parser.add_argument("--team", default=None, help="Only print this team's pitchers (still graded vs. the league), e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Force re-download / re-score instead of using cached pulls")
    args = parser.parse_args()
    run(team=args.team, force_refresh=args.refresh)
