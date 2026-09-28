"""
Volatility Discount Metric.

Computes outing-to-outing variance in release point and velocity, then
regresses next-outing performance on current (trailing) volatility to test
whether volatility actually predicts decline -- as opposed to just being
something the market appears to penalize.

Two paths, labeled in the VolatilityPath column: starters (8+ starts of 20+
pitches) are measured start to start; everyone else is measured appearance
to appearance (15+ appearances of 10+ pitches), so relievers get a real
score instead of defaulting to the 50th percentile downstream.

Small samples: pitchers who qualify for neither path but threw
config.MIN_PITCHES_TO_DISPLAY+ pitches get a score too, from whichever unit
they have more of, labeled Qualified = False (see assign_small_sample_path).
Their z-scores use the QUALIFIED path pool's mean/SD, so they don't move
anyone else's grade. VolatilityScore_Reg = (n * score + k * pool mean) / (n + k)
with n = appearances and k per path from analysis/stabilization.py;
VolatilityPercentile(_Reg) = share of the qualified path pool at or below it.

Graded league-wide (loaded one month at a time via load_season_monthly(),
since a single full-season pull OOMs on an 8GB machine); --team only
filters the printed output.

Also pulls in IL history from injury_history.py (run that first) to see if
volatility predicts going on the IL, not just a bad next start.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from config import END_DATE, MIN_PITCHES_TO_DISPLAY, SEASON, START_DATE
from analysis.data_acquisition import filter_to_team
from analysis.data_acquisition import load_season_monthly
from analysis.injury_history import STINTS_PATH, SUMMARY_PATH
from analysis.stabilization import load_k, regress, reliability
from analysis.utils import CACHE_DIR, pct_vs_pool, zscore_vs

OUT_PATH = CACHE_DIR / "volatility_discount.parquet"
APPEARANCES_PATH = CACHE_DIR / "volatility_appearances.parquet"   # one row per pitcher x game (cached)

# Only what this module uses -- keeps the league-wide 6-month frame small
# (see data_acquisition.load_season_monthly).
RAW_COLUMNS = [
    "pitcher", "home_team", "away_team", "inning_topbot",
    "release_speed", "release_spin_rate", "release_extension", "pfx_z", "pfx_x", "pitch_type",
    "release_pos_x", "release_pos_z", "delta_run_exp",
    "game_pk", "game_date", "at_bat_number", "pitch_number",
]

# Starter path: one unit = one start
MIN_PITCHES_PER_START = 20   # drops openers/short outings from the "start" unit
MIN_STARTS = 8                # need enough starts for variance to mean anything
# Reliever path: one unit = one appearance, for pitchers without MIN_STARTS starts
MIN_PITCHES_PER_RELIEF = 10
MIN_RELIEF_APPEARANCES = 15
MIN_UNITS_FOR_SCORE = 2       # small-sample path: fewest units a raw score (an SD) can be computed from
TRAILING_MIN_STARTS = 3       # min prior appearances before trailing volatility is computed
IL_WINDOW_DAYS = 30           # "went on the IL soon after this start" window


def build_appearance_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level data to one row per PitcherId x game (appearance).

    Started = this pitcher threw his team's first pitch of the game.
    """
    df = df.rename(columns={"pitcher": "PitcherId"})
    first_pitch = (
        df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .drop_duplicates(["game_pk", "PitcherTeam"])[["game_pk", "PitcherTeam", "PitcherId"]]
        .assign(Started=True)
    )
    apps = df.groupby(["PitcherId", "PitcherTeam", "game_pk", "game_date"]).agg(
        Pitches=("release_speed", "size"),
        MeanVelo=("release_speed", "mean"),
        MeanRelX=("release_pos_x", "mean"),
        MeanRelZ=("release_pos_z", "mean"),
        RunValue=("delta_run_exp", lambda s: -s.sum()),  # runs prevented; higher = better outing
    ).reset_index()
    apps = apps.merge(first_pitch, on=["game_pk", "PitcherTeam", "PitcherId"], how="left")
    apps["Started"] = apps["Started"].fillna(False).astype(bool)
    return apps.sort_values(["PitcherId", "game_date"])


def build_appearances(force_refresh: bool = False) -> pd.DataFrame:
    """League-wide appearance table (plus Pitcher name), cached -- also read by analysis/stabilization.py."""
    if APPEARANCES_PATH.exists() and not force_refresh:
        return pd.read_parquet(APPEARANCES_PATH)
    print(f"Loading league-wide Statcast {START_DATE} to {END_DATE} (month by month)...")
    df = load_season_monthly(START_DATE, END_DATE, columns=RAW_COLUMNS, force_refresh=force_refresh)
    name_by_id = df.drop_duplicates("pitcher").set_index("pitcher")["Pitcher"]
    apps = build_appearance_level(df)
    del df
    apps.insert(1, "Pitcher", apps["PitcherId"].map(name_by_id))
    apps.to_parquet(APPEARANCES_PATH, index=False)
    return apps


def primary_team_from_apps(apps: pd.DataFrame) -> pd.Series:
    """PitcherTeam each pitcher threw the most pitches for (same rule as data_acquisition.primary_team)."""
    counts = apps.groupby(["PitcherId", "PitcherTeam"])["Pitches"].sum().rename("n").reset_index()
    return counts.sort_values("n", ascending=False).drop_duplicates("PitcherId").set_index("PitcherId")["PitcherTeam"]


def assign_volatility_path(apps: pd.DataFrame) -> pd.DataFrame:
    """Pick each pitcher's unit of volatility and keep only those appearances.

    starter:  starts with MIN_PITCHES_PER_START+ pitches, needs MIN_STARTS of them
    reliever: everyone else -- appearances with MIN_PITCHES_PER_RELIEF+ pitches,
              needs MIN_RELIEF_APPEARANCES of them
    Pitchers who qualify for neither get no volatility score.
    """
    starts = apps[apps["Started"] & (apps["Pitches"] >= MIN_PITCHES_PER_START)]
    start_counts = starts.groupby("PitcherId").size()
    starters = set(start_counts[start_counts >= MIN_STARTS].index)

    relief = apps[~apps["PitcherId"].isin(starters) & (apps["Pitches"] >= MIN_PITCHES_PER_RELIEF)]
    relief_counts = relief.groupby("PitcherId").size()
    relievers = set(relief_counts[relief_counts >= MIN_RELIEF_APPEARANCES].index)

    units = pd.concat([
        starts[starts["PitcherId"].isin(starters)].assign(VolatilityPath="starter"),
        relief[relief["PitcherId"].isin(relievers)].assign(VolatilityPath="reliever"),
    ], ignore_index=True)
    return units.sort_values(["PitcherId", "game_date"])


def assign_small_sample_path(apps: pd.DataFrame, qualified_ids) -> pd.DataFrame:
    """Units for pitchers outside both qualified paths (but MIN_PITCHES_TO_DISPLAY+ pitches).

    Path = whichever unit they have more of (starts of MIN_PITCHES_PER_START+
    vs. relief appearances of MIN_PITCHES_PER_RELIEF+; ties go to reliever).
    If that path has fewer than MIN_UNITS_FOR_SCORE units, all of the
    pitcher's appearances with MIN_PITCHES_PER_RELIEF+ pitches are used
    instead (e.g. 1 start + 2 long relief outings). Pitchers still under
    MIN_UNITS_FOR_SCORE get no raw score, and their regressed score is the
    pool mean (reliability ~0) -- see compute_season_volatility.
    """
    apps = apps[~apps["PitcherId"].isin(qualified_ids)]
    total = apps.groupby("PitcherId")["Pitches"].sum()
    apps = apps[apps["PitcherId"].isin(total[total >= MIN_PITCHES_TO_DISPLAY].index)]
    starts = apps[apps["Started"] & (apps["Pitches"] >= MIN_PITCHES_PER_START)]
    relief = apps[~apps["Started"] & (apps["Pitches"] >= MIN_PITCHES_PER_RELIEF)]
    n_starts = starts.groupby("PitcherId").size().reindex(total.index, fill_value=0)
    n_relief = relief.groupby("PitcherId").size().reindex(total.index, fill_value=0)
    path = pd.Series(np.where(n_starts > n_relief, "starter", "reliever"), index=total.index)
    path_units = np.where(path == "starter", n_starts, n_relief)
    short = set(total.index[path_units < MIN_UNITS_FOR_SCORE])
    any_app = apps[apps["Pitches"] >= MIN_PITCHES_PER_RELIEF]
    units = pd.concat([
        starts[starts["PitcherId"].isin(path[path == "starter"].index.difference(short))],
        relief[relief["PitcherId"].isin(path[path == "reliever"].index.difference(short))],
        any_app[any_app["PitcherId"].isin(short)],
    ], ignore_index=True)
    units["VolatilityPath"] = units["PitcherId"].map(path)
    return units.sort_values(["PitcherId", "game_date"])


def compute_season_volatility(units: pd.DataFrame, team_by_id: pd.Series, small_units: pd.DataFrame | None = None) -> pd.DataFrame:
    """Season-long outing-to-outing volatility -- the number the dashboard shows.

    Z-scores/percentiles are computed within each path (starters vs. starters,
    relievers vs. relievers), league-wide: per-appearance means over ~15
    reliever pitches are noisier than over ~90 starter pitches, so pooling
    them would rank nearly every reliever as "volatile". The mean/SD and
    percentile pool are the QUALIFIED pitchers in `units` only;
    `small_units` pitchers are graded against that pool.
    """
    units = pd.concat([units.assign(Qualified=True), (small_units if small_units is not None else units.iloc[:0]).assign(Qualified=False)],
                      ignore_index=True)
    per_pitcher = units.groupby("PitcherId").agg(
        Qualified=("Qualified", "first"),
        VolatilityPath=("VolatilityPath", "first"),
        Appearances=("game_pk", "size"),
        VeloVolatility=("MeanVelo", "std"),
        RelXVolatility=("MeanRelX", "std"),
        RelZVolatility=("MeanRelZ", "std"),
    ).reset_index()
    per_pitcher.insert(1, "PitcherTeam", per_pitcher["PitcherId"].map(team_by_id))

    per_pitcher["ReleaseVolatility"] = np.sqrt(
        per_pitcher["RelXVolatility"] ** 2 + per_pitcher["RelZVolatility"] ** 2
    )
    for path, rows in per_pitcher.groupby("VolatilityPath"):
        pool = rows[rows["Qualified"]]
        score = (zscore_vs(rows["VeloVolatility"], pool["VeloVolatility"])
                 + zscore_vs(rows["ReleaseVolatility"], pool["ReleaseVolatility"]))
        pool_score = score[rows["Qualified"]]
        k = load_k(f"VolatilityScore[{path}]")
        # under MIN_UNITS_FOR_SCORE appearances there's no SD, so no raw score: regressed = pool mean
        reg = regress(score.fillna(pool_score.mean()), rows["Appearances"], k, pool_score.mean())
        per_pitcher.loc[rows.index, "VolatilityScore"] = score
        per_pitcher.loc[rows.index, "VolatilityScore_Reg"] = reg
        per_pitcher.loc[rows.index, "VolatilityReliability"] = reliability(rows["Appearances"], k).round(3)
        per_pitcher.loc[rows.index, "VolatilityPercentile"] = pct_vs_pool(score, pool_score).round(1)
        per_pitcher.loc[rows.index, "VolatilityPercentile_Reg"] = pct_vs_pool(reg, pool_score).round(1)
    return per_pitcher


def compute_trailing_volatility(units: pd.DataFrame) -> pd.DataFrame:
    """Expanding (prior-outings-only) volatility as of each outing, paired with the NEXT outing's result.

    Uses an expanding window rather than a fixed 30-day one so it works off
    outing count (consistent across pitchers regardless of missed time);
    swap in a date-based rolling window here if pitcher-specific rest gaps
    matter more than outing count.
    """
    units = units.sort_values(["PitcherId", "game_date"]).copy()
    grouped = units.groupby("PitcherId")

    units["TrailingVeloVolatility"] = grouped["MeanVelo"].transform(
        lambda s: s.expanding(min_periods=TRAILING_MIN_STARTS).std().shift(1)
    )
    rel_x_std = grouped["MeanRelX"].transform(lambda s: s.expanding(min_periods=TRAILING_MIN_STARTS).std().shift(1))
    rel_z_std = grouped["MeanRelZ"].transform(lambda s: s.expanding(min_periods=TRAILING_MIN_STARTS).std().shift(1))
    units["TrailingReleaseVolatility"] = np.sqrt(rel_x_std ** 2 + rel_z_std ** 2)

    units["NextRunValue"] = grouped["RunValue"].shift(-1)
    return units


def run_predictive_regression(starts_with_trailing: pd.DataFrame, label: str = "starter") -> LinearRegression | None:
    """Does trailing volatility predict the NEXT outing's run value?

    Pooled OLS across all pitcher-starts (not per-pitcher) -- exploratory,
    not causal: confirms whether the market's volatility penalty is
    justified by actual future performance, or is overreacting to noise.
    """
    reg_cols = ["TrailingVeloVolatility", "TrailingReleaseVolatility", "NextRunValue"]
    data = starts_with_trailing.dropna(subset=reg_cols)
    if len(data) < 10:
        print(f"\nToo few {label} outings with trailing volatility to fit ({len(data)})")
        return None

    X = data[["TrailingVeloVolatility", "TrailingReleaseVolatility"]]
    y = data["NextRunValue"]

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    print(f"\n── Volatility -> Next-Outing Performance Regression ({label} path) ──")
    print(f"n = {len(data)} pitcher-outings")
    print(f"R^2 = {r2:.4f}")
    print(f"coef TrailingVeloVolatility    = {model.coef_[0]:.4f}")
    print(f"coef TrailingReleaseVolatility = {model.coef_[1]:.4f}")
    print(
        "(negative coefficients would mean more trailing volatility -> worse "
        "next start, i.e. volatility is predictive, not just penalized)"
    )
    return model


def attach_injury_history(season_volatility: pd.DataFrame) -> pd.DataFrame:
    """Join IL history on PitcherId. NaN = not pulled yet, 0 = no IL time."""
    if not SUMMARY_PATH.exists():
        print(f"No {SUMMARY_PATH.name} yet -- run python -m analysis.injury_history --team XXX first")
        return season_volatility
    summary = pd.read_csv(SUMMARY_PATH)
    cols = ["PitcherId", "TotalDaysMissed", "DaysMissedLastSeason", "ArmILStints", "Had60DayIL"]
    return season_volatility.merge(summary[cols], on="PitcherId", how="left")


def add_il_outcomes(starts: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Flag starts followed by an IL stint within IL_WINDOW_DAYS.

    PriorILDays/PriorArmILStints only count seasons before SEASON. The summary
    totals include this season, so using them would include the stints we're
    trying to predict.
    """
    starts = starts.copy()
    stints = stints.copy()
    stints["StartDate"] = pd.to_datetime(stints["StartDate"])
    # a stint split across two seasons is still one IL trip
    placements = stints.drop_duplicates("StintId")

    covered = set(pd.read_csv(SUMMARY_PATH)["PitcherId"]) if SUMMARY_PATH.exists() else set()
    game_dates = pd.to_datetime(starts["game_date"])

    went_on_il = []
    for pid, date in zip(starts["PitcherId"], game_dates):
        if pid not in covered:
            went_on_il.append(np.nan)
            continue
        il_starts = placements.loc[placements["PitcherId"] == pid, "StartDate"]
        hit = ((il_starts > date) & (il_starts <= date + pd.Timedelta(days=IL_WINDOW_DAYS))).any()
        went_on_il.append(float(hit))
    starts[f"ILWithin{IL_WINDOW_DAYS}Days"] = went_on_il

    prior = stints[stints["Season"] < SEASON]
    prior_days = prior.groupby("PitcherId")["DaysMissed"].sum()
    prior_arm = prior[prior["IsArmInjury"]].drop_duplicates("StintId").groupby("PitcherId").size()
    in_covered = starts["PitcherId"].isin(covered)
    starts["PriorILDays"] = np.where(in_covered, starts["PitcherId"].map(prior_days).fillna(0), np.nan)
    starts["PriorArmILStints"] = np.where(in_covered, starts["PitcherId"].map(prior_arm).fillna(0), np.nan)
    return starts


def run_injury_regression(starts_with_outcomes: pd.DataFrame):
    """Logistic regression: trailing volatility -> IL stint in the next IL_WINDOW_DAYS.

    Prior IL history as controls. Exploratory, sample is small until more
    teams have injury data pulled.
    """
    target = f"ILWithin{IL_WINDOW_DAYS}Days"
    features = ["TrailingVeloVolatility", "TrailingReleaseVolatility", "PriorILDays", "PriorArmILStints"]
    data = starts_with_outcomes.dropna(subset=features + [target])

    print(f"\n── Volatility -> IL Within {IL_WINDOW_DAYS} Days (Logistic) ──")
    print(f"n = {len(data)} pitcher-starts, {int(data[target].sum())} followed by an IL stint")
    if data[target].nunique() < 2:
        print("Not enough IL outcomes to fit yet -- pull injury history for more teams.")
        return None

    X = (data[features] - data[features].mean()) / data[features].std()  # standardize so coefs are comparable
    model = LogisticRegression().fit(X, data[target])
    for name, coef in zip(features, model.coef_[0]):
        print(f"coef {name:<26} = {coef:.4f}")
    print("(positive volatility coefficients = more volatile pitchers are likelier to get hurt soon after)")
    return model


def run(team: str | None = None, force_refresh: bool = False):
    apps = build_appearances(force_refresh=force_refresh)
    team_by_id = primary_team_from_apps(apps)
    name_by_id = apps.drop_duplicates("PitcherId").set_index("PitcherId")["Pitcher"]

    units = assign_volatility_path(apps)
    small_units = assign_small_sample_path(apps, units["PitcherId"].unique())

    season_volatility = compute_season_volatility(units, team_by_id, small_units)
    season_volatility.insert(1, "Pitcher", season_volatility["PitcherId"].map(name_by_id))
    season_volatility = attach_injury_history(season_volatility)
    # Always save the league-wide table; --team only filters what's printed.
    season_volatility.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(season_volatility)} league-wide rows to {OUT_PATH}")
    print(season_volatility.groupby(["VolatilityPath", "Qualified"]).size().to_string())

    if team:
        shown = filter_to_team(season_volatility, team).sort_values("VolatilityPercentile", ascending=False)
        title = f"{team}, graded vs. league"
    else:
        shown = season_volatility.sort_values("VolatilityPercentile", ascending=False).head(20)
        title = "top 20 most volatile league-wide"
    print(f"\n── Volatility ({title}) ──")
    print(shown.round(2).to_string(index=False))

    trailing = compute_trailing_volatility(units)
    for path, path_units in trailing.groupby("VolatilityPath"):
        run_predictive_regression(path_units, label=path)

    if STINTS_PATH.exists():
        trailing = add_il_outcomes(trailing, pd.read_csv(STINTS_PATH))
        run_injury_regression(trailing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volatility Discount Metric.")
    parser.add_argument("--team", default=None, help="Only print this team's pitchers (still graded vs. the league), e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using a cached raw pull")
    args = parser.parse_args()
    run(team=args.team, force_refresh=args.refresh)
