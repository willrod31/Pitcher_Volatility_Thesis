"""
Risk-Adjusted Value Projection.

Combines asymmetric_upside.py and volatility_discount.py (plus IL history from
injury_history.py) into a projected WAR proxy, shrunk toward replacement
level as volatility and injury risk increase. Ranks pitchers by multi-year
surplus value over their remaining years of team control.

simple_projected_war is an UNCALIBRATED PLACEHOLDER: it converts overall
talent (Stuff+ and command -- Location+ by default, see config.COMMAND_METRIC --
z-scored against the league and summed)
into WAR using a fixed WAR-per-standard-deviation constant
(config.WAR_PER_TALENT_Z), not a regression against actual historical WAR.
Treat it as a rank-ordering device, not a real WAR projection, until it's
regressed against a real training sample -- see README "Known limitations".

Contracts come from the hand-collected data/salaries.csv (joined on
PitcherId). Blank ContractStatus / YearsControl / PreArbYearsLeft /
ArbYearsLeft fall back to contract_status.py's debut-date estimate; blank
Salary for a pre-arb pitcher falls back to the league minimum. Hand-entered
values always win.

Multi-year surplus (project_control_years): one row per control year, capped
at config.MAX_CONTROL_YEARS, with ProjectedWAR held FLAT at this season's
risk_adjusted_war (no aging curve yet). Salary per year: actual Salary this
season; GuaranteedFuture when listed; league minimum for pre-arb years;
max(prior salary, ARB_PCT_OF_MARKET x WAR x $/WAR) for arb years. Surplus =
WAR x $/WAR - Salary, discounted by (1 + DISCOUNT_RATE) ** years_from_now.
All of those rates are assumptions in config.py.
"""
import argparse

import numpy as np
import pandas as pd

from config import (
    ARB_PCT_OF_MARKET,
    DISCOUNT_RATE,
    DOLLARS_PER_WAR,
    DOLLARS_PER_WAR_GROWTH,
    INJURY_ARM_STINTS_FOR_MAX_RISK,
    INJURY_DAYS_FOR_MAX_RISK,
    INJURY_SHRINKAGE,
    LEAGUE_AVG_STARTER_WAR,
    LEAGUE_MIN_SALARY_BY_YEAR,
    MAX_CONTROL_YEARS,
    REPLACEMENT_LEVEL_WAR,
    SEASON,
    VOLATILITY_SHRINKAGE,
    WAR_PER_TALENT_Z,
)
from analysis.asymmetric_upside import COMMAND_COLUMN, OUT_PATH as UPSIDE_PATH
from analysis.data_acquisition import filter_to_team, team_roster_ids
from analysis.contract_status import SALARIES_PATH, estimate_contract_status, save_estimates, status_group
from analysis.injury_history import SUMMARY_PATH as INJURY_SUMMARY_PATH
from analysis.volatility_discount import OUT_PATH as VOLATILITY_PATH
from analysis.utils import CACHE_DIR, zscore

OUT_PATH = CACHE_DIR / "risk_adjusted_value.parquet"
SURPLUS_BY_YEAR_PATH = CACHE_DIR / "surplus_by_year.csv"

CONTRACT_FIELDS = ["ContractStatus", "YearsControl", "PreArbYearsLeft", "ArbYearsLeft"]


def league_min_salary(year: int) -> int:
    """LEAGUE_MIN_SALARY_BY_YEAR, carrying the last known value forward."""
    known = [y for y in LEAGUE_MIN_SALARY_BY_YEAR if y <= year]
    return LEAGUE_MIN_SALARY_BY_YEAR[max(known) if known else min(LEAGUE_MIN_SALARY_BY_YEAR)]


def load_salaries() -> pd.DataFrame | None:
    if not SALARIES_PATH.exists():
        print(f"No {SALARIES_PATH} found -- only pre-arb estimates (league minimum) will get a Salary.")
        return None
    salaries = pd.read_csv(SALARIES_PATH)
    if "Season" in salaries:
        salaries = salaries[salaries["Season"].isna() | (salaries["Season"] == SEASON)]
    return salaries.drop(columns=["Pitcher", "Season"], errors="ignore").drop_duplicates("PitcherId", keep="last")


def load_injury_summary() -> pd.DataFrame | None:
    if not INJURY_SUMMARY_PATH.exists():
        print(f"No {INJURY_SUMMARY_PATH.name} yet -- run python -m analysis.injury_history --team XXX first")
        return None
    return pd.read_csv(INJURY_SUMMARY_PATH)[["PitcherId", "TotalDaysMissed", "ArmILStints"]]


def compute_projected_war(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    # command = Location+ or the legacy proxy, per config.COMMAND_METRIC
    talent_z = zscore(merged["StuffPlus"]) + zscore(merged[COMMAND_COLUMN])
    merged["simple_projected_war"] = LEAGUE_AVG_STARTER_WAR + WAR_PER_TALENT_Z * talent_z
    return merged


def injury_risk_score(merged: pd.DataFrame) -> pd.Series:
    """0 (no IL history) to 1 (maximum), half IL days and half arm IL stints over the lookback.

    Absolute scale (config.INJURY_*_FOR_MAX_RISK), not a percentile, so the
    score doesn't depend on which teams have had injury history pulled.
    """
    days = (merged["TotalDaysMissed"] / INJURY_DAYS_FOR_MAX_RISK).clip(upper=1)
    arm = (merged["ArmILStints"] / INJURY_ARM_STINTS_FOR_MAX_RISK).clip(upper=1)
    return 0.5 * days + 0.5 * arm


def apply_risk_shrinkage(merged: pd.DataFrame) -> pd.DataFrame:
    """Pull projected WAR back toward replacement level as volatility and injury risk rise.

    Volatility: 0th percentile keeps full upside, 100th loses
    VOLATILITY_SHRINKAGE of it. Injury: a risk score of 1 loses
    INJURY_SHRINKAGE of it. The two combine multiplicatively so the total
    never pulls a pitcher past replacement level. Pitchers with no volatility
    path default to the 50th percentile; pitchers with no injury history
    pulled get 0 injury risk (InjuryDataPulled = False).
    """
    merged = merged.copy()
    merged["VolatilityPercentileUsed"] = merged["VolatilityPercentile"].fillna(50)
    merged["InjuryDataPulled"] = merged["TotalDaysMissed"].notna()
    merged["InjuryRiskScore"] = injury_risk_score(merged).fillna(0)

    keep = (
        (1 - VOLATILITY_SHRINKAGE * merged["VolatilityPercentileUsed"] / 100)
        * (1 - INJURY_SHRINKAGE * merged["InjuryRiskScore"])
    )
    upside_over_replacement = merged["simple_projected_war"] - REPLACEMENT_LEVEL_WAR
    merged["risk_adjusted_war"] = REPLACEMENT_LEVEL_WAR + upside_over_replacement * keep
    return merged


def resolve_contracts(merged: pd.DataFrame, salaries: pd.DataFrame | None) -> pd.DataFrame:
    """Hand-entered salaries.csv values, falling back field-by-field to the debut-date estimate."""
    est = estimate_contract_status(merged[["PitcherId", "Pitcher"]])
    save_estimates(est)
    merged = merged.merge(est.drop(columns="Pitcher"), on="PitcherId", how="left")

    merged["InSalariesCsv"] = merged["PitcherId"].isin(salaries["PitcherId"]) if salaries is not None else False
    if salaries is not None:
        merged = merged.merge(salaries, on="PitcherId", how="left")
    for col in ["Salary", "AAV", "GuaranteedFuture", *CONTRACT_FIELDS]:
        if col not in merged:
            merged[col] = np.nan

    merged["ContractStatusHand"] = merged["ContractStatus"]
    hand_group = merged["ContractStatus"].map(status_group)
    merged["StatusMismatch"] = hand_group.notna() & merged["ContractStatus_est"].notna() & (hand_group != merged["ContractStatus_est"])
    merged["ContractStatus"] = hand_group.fillna(merged["ContractStatus_est"])

    merged["ControlSource"] = np.where(merged["YearsControl"].notna(), "hand", "estimate")
    merged["YearsControl"] = merged["YearsControl"].fillna(merged["EstYearsControl"])
    merged["PreArbYearsLeft"] = merged["PreArbYearsLeft"].fillna(merged["EstPreArbYearsLeft"])
    merged["ArbYearsLeft"] = merged["ArbYearsLeft"].fillna(merged["EstArbYearsLeft"])
    # a pitcher on this season's roster is under control for at least this season --
    # but if nothing says how long, flag it rather than silently assuming 1 year
    no_control = merged["YearsControl"].isna() | (merged["YearsControl"] < 1)
    merged.loc[no_control, "ControlSource"] = "missing (1 yr assumed)"
    merged.loc[no_control, "YearsControl"] = 1
    for col in ["YearsControl", "PreArbYearsLeft", "ArbYearsLeft"]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["SalarySource"] = np.where(merged["Salary"].notna(), "hand", None)
    fill_min = merged["Salary"].isna() & (merged["ContractStatus"] == "pre-arb")
    merged.loc[fill_min, "Salary"] = league_min_salary(SEASON)
    merged.loc[fill_min, "SalarySource"] = "league min"
    return merged


def parse_guaranteed(value) -> dict[int, float]:
    """'2026:16911500;2027:18411500' -> {2026: 16911500.0, 2027: 18411500.0}"""
    if pd.isna(value) or not str(value).strip():
        return {}
    out = {}
    for part in str(value).split(";"):
        if ":" in part:
            year, amount = part.split(":", 1)
            out[int(year.strip())] = float(amount.strip())
    return out


def project_control_years(row) -> pd.DataFrame:
    """One row per remaining control year (this season = year 0), capped at MAX_CONTROL_YEARS."""
    years = min(int(row["YearsControl"]), MAX_CONTROL_YEARS)
    pre_arb_left = int(row["PreArbYearsLeft"])
    arb_left = int(row["ArbYearsLeft"])
    # which arb year (0-based into ARB_PCT_OF_MARKET) the first remaining arb season is:
    # 4 left = Super Two year 1; otherwise assume 3 total arb years
    first_arb_idx = max(3, arb_left) - arb_left
    guaranteed = parse_guaranteed(row["GuaranteedFuture"])
    war = row["risk_adjusted_war"]

    rows = []
    prior_salary = np.nan
    for t in range(years):
        season = SEASON + t
        dollars_per_war = DOLLARS_PER_WAR * (1 + DOLLARS_PER_WAR_GROWTH) ** t
        market_value = war * dollars_per_war
        if t < pre_arb_left:
            stage = "pre-arb"
        elif t < pre_arb_left + arb_left:
            stage = "arb"
        else:
            stage = "contract"

        if t == 0:
            salary, source = row["Salary"], row["SalarySource"]
        elif season in guaranteed:
            salary, source = guaranteed[season], "guaranteed"
        elif stage == "pre-arb":
            salary, source = league_min_salary(season), "league min"
        elif stage == "arb":
            arb_idx = min(first_arb_idx + (t - pre_arb_left), len(ARB_PCT_OF_MARKET) - 1)
            salary = np.nanmax([prior_salary, ARB_PCT_OF_MARKET[arb_idx] * market_value])
            source = f"arb est (yr {arb_idx + 1})"
        else:
            # beyond the listed schedule with no guaranteed figure: carry AAV/current salary
            salary = row["AAV"] if pd.notna(row["AAV"]) else prior_salary
            source = "AAV" if pd.notna(row["AAV"]) else "prior salary"

        discount = (1 + DISCOUNT_RATE) ** t
        surplus = market_value - salary
        rows.append({
            "PitcherId": row["PitcherId"], "Pitcher": row["Pitcher"], "Season": season, "YearIndex": t,
            "Stage": stage, "ProjectedWAR": war, "DollarsPerWAR": dollars_per_war,
            "Salary": salary, "SalarySource": source, "Surplus": surplus,
            "DiscountFactor": discount, "DiscountedSurplus": surplus / discount,
        })
        prior_salary = salary
    return pd.DataFrame(rows)


def add_multi_year_surplus(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds SurplusCurrentSeason / MultiYearSurplus / ControlYearsUsed / SurplusPerControlYear."""
    merged = merged.copy()
    merged["SurplusCurrentSeason"] = merged["risk_adjusted_war"] * DOLLARS_PER_WAR - merged["Salary"]

    has_salary = merged[merged["Salary"].notna()]
    by_year = pd.concat([project_control_years(r) for _, r in has_salary.iterrows()], ignore_index=True) \
        if not has_salary.empty else pd.DataFrame()

    if by_year.empty:
        merged["MultiYearSurplus"] = np.nan
        merged["ControlYearsUsed"] = np.nan
    else:
        totals = by_year.groupby("PitcherId").agg(
            MultiYearSurplus=("DiscountedSurplus", "sum"), ControlYearsUsed=("YearIndex", "size"),
        )
        merged = merged.merge(totals, on="PitcherId", how="left")
    merged["SurplusPerControlYear"] = merged["MultiYearSurplus"] / merged["ControlYearsUsed"]
    return merged, by_year


def report_flags(scope: pd.DataFrame):
    """Print pitchers whose contract data is estimated, missing, or disagrees with the estimate."""
    mismatch = scope[scope["StatusMismatch"]]
    print(f"\n{len(mismatch)} pitcher(s) where the estimate disagrees with the hand-entered ContractStatus:")
    if not mismatch.empty:
        print(mismatch[["Pitcher", "ContractStatusHand", "ContractStatus_est", "EstServiceYears", "MLBDebutDate"]].to_string(index=False))

    not_hand = scope[scope["ControlSource"] != "hand"]
    print(f"\n{len(not_hand)} pitcher(s) with blank YearsControl in salaries.csv (using estimate or 1 year):")
    if not not_hand.empty:
        print(not_hand[["Pitcher", "ContractStatus", "YearsControl", "ControlSource", "EstYearsControl"]].to_string(index=False))

    no_salary = scope[scope["Salary"].isna()]
    if not no_salary.empty:
        print(f"\n{len(no_salary)} pitcher(s) with no Salary (not pre-arb, not in salaries.csv) -- no surplus computed:")
        print(no_salary[["Pitcher", "ContractStatus"]].to_string(index=False))


def run(team: str | None = None):
    for path, module in [(UPSIDE_PATH, "asymmetric_upside"), (VOLATILITY_PATH, "volatility_discount")]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python -m analysis.{module}` first.")

    upside = pd.read_parquet(UPSIDE_PATH)
    volatility = pd.read_parquet(VOLATILITY_PATH)

    merged = upside.merge(
        volatility[["PitcherId", "VolatilityPath", "VolatilityPercentile"]], on="PitcherId", how="left"
    )
    merged["VolatilityPath"] = merged["VolatilityPath"].fillna("none (50th pctile default)")

    injury = load_injury_summary()
    merged = merged.merge(injury, on="PitcherId", how="left") if injury is not None \
        else merged.assign(TotalDaysMissed=np.nan, ArmILStints=np.nan)

    # graded against the whole league before any --team filtering
    merged = compute_projected_war(merged)
    merged = apply_risk_shrinkage(merged)

    merged = resolve_contracts(merged, load_salaries())
    merged, by_year = add_multi_year_surplus(merged)

    merged = merged.sort_values("MultiYearSurplus", ascending=False, na_position="last")
    merged.to_parquet(OUT_PATH, index=False)
    by_year.to_csv(SURPLUS_BY_YEAR_PATH, index=False)
    print(f"Saved {len(merged)} league-wide rows to {OUT_PATH}")
    print(f"Saved {len(by_year)} pitcher-season rows to {SURPLUS_BY_YEAR_PATH}")

    scope = filter_to_team(merged, team) if team else merged[merged["MultiYearSurplus"].notna()].head(20)
    cols = [
        "Pitcher", "PitcherTeam", "StuffPlus_Scaled", COMMAND_COLUMN, "VolatilityPath", "VolatilityPercentile",
        "InjuryRiskScore", "simple_projected_war", "risk_adjusted_war", "ContractStatus", "Salary",
        "YearsControl", "ControlYearsUsed", "SurplusCurrentSeason", "MultiYearSurplus", "SurplusPerControlYear",
    ]
    shown = scope[cols].copy()
    money = ["Salary", "SurplusCurrentSeason", "MultiYearSurplus", "SurplusPerControlYear"]
    shown[money] = shown[money] / 1_000_000
    print(f"\n── {'Ranking: ' + team if team else 'Top 20'} by MultiYearSurplus ($M, graded vs. league) ──")
    print(shown.round(2).to_string(index=False))

    report_flags(scope if team else merged[merged["InSalariesCsv"]])
    if team:
        report_ungraded(team, merged)


def report_ungraded(team: str, merged: pd.DataFrame):
    """Roster pitchers missing from the ranking (under MIN_PITCHES_FOR_INCLUSION), incl. blank control in salaries.csv."""
    roster = team_roster_ids(team)
    missing = roster[~roster["PitcherId"].isin(merged["PitcherId"])]
    print(f"\n{len(missing)} {team} roster pitcher(s) not graded (under config.MIN_PITCHES_FOR_INCLUSION pitches):")
    print(", ".join(missing["Pitcher"]) or "none")
    salaries = load_salaries()
    if salaries is not None:
        blank = missing.merge(salaries, on="PitcherId")
        blank = blank[blank["YearsControl"].isna()]
        if not blank.empty:
            print(f"  ...of these, blank YearsControl in salaries.csv: {', '.join(blank['Pitcher'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk-Adjusted Value Projection.")
    parser.add_argument("--team", default=None, help="Only print this team's pitchers (still graded vs. the league), e.g. --team PIT")
    args = parser.parse_args()
    run(team=args.team)
