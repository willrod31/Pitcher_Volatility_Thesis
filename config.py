SEASON = 2025
START_DATE = f"{SEASON}-04-01"
END_DATE = f"{SEASON}-09-30"
MIN_PITCHES_FOR_INCLUSION = 500
MIN_PITCHES_PER_TYPE_FOR_SCALE = 100  # pitcher x pitch type rows in the pool for the 100 + 10z scale

# Command measure used by asymmetric_upside.py / risk_adjusted_value.py:
# "location_plus" (analysis/location_plus_proxy.py) or "legacy" (z(Edge%) - z(Meatball%) - z(BB%))
COMMAND_METRIC = "location_plus"
DOLLARS_PER_WAR = 9_500_000

# Module 3: risk-adjusted value
REPLACEMENT_LEVEL_WAR = 0.0    # WAR a freely-available replacement pitcher provides
LEAGUE_AVG_STARTER_WAR = 2.0   # rough anchor for a qualified, average starter
WAR_PER_TALENT_Z = 1.5         # placeholder scaling: WAR swing per SD of talent -- see Module 3 docstring
VOLATILITY_SHRINKAGE = 0.6     # max fraction of upside pulled back toward replacement at 100th volatility pctile
# Injury history (analysis/injury_history.py): IL stints pulled for SEASON - N through SEASON
INJURY_LOOKBACK_SEASONS = 3

# Module 3: injury risk in the WAR shrinkage (ASSUMPTIONS, not fitted)
INJURY_SHRINKAGE = 0.4           # max fraction of upside pulled back at the maximum injury-risk score
INJURY_DAYS_FOR_MAX_RISK = 180   # IL days over the lookback window that count as maximum risk (~1 season)
INJURY_ARM_STINTS_FOR_MAX_RISK = 2  # arm IL stints over the lookback window that count as maximum risk

# Contracts / multi-year surplus (ASSUMPTIONS -- used by risk_adjusted_value.py)
LEAGUE_MIN_SALARY_BY_YEAR = {2025: 760_000, 2026: 780_000}  # later years = last known value
LEAGUE_MIN_SALARY = LEAGUE_MIN_SALARY_BY_YEAR[SEASON]
# Two-way players (MLBAM ids): pitching metrics are graded, but no surplus value is computed,
# because the salary pays for hitting AND pitching (660271 = Shohei Ohtani)
TWO_WAY_EXCLUDE_VALUE = [660271]
ARB_PCT_OF_MARKET = [0.40, 0.60, 0.80, 0.80]  # arb year 1, 2, 3, 4 (Super Two), as share of WAR x $/WAR
DOLLARS_PER_WAR_GROWTH = 0.05    # yearly inflation in $/WAR
DISCOUNT_RATE = 0.08             # future seasons are worth less than this one
MAX_CONTROL_YEARS = 6            # cap so one pitcher's long tail does not dominate

# Dashboard movement chart: arm angle fallback (APPROXIMATION) when Statcast has no arm_angle --
# degrees(atan2(release_pos_z - SHOULDER_HEIGHT_FT, |release_pos_x|))
SHOULDER_HEIGHT_FT = 5.0

# Small samples (analysis/stabilization.py): below MIN_PITCHES_FOR_INCLUSION a pitcher is still shown,
# but every metric is regressed toward the qualified league mean: (n * obs + k * mean) / (n + k)
MIN_PITCHES_TO_DISPLAY = 50        # pitchers under this show "n/a"
MIN_PITCHES_PER_TYPE_TO_DISPLAY = 15   # per pitch type Stuff+/Location+ rows under this show "n/a"
USE_REGRESSED = True               # downstream modules use the *_Reg columns (raw columns are always kept)
