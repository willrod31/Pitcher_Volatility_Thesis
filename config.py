SEASON = 2025
START_DATE = f"{SEASON}-04-01"
END_DATE = f"{SEASON}-09-30"
MIN_PITCHES_FOR_INCLUSION = 500
DOLLARS_PER_WAR = 9_500_000

# Module 3: risk-adjusted value
REPLACEMENT_LEVEL_WAR = 0.0    # WAR a freely-available replacement pitcher provides
LEAGUE_AVG_STARTER_WAR = 2.0   # rough anchor for a qualified, average starter
WAR_PER_TALENT_Z = 1.5         # placeholder scaling: WAR swing per SD of talent -- see Module 3 docstring
VOLATILITY_SHRINKAGE = 0.6     # max fraction of upside pulled back toward replacement at 100th volatility pctile