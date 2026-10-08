# WAR calibration: summary of results

Everything ran end to end and nothing is committed. The calibrated projection is much better than the old formula and only modestly better than "same as last year". Stuff+ adds signal beyond a pitcher's results, but talent on its own does not beat results on its own.

## 2025 FIP-WAR top 20 (IP in decimal innings)

| # | Pitcher | IP | WAR |
|---|---|---|---|
| 1 | Tarik Skubal | 195.3 | 6.9 |
| 2 | Logan Webb | 207.0 | 6.9 |
| 3 | Cristopher Sánchez | 202.0 | 6.9 |
| 4 | Paul Skenes | 187.7 | 6.8 |
| 5 | Garrett Crochet | 205.3 | 6.1 |
| 6 | Jesús Luzardo | 183.7 | 5.4 |
| 7 | Max Fried | 195.3 | 5.3 |
| 8 | Yoshinobu Yamamoto | 173.7 | 5.0 |
| 9 | Hunter Brown | 185.3 | 4.9 |
| 10 | Framber Valdez | 192.0 | 4.5 |
| 11 | Kevin Gausman | 193.0 | 4.4 |
| 12 | Zack Wheeler | 149.7 | 4.2 |
| 13 | Sonny Gray | 180.7 | 4.2 |
| 14 | Bryan Woo | 186.7 | 4.1 |
| 15 | Chris Sale | 125.7 | 4.0 |
| 16 | Ranger Suarez | 157.3 | 4.0 |
| 17 | Nathan Eovaldi | 130.0 | 4.0 |
| 18 | Nick Pivetta | 181.7 | 4.0 |
| 19 | David Peterson | 168.7 | 3.7 |
| 20 | Carlos Rodón | 195.3 | 3.6 |

Each season's pitcher WAR totals 430. League values: 2023 lgERA 4.33 / lgRA9 4.69; 2024 4.08 / 4.46; 2025 4.16 / 4.52.

## Model comparison (fit on 2023->2024, tested on 2024->2025)

There were 706 rows per pair; about 23% of pitchers threw 0 innings the next season, and they stay in as 0 IP and 0 WAR.

| Model | Test RMSE | Test corr |
|---|---|---|
| R0 old formula | 0.952 | 0.18 |
| R1 results only | 0.896 | 0.32 |
| R2 talent only | 0.947 | 0.19 |
| **R3 talent + results** | **0.8946** | 0.32 |
| R4 R3 + volatility + injury | 0.8946 | 0.32 |
| Same as last year (rate) | 0.905 | 0.31 |
| Playing time vs. same innings as last year | 41.9 vs 46.9 IP | 0.63 vs 0.61 |
| **Next-season WAR: calibrated / old formula / same as last year** | **0.90 / 2.23 / 0.99** | 0.62 / 0.21 / 0.62 |

Rate RMSE is in runs per 9 innings, weighted by next-season IP. Correlation is unweighted Pearson.

- **Chosen model: R3**, by the rule set in advance. R4's RMSE was 0.894624 and R3's was 0.894554, so volatility and injury add nothing to the rate.
- **R2 does not beat R1** (0.947 vs 0.896), so this is not evidence that talent alone beats results.
- **R3 coefficients** (WLS on next-season FIPR9, fit on 2023->2024, n = 480):

| Term | Coef | SE | p |
|---|---|---|---|
| const | 5.2077 | 0.9575 | <0.001 |
| FIPR9 regressed | 0.3720 | 0.0906 | <0.001 |
| K-BB% | -1.4555 | 0.9315 | 0.119 |
| Stuff+ | -0.0185 | 0.0053 | **0.0005 (significant)** |
| Location+ | -0.0051 | 0.0057 | 0.371 (not significant) |
| Starter | 0.0847 | 0.0830 | 0.308 |

- The Stuff+ effect gets stronger (-0.031) when the one extreme-Stuff+ row is dropped.
- The playing-time model uses volatility and injury, so both hand-set shrinkages are set to 0.

## Projected 2026 WAR

| Pitcher | Role | Proj. IP | Proj. WAR | 2025 FIP-WAR |
|---|---|---|---|---|
| Tarik Skubal | starter | 142.5 | 3.48 | 6.92 |
| Garrett Crochet | starter | 151.1 | 3.40 | 6.07 |
| Paul Skenes | starter | 141.9 | 3.34 | 6.84 |
| Tyler Rogers | reliever | 49.1 | 1.75 | 1.45 |
| Jacob Misiorowski | starter | 70.9 | 1.37 | 1.29 |
| Grant Taylor | reliever | 35.8 | 0.58 | 1.34 |
| Mason Montgomery | reliever | 39.8 | 0.34 | 0.31 |

## Sanity checks

All three pass, with caveats:

1. **Correlation for qualified starters is 0.97** (162+ IP, n = 52).
2. **Highest reliever projection is 1.75 WAR** (Tyler Rogers), under the 3 WAR limit.
3. **Tyler Rogers is no longer the top surplus**, but partly because he is a free agent after 2025, not only because the model fixed him. His Stuff+ of 280 still gives him the lowest projected FIPR9 in baseball (1.80). That's a Stuff+ artifact from his submarine delivery; Bautista has it too. The pattern existed before this change, and it was not clipped.

Two more problems:

- **Aces are under-projected.** Out of sample, pitchers with 4+ WAR averaged 5.1 WAR, were projected at 2.8 and actually produced 3.6.

| WAR in season t | n | Projected next | Actual next |
|---|---|---|---|
| 0 or less | 424 | 0.11 | 0.14 |
| 0 to 1 | 660 | 0.35 | 0.32 |
| 1 to 2 | 181 | 0.93 | 0.80 |
| 2 to 3 | 72 | 1.67 | 1.86 |
| 3 to 4 | 44 | 2.19 | 2.21 |
| 4 or more | 31 | 2.80 | 3.57 |

- **The breakout flag doesn't work as a breakout detector.** Only 4 pitchers were flagged for 2023->2024 (3 hits, 75%; base rate 9.6%) and 9 for 2024->2025 (5 hits, 56%; base rate 10.9%). Nearly every flagged pitcher had negative WAR, so the flag mostly catches regression toward zero. Corey Kluber counts as a hit for throwing 0 innings.

## Top 25 by surplus from 2026 ($M)

All are starters. WAR is the 2026 projection and Ctrl is remaining control.

| # | Pitcher | Team | WAR | Ctrl | $M |
|---|---|---|---|---|---|
| 1 | Paul Skenes | PIT | 3.34 | 4 | 70.8 |
| 2 | Noah Cameron | KC | 1.50 | 6 | 57.7 |
| 3 | Bryan Woo | SEA | 2.67 | 4 | 56.4 |
| 4 | Will Warren | NYY | 1.84 | 5 | 55.1 |
| 5 | Jacob Misiorowski | MIL | 1.37 | 6 | 52.6 |
| 6 | Cade Horton | CHC | 1.58 | 6 | 49.7 |
| 7 | Garrett Crochet | BOS | 3.40 | 6 | 49.6 |
| 8 | Jack Leiter | TEX | 1.65 | 5 | 49.3 |
| 9 | Shane Smith | CWS | 1.61 | 5 | 47.9 |
| 10 | Chad Patrick | MIL | 1.51 | 6 | 47.7 |
| 11 | Brandon Pfaadt | AZ | 1.75 | 6 | 47.3 |
| 12 | Cam Schlittler | NYY | 1.23 | 6 | 47.0 |
| 13 | Nolan McLean | NYM | 1.23 | 6 | 46.7 |
| 14 | Andrew Abbott | CIN | 2.08 | 4 | 43.7 |
| 15 | Spencer Schwellenbach | ATL | 1.80 | 5 | 40.9 |
| 16 | Chase Burns | CIN | 1.07 | 6 | 40.7 |
| 17 | Hurston Waldrep | ATL | 1.04 | 6 | 39.4 |
| 18 | Janson Junk | MIA | 1.31 | 5 | 38.8 |
| 19 | Stephen Kolek | SD | 1.27 | 5 | 37.5 |
| 20 | Grant Holmes | ATL | 1.25 | 5 | 36.8 |
| 21 | Gavin Williams | CLE | 1.75 | 4 | 36.7 |
| 22 | Landen Roupp | SF | 1.21 | 5 | 35.9 |
| 23 | Cade Povich | BAL | 1.19 | 5 | 35.2 |
| 24 | Eury Pérez | MIA | 1.68 | 4 | 35.2 |
| 25 | Parker Messick | CLE | 0.93 | 6 | 34.8 |

237 pitchers are free agents after 2025; they have a blank surplus and are listed last.

## Dashboard

- Ran headless for PIT (29 pitchers) and LAD (34) and opened every pitcher's report: 0 errors.
- The Home page has 0 errors when run through `app.py`.
- On the PIT board the free agents sit at the bottom with the note.
- Mitch Keller, who was at -0.33 WAR under the old formula, now projects 1.8.

## Decisions you may want to change

- **Dashboard column names have no year in them.** The CSV uses `ActualWAR` and `SurplusActual_M`, and the dashboard adds the year from `config.SEASON`. The analysis output itself keeps `ActualWAR2025` and `SurplusActual2025`.
- **`run_all` now runs `fip_war` and `war_calibration` before the value step**, so a full 2025 run works without extra manual steps.
- **A stale README line was fixed.** It said the minimum was 50 pitches; config uses 100.
- **No outside sites were contacted beyond the two allowed.** Only the MLB Stats API and Statcast were used. Player names come from pybaseball's existing local copy of the Chadwick register, which is valid until Oct 14.
- **The 2026 season has already been played.** Real 2026 FIP-WAR could test these projections as a third pair; this wasn't done.

## Files

- **Changed:** `config.py`, `README.md`, `analysis/utils.py`, `analysis/run_all.py`, `analysis/stabilization.py`, `analysis/stuff_plus_proxy.py`, `analysis/location_plus_proxy.py`, `analysis/risk_adjusted_value.py`, `analysis/export_report_data.py`, `dashboard/pitchers.py`, `dashboard/home.py`, `dashboard/colors.py`, `dashboard/data/pitcher_report_data.csv`, `dashboard/data/surplus_by_year.csv`
- **New:** `analysis/fip_war.py`, `analysis/war_calibration.py`
- **`dashboard/data/` sizes:** everything is under 6.3 MB (`pitcher_report_data.csv` 0.25 MB, `surplus_by_year.csv` 0.25 MB, `stuff_plus_pitches.csv.gz` 6.28 MB is the largest).
- **Results that won't be committed:** `data/results/` is gitignored, so `war_calibration/` (1.6 MB) and `season_2023/` and `season_2024/` (43 MB each) stay local. To put the calibration CSVs in the repo, they'd need to be force-added.

```bash
git add config.py README.md analysis/utils.py analysis/run_all.py analysis/stabilization.py analysis/stuff_plus_proxy.py analysis/location_plus_proxy.py analysis/risk_adjusted_value.py analysis/export_report_data.py analysis/fip_war.py analysis/war_calibration.py dashboard/pitchers.py dashboard/home.py dashboard/colors.py dashboard/data/pitcher_report_data.csv dashboard/data/surplus_by_year.csv
```
