# Valuing Pitcher Volatility

**Finding undervalued pitchers for low-payroll MLB teams**

## The question

Can public Statcast data identify pitchers with elite pitch quality (Stuff+) whose results are held back by fixable problems like command or pitch mix? If so, small-market teams could acquire them at a discount.

Large-market teams pay a premium for proven, consistent starters. Small-market teams can't afford that premium, so they need to find talent the market is mispricing.

## What this project does

1. Grades every MLB pitcher's **stuff** (pitch shape) and **command** (pitch location) separately.
2. Flags pitchers with great stuff but weaker command, since command is more fixable than stuff.
3. Measures **volatility** (start-to-start inconsistency) and **injury history**.
4. Estimates each pitcher's **dollar value** over his remaining years of team control, adjusted for that risk.
5. Shows it all in a **Streamlit dashboard**: pick a team, pick a pitcher, get a report.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m analysis.run_all          # run the analysis for all 30 teams
streamlit run dashboard/app.py      # open the dashboard
```

## The models

| Model | What it measures | Inputs |
|---|---|---|
| **Stuff+** | How hard a pitch is to hit | Velocity, movement, spin, extension, release point |
| **Location+** | Command | Pitch location and count only |
| **Upside Index** | Great stuff, weaker command | Stuff+ minus Location+ (as z-scores) |
| **Volatility** | Outing-to-outing inconsistency | Changes in velocity and release point |
| **Value** | Surplus dollars over years of control | Projected WAR, salary, injury and volatility risk |

**Stuff+ and Location+** are proprietary stats, so this project builds its own versions. Both use a 100 = average scale where 10 points = one standard deviation. They share no inputs, which lets the Upside Index separate stuff from command.

**Fair grading:** each pitcher is scored by a model that never saw his pitches (out-of-fold scoring), and every pitcher is compared to the whole league, not just his own team.

## Handling small samples

A reliever with 80 pitches shouldn't look as certain as a starter with 3,000. Each stat is pulled toward the league average based on how reliable it is at that sample size:

```
Adjusted value = (n x observed + k x league average) / (n + k)
```

`n` is the pitcher's sample size, and `k` is the sample size where that stat becomes 50% reliable. Stuff+ becomes reliable almost immediately (k = 1 pitch), Location+ takes about 286 pitches, and walk rate about 241 batters.

League averages come only from pitchers with 500+ pitches. Pitchers with fewer than 50 pitches aren't shown.

## Contracts

Salaries for all 30 teams (550 pitchers) were entered from saved FanGraphs RosterResource 2025 payroll pages. They were not scraped. Pirates and Dodgers rows were checked by hand.

- Traded pitchers are valued at their full 2025 salary.
- Young pitchers missing from the file get the league minimum, with team control estimated from their MLB debut date.
- Other pitchers missing from the file get the league minimum and one year of control. The dashboard tags these "est. contract".
- Shohei Ohtani's pitching is graded, but his contract isn't valued, because his salary pays for hitting too.

## Dashboard

- **Team Value Board:** ranks a team's pitchers by surplus value.
- **Pitcher Report:** Stuff+ and Location+ by pitch type, a movement chart, velocity by game with IL stints marked, and surplus value by year.
- Small samples are tagged, and their colors fade toward gray.

The public version runs on Streamlit Community Cloud. To update it, rerun `python -m analysis.run_all`, then commit and push `dashboard/data/`.

## Key findings so far

- **Stuff+ is consistent but only a weak predictor.** It ranks a pitcher the same way from start to start, but it predicts whiffs on new pitchers only weakly (AUC 0.51 to 0.59). Treat it as a ranking of pitch shape, not a whiff probability.
- **Some command stats are mostly noise.** Edge% barely stabilizes even after thousands of pitches. Location+ is far more reliable.

## Limitations

- The Stuff+ and Location+ proxies are simpler than the commercial versions.
- Location+ can't see where the catcher set up, so it measures location, not intent.
- The WAR projection is a placeholder that hasn't been calibrated against real WAR yet, and it overstates relievers.
- Value assumptions (arbitration raises, cost per WAR, discount rate) are set by hand, not fitted.
- Some contract and control values are estimates.

## Repo layout

```
analysis/     analysis scripts (one per model, plus run_all.py)
dashboard/    Streamlit app and the data files it reads
data/         salaries.csv and missing-contract report (raw caches gitignored)
config.py     season, thresholds, and model assumptions
notebooks/    exploratory work
```