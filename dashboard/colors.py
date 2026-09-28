"""
Savant-style good/bad colors for dashboard numbers.

stat_color() maps a value to a diverging gray -> red (better than league
average) / gray -> blue (worse) scale, from its z-score against the
LEAGUE-WIDE qualified pool. Red/blue instead of green/red so it reads for
red-green colorblind viewers. Color is never the only cue: the number (and
percentile where shown) is always printed next to it.
"""
import numpy as np
import pandas as pd

GOOD_COLOR = "#D82129"      # z = +2 (better than league average)
NEUTRAL_COLOR = "#BFBFBF"   # z = 0 (league average)
# #BFBFBF text is too faint on a white page, so colored NUMBERS on the light theme
# blend from this darker gray instead (chart fills and the dark theme keep #BFBFBF)
NEUTRAL_TEXT_LIGHT = "#7A7A7A"
BAD_COLOR = "#3661AD"       # z = -2 (worse than league average)
Z_CLIP = 2.0                # one extreme pitcher shouldn't wash out everyone else
TINT_ALPHA = 0.30           # default alpha for rgba() fills

# Metrics on the 100 + 10z scale: league mean 100, SD 10 by construction.
SCALED_METRICS = {"StuffPlus_Scaled", "LocationPlus"}

# True = higher is better (red when high); False = lower is better (red when low).
METRIC_DIRECTION = {
    "StuffPlus_Scaled": True,
    "LocationPlus": True,
    "AsymmetricUpsideIndex": True,
    "RiskAdjWAR": True,
    "MultiYearSurplus_M": True,
    "SurplusCurrentSeason_M": True,
    "EdgePct": True,
    "FirstPitchStrikePct": True,
    "FastballVelo": True,          # avg velocity, fastballs (FF/SI/FC) only
    "BBPct": False,
    "MeatballPct": False,
    "ThreeBallPct": False,
    "VolatilityScore": False,
    "VolatilityPercentile": False,
    "TotalDaysMissed": False,
    "ArmILStints": False,
    "Salary_M": False,
}
# Context-dependent -- coloring these would mislead, so they're never colored:
# IVB, horizontal break, arm angle, spin axis, usage %, pitch counts, service
# time, years of control, age, names/IDs, dates.

FASTBALL_TYPES = {"FF", "SI", "FC"}


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(c)):02X}" for c in rgb)


def stat_z(value, league_mean, league_sd, higher_is_better: bool) -> float | None:
    """z vs. the league pool, sign-flipped so positive = good, clipped to +/- Z_CLIP."""
    if value is None or pd.isna(value) or not league_sd or pd.isna(league_sd):
        return None
    z = (value - league_mean) / league_sd
    if not higher_is_better:
        z = -z
    return float(np.clip(z, -Z_CLIP, Z_CLIP))


def z_color(z: float | None, neutral: str = NEUTRAL_COLOR) -> str | None:
    """Blend neutral gray toward red (z > 0) or blue (z < 0) by |z| / Z_CLIP."""
    if z is None:
        return None
    end = GOOD_COLOR if z > 0 else BAD_COLOR
    t = abs(z) / Z_CLIP
    return _rgb_to_hex((1 - t) * _hex_to_rgb(neutral) + t * _hex_to_rgb(end))


def stat_color(value, league_mean, league_sd, higher_is_better: bool, neutral: str = NEUTRAL_COLOR) -> str | None:
    """Hex color for a value, or None (no color) when it's missing."""
    return z_color(stat_z(value, league_mean, league_sd, higher_is_better), neutral)


def rgba(color: str | None, alpha: float = TINT_ALPHA) -> str:
    if color is None:
        return "transparent"
    r, g, b = _hex_to_rgb(color)
    return f"rgba({int(r)}, {int(g)}, {int(b)}, {alpha})"


def league_percentile(value, pool: pd.Series, higher_is_better: bool) -> float | None:
    """Share of the league pool this value is better than (0-100), ties split half."""
    pool = pool.dropna()
    if value is None or pd.isna(value) or pool.empty:
        return None
    worse = (pool < value) if higher_is_better else (pool > value)
    return 100 * (worse.mean() + 0.5 * (pool == value).mean())


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class LeaguePools:
    """Mean / SD / pool per metric, from the league-wide graded table (pitcher_report_data.csv)."""

    def __init__(self, report: pd.DataFrame | None, neutral: str = NEUTRAL_COLOR):
        self.report = report
        self.neutral = neutral  # league-average color (NEUTRAL_TEXT_LIGHT for numbers on the light theme)

    def pool(self, metric: str) -> pd.Series:
        if self.report is None or metric not in self.report:
            return pd.Series(dtype=float)
        return self.report[metric].dropna()

    def mean_sd(self, metric: str) -> tuple[float, float]:
        if metric in SCALED_METRICS:
            return 100.0, 10.0
        pool = self.pool(metric)
        return (pool.mean(), pool.std()) if len(pool) >= 2 else (np.nan, np.nan)

    def color(self, metric: str, value) -> str | None:
        mean, sd = self.mean_sd(metric)
        return stat_color(value, mean, sd, METRIC_DIRECTION[metric], self.neutral)

    def z(self, metric: str, value) -> float | None:
        mean, sd = self.mean_sd(metric)
        return stat_z(value, mean, sd, METRIC_DIRECTION[metric])

    def percentile(self, metric: str, value) -> float | None:
        return league_percentile(value, self.pool(metric), METRIC_DIRECTION[metric])


def stat_html(label: str, value_text: str, color: str | None, percentile: float | None) -> str:
    """Stat card: label, the number in its good/bad color, league percentile underneath."""
    pct = f"{ordinal(percentile)} pct" if percentile is not None else "&nbsp;"
    return (
        '<div style="padding:4px 0 10px 0;">'
        f'<div style="font-size:0.85rem;opacity:0.75;">{label}</div>'
        f'<div style="font-size:2rem;font-weight:700;color:{color or "inherit"};">{value_text}</div>'
        f'<div style="font-size:0.75rem;opacity:0.7;">{pct}</div>'
        "</div>"
    )


def legend_html() -> str:
    """Gradient key: blue (worse) <- gray (league average) -> red (better)."""
    return (
        '<div style="max-width:520px;margin:4px 0 12px 0;font-size:0.8rem;">'
        f'<div style="height:10px;border-radius:5px;background:linear-gradient(to right, '
        f'{BAD_COLOR}, {NEUTRAL_COLOR}, {GOOD_COLOR});"></div>'
        '<div style="display:flex;justify-content:space-between;margin-top:2px;">'
        "<span>Worse than league average</span><span>League average</span>"
        "<span>Better than league average</span></div>"
        '<div style="opacity:0.7;">Compared to all qualified MLB pitchers.</div>'
        "</div>"
    )


def colored_numbers(df: pd.DataFrame, colors: pd.DataFrame, formats: dict | None = None):
    """pandas Styler: each number's text in its good/bad color (`colors`: same shape, hex or None).

    Uncolored cells keep the theme's text color; missing values show "n/a".
    """
    css = colors.reindex(index=df.index, columns=df.columns).map(
        lambda c: f"color: {c}; font-weight: 600" if isinstance(c, str) else ""
    )
    return df.style.format(formats or {}, na_rep="n/a").apply(lambda _: css, axis=None)
