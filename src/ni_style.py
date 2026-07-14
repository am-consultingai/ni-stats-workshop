"""
ni_style.py — shared theme, data loaders, and statistics helpers for the
Natural Intelligence "Practical Statistics for Analysts" workshop notebooks.

Import once at the top of every notebook:

    import ni_style as ni
    ni.set_style()
    visits = ni.load_visits()
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------------------- #
# Paths (resolved from this file, so notebooks work regardless of cwd)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Brand palette (echoes the NI navy/blue from the syllabus email)
# --------------------------------------------------------------------------- #
NAVY   = "#0B1E3A"
BLUE   = "#2E6BE6"
SKY    = "#7FB0FF"
LIGHT  = "#D7E6FF"
TEAL   = "#17A2B8"
GREEN  = "#2BB673"
ORANGE = "#F0883E"
RED    = "#E5484D"
GOLD   = "#E8B910"
GREY   = "#8A94A6"

SEQ = [BLUE, TEAL, GREEN, ORANGE, RED, GOLD, NAVY, GREY]

ENGINE_COLORS = {"Google": BLUE, "Bing": TEAL, "Organic": GREEN, "Social": ORANGE}
DEVICE_COLORS = {"desktop": NAVY, "mobile": BLUE, "tablet": SKY}
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def set_style() -> None:
    """Apply the workshop's house chart style."""
    mpl.rcParams.update({
        "figure.figsize":      (9, 5.2),
        "figure.dpi":          110,
        "savefig.dpi":         150,
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#C9D2E0",
        "axes.grid":           True,
        "grid.color":          "#E7ECF4",
        "grid.linewidth":      1.0,
        "axes.axisbelow":      True,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.titlesize":      14,
        "axes.titleweight":    "bold",
        "axes.titlecolor":     NAVY,
        "axes.titlepad":       12,
        "axes.labelsize":      11,
        "axes.labelcolor":     "#33415C",
        "xtick.color":         "#5A6580",
        "ytick.color":         "#5A6580",
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.frameon":      False,
        "legend.fontsize":     10,
        "font.size":           11,
        "axes.prop_cycle":     mpl.cycler(color=SEQ),
    })


def savefig(fig, name: str) -> Path:
    """Save a figure to figures/ as PNG (for dropping into slides) and return its path."""
    path = FIGS / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path


def titlebox(ax, title: str, subtitle: str | None = None) -> None:
    """Two-line title: bold title + grey subtitle."""
    ax.set_title(title, loc="left")
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=10.5, color=GREY, ha="left", va="bottom")


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_visits() -> pd.DataFrame:
    df = pd.read_csv(DATA / "visits.csv", parse_dates=["date"])
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DOW_ORDER, ordered=True)
    return df


def load_agg(name: str) -> pd.DataFrame:
    parse = ["date"] if "daily" in name else None
    return pd.read_csv(DATA / f"{name}.csv", parse_dates=parse)


# --------------------------------------------------------------------------- #
# Statistics helpers (kept tiny and transparent — these are teaching tools)
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, centre - half, centre + half


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int):
    """Pooled two-proportion z-test. Returns dict with diff, z, p (two-sided)."""
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": z, "p_value": p}


def diff_proportion_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96):
    """Unpooled CI for the difference of two proportions (p1 - p2)."""
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - z * se, d + z * se


def bootstrap_ratio_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                       seed: int = 0, ci: float = 95.0):
    """
    Bootstrap CI for (mean(a)/mean(b) - 1), i.e. the % difference of group a vs b.
    Used for EPV ratios (mobile vs desktop). Returns (point, lo, hi) in PERCENT.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    point = a.mean() / b.mean() - 1.0
    na, nb = len(a), len(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = a[rng.integers(0, na, na)].mean() / b[rng.integers(0, nb, nb)].mean() - 1.0
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return point * 100, lo * 100, hi * 100, boots * 100


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0, ci: float = 95.0):
    """Bootstrap CI for a mean (robust to skew). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return x.mean(), lo, hi


def power_two_proportions(p1: float, p2: float, n_per_group: int, alpha: float = 0.05):
    """Approximate power of a two-sided two-proportion z-test at a given n/group."""
    p_bar = (p1 + p2) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se1 = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z = (abs(p1 - p2) - z_a * se0) / se1
    return stats.norm.cdf(z)
