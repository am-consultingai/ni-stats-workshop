#!/usr/bin/env python3
"""
ni_core.py — shared deterministic primitives for the NI workshop skills (v2).

WHY THIS FILE EXISTS
--------------------
The workshop's core principle: *the statistics lives in fixed, reviewed Python;
Claude authors and routes to it — it is never the calculator.*

Every skill is a thin script that imports these primitives, runs ONE analysis,
and prints a report ending in a `VALIDATION` block. Because the maths is fixed
and every random step is seeded (SEED), running a skill twice yields
**byte-identical** output (determinism, R1). Because the code is short and
readable, an analyst can audit exactly what ran (validation literacy, R2).

This module holds only the reusable maths + loaders + formatting. The per-skill
scripts (profile.py, significance.py, bayesian_update.py, trend.py, …) live one directory
up, in `.claude/skills/<skill>/`, and import `ni_core`.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Paths & constants.  This file is  new_workshop/.claude/skills/_lib/ni_core.py
# parents: [0]=_lib [1]=skills [2]=.claude [3]=new_workshop
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VISITS = ROOT / "data" / "online_banking_visit_clickouts.csv"
DEFAULT_COST = ROOT / "data" / "online_banking_daily_cost.csv"

SEED = 42
N_BOOT = 5000
JOIN_DIMS = ["day_of_week", "is_weekend", "channel", "platform", "segment", "campaign"]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_clickouts(path: Path = DEFAULT_VISITS) -> pd.DataFrame:
    """One row per click-out, plus one row per non-clicking visit (raw grain).

    `clicked` is True where a partner was clicked; those rows carry
    partner / clickout_position / revenue and are the grain for EPC / drill-down
    questions ("is partner B really better than A in Bing?").
    """
    df = pd.read_csv(path, parse_dates=["click_timestamp"])
    df["clicked"] = df["click_id"].notna()
    df["date"] = df["click_timestamp"].dt.normalize()
    df["month"] = df["click_timestamp"].dt.strftime("%Y-%m")
    return df


def load_visits(path: Path = DEFAULT_VISITS) -> pd.DataFrame:
    """Collapse to one row per visit (the analyst's unit of value).

    revenue is summed across the visit's click-outs; `converted` = any click-out
    converted; EPV == revenue at this grain. Collapsing to the visit is also the
    simplest way to restore i.i.d. (one independent row per visit).
    """
    raw = load_clickouts(path)
    vis = (raw.groupby("visit_iid")
              .agg(click_timestamp=("click_timestamp", "first"),
                   date=("date", "first"),
                   month=("month", "first"),
                   day_of_week=("day_of_week", "first"),
                   is_weekend=("is_weekend", "first"),
                   channel=("channel", "first"),
                   platform=("platform", "first"),
                   segment=("segment", "first"),
                   campaign=("campaign", "first"),
                   clicked=("clicked", "max"),
                   n_clickouts=("clicked", "sum"),
                   converted=("converted", "max"),
                   revenue=("revenue", "sum"))
              .reset_index())
    vis["epv"] = vis["revenue"]
    return vis


def load_cost(path: Path = DEFAULT_COST) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


# --------------------------------------------------------------------------- #
# Shape diagnostics — decide which tools are even valid
# --------------------------------------------------------------------------- #
def describe_shape(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    q99 = np.quantile(x, 0.99) if n else np.nan
    top1_share = (x[x >= q99].sum() / x.sum()) if (n and x.sum() > 0) else np.nan
    return {
        "n": n,
        "mean": x.mean() if n else np.nan,
        "median": np.median(x) if n else np.nan,
        "trimmed_mean_10": stats.trim_mean(x, 0.10) if n else np.nan,
        "std": x.std(ddof=1) if n > 1 else np.nan,
        "skew": stats.skew(x) if n > 2 else np.nan,
        "zero_share": float((x == 0).mean()) if n else np.nan,
        "top1_share": top1_share,
        "min": x.min() if n else np.nan,
        "max": x.max() if n else np.nan,
    }


# --------------------------------------------------------------------------- #
# Proportions (rates)
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, centre - half, centre + half


def two_proportion_ztest(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": z, "p_value": p}


def diff_proportion_ci(k1, n1, k2, n2, z=1.96):
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - z * se, d + z * se


def power_two_proportions(p1, p2, n_per_group, alpha=0.05):
    if n_per_group <= 0:
        return np.nan
    p_bar = (p1 + p2) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se1 = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z = (abs(p1 - p2) - z_a * se0) / se1 if se1 > 0 else np.nan
    return stats.norm.cdf(z)


# --------------------------------------------------------------------------- #
# Bootstrap (i.i.d.) — robust to skew; the default for EPV / revenue / EPC
# --------------------------------------------------------------------------- #
def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=SEED, ci=95.0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return x.mean(), lo, hi


def bootstrap_mean_diff_ci(a, b, n_boot=N_BOOT, seed=SEED, ci=95.0):
    """Bootstrap CI for mean(a) - mean(b) and for the % lift mean(a)/mean(b) - 1."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    ma = a[rng.integers(0, na, size=(n_boot, na))].mean(axis=1)
    mb = b[rng.integers(0, nb, size=(n_boot, nb))].mean(axis=1)
    diff = ma - mb
    lift = ma / mb - 1.0
    lo_d, hi_d = np.percentile(diff, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    lo_l, hi_l = np.percentile(lift, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"diff": a.mean() - b.mean(), "diff_lo": lo_d, "diff_hi": hi_d,
            "lift_pct": (a.mean() / b.mean() - 1) * 100 if b.mean() else np.nan,
            "lift_lo": lo_l * 100, "lift_hi": hi_l * 100}


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


# --------------------------------------------------------------------------- #
# i.i.d. / clustering  (Module 2) — how much data is REALLY here
# --------------------------------------------------------------------------- #
def design_effect(values, clusters) -> dict:
    """One-way-random-effects ICC + design effect for clustered rows.

    deff = 1 + (mean_cluster_size - 1) * ICC ;   effective_n = n / deff.
    Rows sharing a cluster (e.g. two click-outs from one visit) are not
    independent — deff quantifies how much that inflates false positives.
    """
    df = pd.DataFrame({"v": np.asarray(values, float), "c": np.asarray(clusters)})
    n = len(df)
    grp = df.groupby("c")["v"]
    k = grp.ngroups
    if k <= 1 or n <= k:
        return {"n": n, "k": k, "mbar": n / max(k, 1), "icc": 0.0, "deff": 1.0, "n_eff": float(n)}
    n_i = grp.size().to_numpy(float)
    mean_i = grp.mean().to_numpy(float)
    grand = df["v"].mean()
    ms_between = float((n_i * (mean_i - grand) ** 2).sum() / (k - 1))
    group_mean_map = grp.transform("mean")
    ss_within = float(((df["v"] - group_mean_map) ** 2).sum())
    ms_within = ss_within / (n - k)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    icc = float((ms_between - ms_within) / denom) if denom > 0 else 0.0
    icc = max(0.0, icc)
    mbar = n / k
    deff = 1 + (mbar - 1) * icc
    return {"n": n, "k": k, "mbar": mbar, "icc": icc, "deff": deff, "n_eff": n / deff}


def cluster_bootstrap_mean_ci(values, clusters, n_boot=N_BOOT, seed=SEED, ci=95.0):
    """Bootstrap the mean by resampling whole CLUSTERS (visits), not rows.

    This is the honest interval when rows are clustered: it widens correctly
    instead of pretending every row is independent.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    order = np.argsort(c, kind="stable")
    v, c = v[order], c[order]
    uniq, start = np.unique(c, return_index=True)
    groups = np.split(v, start[1:])
    kk = len(groups)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, kk, size=kk)
        boots[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return v.mean(), lo, hi


def cluster_bootstrap_diff_ci(a_vals, a_clusters, b_vals, b_clusters,
                              n_boot=N_BOOT, seed=SEED, ci=95.0):
    """Cluster-bootstrap CI for mean(a) - mean(b), resampling visits per group."""
    rng = np.random.default_rng(seed)

    def _groups(vals, clus):
        v = np.asarray(vals, float); c = np.asarray(clus)
        order = np.argsort(c, kind="stable")
        v, c = v[order], c[order]
        _, start = np.unique(c, return_index=True)
        return np.split(v, start[1:])

    ga, gb = _groups(a_vals, a_clusters), _groups(b_vals, b_clusters)
    ka, kb = len(ga), len(gb)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ma = np.concatenate([ga[j] for j in rng.integers(0, ka, size=ka)]).mean()
        mb = np.concatenate([gb[j] for j in rng.integers(0, kb, size=kb)]).mean()
        diffs[i] = ma - mb
    lo, hi = np.percentile(diffs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"diff": float(np.mean(a_vals) - np.mean(b_vals)),
            "diff_lo": float(lo), "diff_hi": float(hi)}


# --------------------------------------------------------------------------- #
# Bayesian / partial pooling  (Module 4) — borrow strength from history
# --------------------------------------------------------------------------- #
def mean_and_se2(x):
    """Return (mean, variance-of-the-mean = se^2, n). se^2 shrinks like 1/n."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, 0)
    if n == 1:
        return (float(x[0]), np.inf, 1)
    return (float(x.mean()), float(x.var(ddof=1) / n), n)


def posterior_from_prior(prior_mean, prior_se2, data_mean, data_se2) -> dict:
    """Variance-weighted normal update (conjugate). Narrower than either input.

    Precision adds: 1/var_post = 1/prior + 1/data. Used to anchor a thin slice
    on an informative prior (the partner's history / other slices).
    """
    vp = 1.0 / (1.0 / prior_se2 + 1.0 / data_se2)
    mp = vp * (prior_mean / prior_se2 + data_mean / data_se2)
    se = float(np.sqrt(vp))
    return {"mean": float(mp), "se": se, "var": float(vp),
            "lo": float(mp - 1.96 * se), "hi": float(mp + 1.96 * se)}


def p_better(mean_a, var_a, mean_b, var_b) -> float:
    """P(theta_a > theta_b) for independent normal posteriors."""
    s = np.sqrt(var_a + var_b)
    if s == 0:
        return float(mean_a > mean_b)
    return float(stats.norm.cdf((mean_a - mean_b) / s))


def expected_loss_choosing(mean_a, var_a, mean_b, var_b) -> dict:
    """Expected regret of picking each arm, under normal posteriors.

    D = theta_a - theta_b ~ N(mD, sD^2). Loss of choosing A = E[max(0, -D)]
    (regret if B is truly better), and symmetrically for B.
    """
    mD = mean_a - mean_b
    sD = np.sqrt(var_a + var_b)
    if sD == 0:
        return {"choose_a": max(0.0, -mD), "choose_b": max(0.0, mD)}
    phi = stats.norm.pdf(mD / sD)
    Phi = stats.norm.cdf(mD / sD)
    loss_a = float(sD * phi - mD * (1 - Phi))   # E[max(0,-D)]
    loss_b = float(sD * phi + mD * Phi)         # E[max(0, D)]
    return {"choose_a": loss_a, "choose_b": loss_b}


# --------------------------------------------------------------------------- #
# Trend / seasonality  (Module 5) — is the shift real over time?
# --------------------------------------------------------------------------- #
def daily_series(df, value_col, date_col="date", how="mean") -> pd.Series:
    """Daily series of a metric (mean, or rate for 0/1)."""
    g = df.groupby(date_col)[value_col]
    s = g.mean() if how in ("mean", "rate") else g.sum()
    return s.sort_index()


def dow_baseline(df, value_col, date_col="date", dow_col="day_of_week") -> pd.DataFrame:
    """Per-day-of-week mean & std of the daily series (the seasonality baseline)."""
    daily = df.groupby([date_col, dow_col])[value_col].mean().reset_index()
    tab = (daily.groupby(dow_col)[value_col]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "dow_mean", "std": "dow_std"}))
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tab = tab.reindex([d for d in order if d in tab.index])
    return tab


def trend_check(df, value_col, date_col="date", dow_col="day_of_week",
                recent_days=7, how="mean") -> dict:
    """Is the latest day's value a real move, or within normal variation?

    Compares the latest day against (a) the month-wide band (mean ± 2 std of the
    daily series) and (b) its own day-of-week baseline, and runs a light
    regime-break test: mean of the last `recent_days` vs the prior distribution.
    """
    s = daily_series(df, value_col, date_col, how)
    if len(s) < 3:
        return {"n_days": len(s), "insufficient": True}
    latest_date = s.index[-1]
    latest = float(s.iloc[-1])
    month_mean, month_std = float(s.mean()), float(s.std(ddof=1))
    band_lo, band_hi = month_mean - 2 * month_std, month_mean + 2 * month_std

    dow = df.loc[df[date_col] == latest_date, dow_col]
    dow = dow.iloc[0] if len(dow) else None
    base = dow_baseline(df, value_col, date_col, dow_col)
    dow_mean = float(base.loc[dow, "dow_mean"]) if (dow in base.index) else np.nan
    dow_std = float(base.loc[dow, "dow_std"]) if (dow in base.index) else np.nan

    recent = s.iloc[-recent_days:]
    prior = s.iloc[:-recent_days]
    regime_z = np.nan
    if len(prior) >= 3 and prior.std(ddof=1) > 0:
        se = prior.std(ddof=1) / np.sqrt(len(recent))
        regime_z = float((recent.mean() - prior.mean()) / se) if se > 0 else np.nan

    return {
        "n_days": len(s), "latest_date": str(latest_date.date()), "dow": dow,
        "latest": latest, "month_mean": month_mean, "month_std": month_std,
        "band_lo": band_lo, "band_hi": band_hi,
        "within_band": band_lo <= latest <= band_hi,
        "dow_mean": dow_mean, "dow_std": dow_std,
        "dow_within": (abs(latest - dow_mean) <= 2 * dow_std) if np.isfinite(dow_std) and dow_std > 0 else None,
        "recent_mean": float(recent.mean()), "prior_mean": float(prior.mean()) if len(prior) else np.nan,
        "regime_z": regime_z, "regime_break": (abs(regime_z) > 2) if np.isfinite(regime_z) else False,
    }


# --------------------------------------------------------------------------- #
# Funnel / profit  (Module 6) — shared by budget + decide
# --------------------------------------------------------------------------- #
def budget_table(vis, cost, by=("channel", "platform")) -> pd.DataFrame:
    """Profit-per-visit table by the given dims. Cost is aggregate (no per-user
    cost — Maor's real constraint): CPV = summed cost / visits in the slice."""
    by = list(by)
    cost_by = cost.groupby(by)["cost"].sum()
    rows = []
    for key, g in vis.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        c = float(cost_by.get(key if len(key) > 1 else key[0], 0.0))
        epv = g["revenue"].mean()
        _, lo, hi = bootstrap_mean_ci(g["revenue"].to_numpy(float))
        row = dict(zip(by, key))
        row.update({"visits": n, "conv%": g["converted"].mean() * 100,
                    "EPV": epv, "CPV": c / n if n else np.nan,
                    "profit/visit": epv - (c / n if n else np.nan),
                    "EPV_lo": lo, "EPV_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("profit/visit", ascending=False)


# --------------------------------------------------------------------------- #
# Report formatting + slice selection helpers
# --------------------------------------------------------------------------- #
def hr(title):
    return f"\n{'─' * 72}\n{title}\n{'─' * 72}"


def flag(msg):
    return f"  🚩 {msg}"


def ok(msg):
    return f"  ✅ {msg}"


def apply_slice(df, slices):
    """Filter df by ["col=val", ...] (e.g. ['channel=Bing'])."""
    for spec in (slices or []):
        col, _, val = spec.partition("=")
        df = df[df[col].astype(str) == val]
    return df


def emit(lines):
    print("\n".join(lines))
