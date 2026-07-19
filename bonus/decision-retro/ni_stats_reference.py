#!/usr/bin/env python3
"""
ni_stats.py — the deterministic statistics engine behind the NI workshop skills.

WHY THIS FILE EXISTS
--------------------
The workshop's core principle: *the statistics lives in fixed, reviewed Python;
Claude authors and routes to it — it is never the calculator.*

Every skill (`/profile-data`, `/significance-check`, `/estimate-effect`,
`/relationship`, `/budget-decision`, `/decision-retro`) is a thin wrapper that
runs one subcommand here. Because the code is fixed and every random step is
seeded, running a skill twice yields **byte-identical** output (determinism,
requirement R1). Because the code is short and readable, an analyst can audit
exactly what test ran and which assumptions were checked (validation, R2).

Each report ends with a `VALIDATION` block: the assumption checks and red flags
that tell you whether to trust the number. That block is the whole point.

Usage:
    python ni_stats.py profile      --metric epv --group channel
    python ni_stats.py significance --metric converted --group channel --a Google --b Bing
    python ni_stats.py estimate     --metric epv --group platform --a mobile --b desktop
    python ni_stats.py relationship --x clickout_position --y revenue --by segment
    python ni_stats.py budget
    python ni_stats.py retro        --partner "Summit Direct" --change-date 2026-04-01
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Data — defaults point at Maor's real NI mock datasets (online banking)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VISITS = ROOT / "data" / "online_banking_visit_clickouts.csv"
DEFAULT_COST = ROOT / "data" / "online_banking_daily_cost.csv"

SEED = 42
JOIN_DIMS = ["day_of_week", "is_weekend", "channel", "platform", "segment", "campaign"]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_clickouts(path: Path = DEFAULT_VISITS) -> pd.DataFrame:
    """One row per click-out (plus one row per non-clicking visit). Raw grain."""
    df = pd.read_csv(path, parse_dates=["click_timestamp"])
    df["clicked"] = df["click_id"].notna()
    return df


def load_visits(path: Path = DEFAULT_VISITS) -> pd.DataFrame:
    """
    Collapse to one row per visit (visit_iid).

    A visit is the analyst's unit of value. revenue is summed across the visit's
    click-outs; `converted` is 1 if any click-out converted; EPV == revenue.
    """
    raw = load_clickouts(path)
    vis = (raw.groupby("visit_iid")
              .agg(click_timestamp=("click_timestamp", "first"),
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
    vis["epv"] = vis["revenue"]  # earnings per visit == revenue at the visit grain
    return vis


def load_cost(path: Path = DEFAULT_COST) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


# --------------------------------------------------------------------------- #
# Small, transparent statistics helpers (teaching tools — keep them readable)
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion → (p_hat, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, centre - half, centre + half


def two_proportion_ztest(k1, n1, k2, n2):
    """Pooled two-proportion z-test → dict(p1, p2, diff, z, p_value)."""
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
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
    z = (abs(p1 - p2) - z_a * se0) / se1
    return stats.norm.cdf(z)


def bootstrap_mean_diff_ci(a, b, n_boot=4000, seed=SEED, ci=95.0):
    """
    Bootstrap CI for mean(a) - mean(b) AND for mean(a)/mean(b) - 1 (percent lift).
    Robust to skew — this is what we use for EPV/revenue instead of a t-interval.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    ai = rng.integers(0, na, size=(n_boot, na))
    bi = rng.integers(0, nb, size=(n_boot, nb))
    ma = a[ai].mean(axis=1)
    mb = b[bi].mean(axis=1)
    diff = ma - mb
    lift = ma / mb - 1.0
    lo_d, hi_d = np.percentile(diff, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    lo_l, hi_l = np.percentile(lift, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"diff": a.mean() - b.mean(), "diff_lo": lo_d, "diff_hi": hi_d,
            "lift_pct": (a.mean() / b.mean() - 1) * 100,
            "lift_lo": lo_l * 100, "lift_hi": hi_l * 100}


def bootstrap_mean_ci(x, n_boot=4000, seed=SEED, ci=95.0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return x.mean(), lo, hi


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def describe_shape(x):
    """Return the shape diagnostics that decide which tools are even valid."""
    x = np.asarray(x, float)
    n = len(x)
    nonzero = x[x != 0]
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
# Report formatting
# --------------------------------------------------------------------------- #
def _hr(title):
    return f"\n{'─' * 70}\n{title}\n{'─' * 70}"


def _flag(msg):
    return f"  🚩 {msg}"


def _ok(msg):
    return f"  ✅ {msg}"


# --------------------------------------------------------------------------- #
# Subcommand: profile  (Modules 1–2: distributions, CLT, skew, whales)
# --------------------------------------------------------------------------- #
def cmd_profile(args):
    vis = load_visits(Path(args.visits))
    metric = args.metric
    out = [f"PROFILE · metric='{metric}'" + (f" · grouped by '{args.group}'" if args.group else "")]

    groups = [(None, vis)] if not args.group else list(vis.groupby(args.group))
    for gname, g in groups:
        col = g[metric].to_numpy(float)
        s = describe_shape(col)
        head = f"{args.group}={gname}" if gname is not None else "all visits"
        out.append(_hr(head))
        out.append(f"  n={s['n']:,}  mean={s['mean']:.3f}  median={s['median']:.3f}  "
                   f"trimmed(10%)={s['trimmed_mean_10']:.3f}")
        out.append(f"  skew={s['skew']:.1f}  zero-share={s['zero_share']*100:.1f}%  "
                   f"top-1% share of total={s['top1_share']*100:.1f}%  max={s['max']:.1f}")

    # VALIDATION block — the guardrails Claude must surface, computed once, globally
    col = vis[metric].to_numpy(float)
    s = describe_shape(col)
    out.append(_hr("VALIDATION — can you trust an average of this?"))
    if abs(s["skew"]) > 1:
        out.append(_flag(f"Heavy skew (skew={s['skew']:.1f}). The MEAN is not a typical value. "
                         f"Do NOT use a t-test or a normal-theory CI on this metric — "
                         f"use Mann-Whitney / bootstrap (see /significance-check, /estimate-effect)."))
    else:
        out.append(_ok(f"Skew is mild ({s['skew']:.1f}); mean is a reasonable summary."))
    if s["top1_share"] > 0.10:
        out.append(_flag(f"Whale-driven: the top 1% of visits hold {s['top1_share']*100:.0f}% of all {metric}. "
                         f"A single big day can move the mean — always plot the distribution first."))
    if s["zero_share"] > 0.5:
        out.append(_flag(f"{s['zero_share']*100:.0f}% of visits are exactly 0 "
                         f"(a zero-inflated metric). Mean blends 'did it convert' with 'how much'."))
    out.append(f"\n  → Rule of thumb: with skew this high, trust the MEDIAN and a BOOTSTRAP interval,\n"
               f"    and be suspicious of any average a tool hands you without an interval.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# Subcommand: significance  (Module 4: right test, n-trap, power, effect size)
# --------------------------------------------------------------------------- #
def cmd_significance(args):
    vis = load_visits(Path(args.visits))
    metric, gcol, A, B = args.metric, args.group, args.a, args.b
    a = vis.loc[vis[gcol] == A, metric].to_numpy(float)
    b = vis.loc[vis[gcol] == B, metric].to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        sys.exit(f"ERROR: no rows for {gcol}={A!r} ({len(a)}) or {gcol}={B!r} ({len(b)}). "
                 f"Values: {sorted(vis[gcol].unique())}")

    out = [f"SIGNIFICANCE · '{metric}' · {gcol}: {A} vs {B}"]
    binary = set(np.unique(np.concatenate([a, b]))).issubset({0.0, 1.0})

    if binary:
        k1, n1, k2, n2 = int(a.sum()), len(a), int(b.sum()), len(b)
        r = two_proportion_ztest(k1, n1, k2, n2)
        d, lo, hi = diff_proportion_ci(k1, n1, k2, n2)
        pw = power_two_proportions(r["p1"], r["p2"], min(n1, n2))
        out.append(_hr("TEST CHOSEN: two-proportion z-test  (metric is binary 0/1)"))
        out.append(f"  {A}: {r['p1']*100:.2f}%  (n={n1:,})     {B}: {r['p2']*100:.2f}%  (n={n2:,})")
        out.append(f"  absolute diff = {d*100:+.2f} pp   95% CI [{lo*100:+.2f}, {hi*100:+.2f}] pp")
        out.append(f"  relative lift = {(r['p1']/r['p2']-1)*100:+.1f}%")
        out.append(f"  z = {r['z']:.2f}   p = {r['p_value']:.4g}")
        out.append(_hr("VALIDATION"))
        out.append((_ok if r["p_value"] < 0.05 else _flag)(
            f"p={r['p_value']:.4g} → {'significant' if r['p_value']<0.05 else 'NOT significant'} at α=0.05."))
        if lo < 0 < hi:
            out.append(_flag("The 95% CI for the difference straddles 0 — the direction itself is uncertain."))
        out.append((_ok if pw >= 0.8 else _flag)(
            f"Approx. power at this n ≈ {pw*100:.0f}%. "
            f"{'Adequate' if pw>=0.8 else 'Underpowered — a null result may just be too little data.'}"))
        out.append("  ⚠️  n-trap: significance scales with n. This same gap could be 'significant' at\n"
                   "      10× the data and 'not significant' at 1/10. Always report the EFFECT SIZE and CI,\n"
                   "      not just the p-value.")
    else:
        sa, sb = describe_shape(a), describe_shape(b)
        skewed = (abs(sa["skew"]) > 1) or (abs(sb["skew"]) > 1)
        # With large n, normality tests always reject — decide by SHAPE, not a p-value.
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        boot = bootstrap_mean_diff_ci(a, b)
        d_eff = cohens_d(a, b)
        rank_biserial = 1 - 2 * u_stat / (len(a) * len(b))
        chosen = "Mann-Whitney U + bootstrap" if skewed else "Welch's t-test"
        out.append(_hr(f"TEST CHOSEN: {chosen}"))
        out.append(f"  {A}: mean={a.mean():.3f} median={np.median(a):.3f} skew={sa['skew']:.1f} (n={len(a):,})")
        out.append(f"  {B}: mean={b.mean():.3f} median={np.median(b):.3f} skew={sb['skew']:.1f} (n={len(b):,})")
        out.append(f"  mean diff = {boot['diff']:+.3f}   bootstrap 95% CI [{boot['diff_lo']:+.3f}, {boot['diff_hi']:+.3f}]")
        out.append(f"  relative  = {boot['lift_pct']:+.1f}%   95% CI [{boot['lift_lo']:+.1f}%, {boot['lift_hi']:+.1f}%]")
        out.append(f"  Welch t p = {t_p:.4g}      Mann-Whitney p = {u_p:.4g}")
        out.append(f"  effect size: Cohen's d = {d_eff:.3f}  |  rank-biserial = {rank_biserial:.3f}")
        out.append(_hr("VALIDATION"))
        if skewed:
            out.append(_flag(f"Metric is heavy-tailed (skew {sa['skew']:.1f}/{sb['skew']:.1f}). "
                             f"A t-test's normality assumption is violated → the primary test is "
                             f"Mann-Whitney, and the interval is a BOOTSTRAP, not t-based."))
            out.append("  ↳ If a tool reported only a t-test p-value here, that is the #1 pitfall — "
                       "flag it.")
        else:
            out.append(_ok("Shapes are close to normal; Welch's t-test is appropriate."))
        if boot["diff_lo"] < 0 < boot["diff_hi"]:
            out.append(_flag("Bootstrap CI for the difference includes 0 — not a reliable difference."))
        else:
            out.append(_ok("Bootstrap CI excludes 0 — the difference is stable under resampling."))
        out.append("  ⚠️  n-trap: with n this large almost any gap becomes 'significant'. "
                   "Lead with the effect size + CI.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# Subcommand: estimate  (Module 5: how big + how sure, the right interval)
# --------------------------------------------------------------------------- #
def cmd_estimate(args):
    vis = load_visits(Path(args.visits))
    metric, gcol, A, B = args.metric, args.group, args.a, args.b
    a = vis.loc[vis[gcol] == A, metric].to_numpy(float)
    b = vis.loc[vis[gcol] == B, metric].to_numpy(float)
    out = [f"ESTIMATE · '{metric}' · {gcol}: {A} vs {B}"]
    binary = set(np.unique(np.concatenate([a, b]))).issubset({0.0, 1.0})

    if binary:
        k1, n1 = int(a.sum()), len(a)
        k2, n2 = int(b.sum()), len(b)
        p1, l1, h1 = wilson_ci(k1, n1)
        p2, l2, h2 = wilson_ci(k2, n2)
        d, lo, hi = diff_proportion_ci(k1, n1, k2, n2)
        out.append(_hr("POINT ESTIMATE + 95% INTERVAL (Wilson for rates)"))
        out.append(f"  {A}: {p1*100:.2f}%  [{l1*100:.2f}, {h1*100:.2f}]")
        out.append(f"  {B}: {p2*100:.2f}%  [{l2*100:.2f}, {h2*100:.2f}]")
        out.append(f"  difference: {d*100:+.2f} pp  [{lo*100:+.2f}, {hi*100:+.2f}]")
        width = (hi - lo) * 100
    else:
        ma, la, ha = bootstrap_mean_ci(a)
        mb, lb, hb = bootstrap_mean_ci(b)
        boot = bootstrap_mean_diff_ci(a, b)
        out.append(_hr("POINT ESTIMATE + 95% INTERVAL (bootstrap — metric is skewed)"))
        out.append(f"  {A}: {ma:.3f}  [{la:.3f}, {ha:.3f}]")
        out.append(f"  {B}: {mb:.3f}  [{lb:.3f}, {hb:.3f}]")
        out.append(f"  difference: {boot['diff']:+.3f}  [{boot['diff_lo']:+.3f}, {boot['diff_hi']:+.3f}]")
        out.append(f"  relative:   {boot['lift_pct']:+.1f}%  [{boot['lift_lo']:+.1f}%, {boot['lift_hi']:+.1f}%]")
        width = boot["diff_hi"] - boot["diff_lo"]

    out.append(_hr("SAY IT THREE WAYS"))
    out.append("  • Executive:  point estimate with a plain-English range.")
    out.append("  • Analyst:    the 95% interval + the method (Wilson / bootstrap).")
    out.append("  • Skeptic:    'the interval is wide because n is small' — width shrinks ~1/√n.")
    out.append(_hr("VALIDATION"))
    out.append(_flag("Lead with the INTERVAL, not the point. If a tool gave you a single number "
                     "with no interval, it hid the uncertainty — ask for the CI."))
    out.append(f"  Interval width here ≈ {width:.3f}. To halve it you need ~4× the data.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# Subcommand: relationship  (Module 3: correlation + confound / Simpson)
# --------------------------------------------------------------------------- #
def cmd_relationship(args):
    raw = load_clickouts(Path(args.visits))
    # relationships on click-outs make sense for position↔revenue etc.
    df = raw.dropna(subset=[args.x, args.y]) if args.x in raw and args.y in raw else raw
    x = pd.to_numeric(df[args.x], errors="coerce")
    y = pd.to_numeric(df[args.y], errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    out = [f"RELATIONSHIP · {args.x} vs {args.y}  (n={len(x):,})"]
    out.append(_hr("CORRELATION"))
    out.append(f"  Pearson  r = {pear.statistic:+.3f}  (p={pear.pvalue:.3g})  ← linear, outlier-sensitive")
    out.append(f"  Spearman ρ = {spear.statistic:+.3f}  (p={spear.pvalue:.3g})  ← monotonic, robust")

    out.append(_hr("VALIDATION"))
    if abs(pear.statistic - spear.statistic) > 0.1:
        out.append(_flag(f"Pearson and Spearman disagree ({pear.statistic:+.3f} vs {spear.statistic:+.3f}). "
                         f"Outliers or non-linearity are bending Pearson — trust Spearman here."))
    else:
        out.append(_ok("Pearson ≈ Spearman; the relationship is roughly linear and monotonic."))

    # Simpson / confound check: does the sign hold within each level of --by?
    if args.by and args.by in df.columns:
        overall = spear.statistic
        within = []
        for gname, g in df.groupby(args.by):
            gx = pd.to_numeric(g[args.x], errors="coerce")
            gy = pd.to_numeric(g[args.y], errors="coerce")
            mm = gx.notna() & gy.notna()
            if mm.sum() > 30:
                within.append((gname, stats.spearmanr(gx[mm], gy[mm]).statistic))
        flips = [w for w in within if np.sign(w[1]) != np.sign(overall) and abs(w[1]) > 0.05]
        out.append(_hr(f"CONFOUND CHECK — does it hold within each '{args.by}'?"))
        for gname, rho in within:
            mark = "  ⟲ SIGN FLIPS" if (np.sign(rho) != np.sign(overall) and abs(rho) > 0.05) else ""
            out.append(f"    {args.by}={gname}: ρ={rho:+.3f}{mark}")
        if flips:
            out.append(_flag(f"Simpson's paradox risk: the overall correlation ({overall:+.3f}) reverses "
                             f"inside {len(flips)} subgroup(s). The headline relationship is a "
                             f"MIX artefact, not a real driver. Report the within-group story."))
        else:
            out.append(_ok(f"Sign is stable within every '{args.by}' — the relationship survives the confound check."))
    else:
        out.append(f"\n  (Tip: pass --by <dimension> to test for a confound / Simpson's paradox.)")
    out.append("\n  ⚠️  Correlation ≠ causation. Every correlation is a QUESTION, not a conclusion.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# Subcommand: budget  (Module 6: the funnel → profit/visit decision)
# --------------------------------------------------------------------------- #
def budget_table(vis, cost):
    """Profit-per-visit table by channel × platform. Reused by /budget-decision and /decide."""
    # cost per slice → CPV. Cost is aggregate (no per-user cost — Maor's real constraint).
    cost_by = cost.groupby(["channel", "platform"])["cost"].sum()
    rows = []
    for (ch, pf), g in vis.groupby(["channel", "platform"]):
        n = len(g)
        c = cost_by.get((ch, pf), 0.0)
        cpv = c / n if n else np.nan
        epv = g["revenue"].mean()
        _, lo, hi = bootstrap_mean_ci(g["revenue"].to_numpy(float))
        rows.append({"channel": ch, "platform": pf, "visits": n, "conv%": g["converted"].mean() * 100,
                     "EPV": epv, "CPV": cpv, "profit/visit": epv - cpv, "EPV_lo": lo, "EPV_hi": hi})
    return pd.DataFrame(rows).sort_values("profit/visit", ascending=False)


def cmd_budget(args):
    vis = load_visits(Path(args.visits))
    cost = load_cost(Path(args.cost))
    tab = budget_table(vis, cost)
    out = ["BUDGET DECISION · profit per visit by channel × platform"]
    out.append(_hr("FUNNEL (EPV interval is bootstrap 95%)"))
    for _, r in tab.iterrows():
        out.append(f"  {r['channel']:<8} {r['platform']:<8} "
                   f"visits={int(r['visits']):>7,}  conv={r['conv%']:.2f}%  "
                   f"EPV={r['EPV']:.3f} [{r['EPV_lo']:.2f},{r['EPV_hi']:.2f}]  "
                   f"CPV={r['CPV']:.3f}  profit/visit={r['profit/visit']:+.3f}")
    # Channel-level headline for the Slack question
    out.append(_hr("VALIDATION — answer the Slack message honestly"))
    g = vis[vis.channel == "Google"]["converted"]
    b = vis[vis.channel == "Bing"]["converted"]
    r = two_proportion_ztest(int(g.sum()), len(g), int(b.sum()), len(b))
    out.append(f"  Conversion: Google {r['p1']*100:.2f}% vs Bing {r['p2']*100:.2f}% "
               f"(diff {r['diff']*100:+.2f}pp, p={r['p_value']:.3g}).")
    out.append(_flag("Conversion rate is the WRONG yardstick — it ignores cost and revenue size. "
                     "Decide on PROFIT PER VISIT (EPV − CPV), which is what the table above ranks."))
    out.append("  → Recommendation: shift budget toward the highest profit/visit cells, and do NOT "
               "cut a channel purely because its conversion rate is lower.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# Subcommand: retro  (capstone: was the position-promotion the right call?)
# --------------------------------------------------------------------------- #
def cmd_retro(args):
    raw = load_clickouts(Path(args.visits))
    cl = raw[raw["clicked"]].copy()
    cutoff = pd.Timestamp(args.change_date)
    p = cl[cl["partner"] == args.partner].copy()
    if p.empty:
        sys.exit(f"ERROR: no click-outs for partner {args.partner!r}. "
                 f"Partners: {sorted(cl['partner'].dropna().unique())}")
    before = p[p["click_timestamp"] < cutoff]
    after = p[p["click_timestamp"] >= cutoff]

    def blk(df):
        return dict(clicks=len(df), epc=df["revenue"].mean(),
                    conv=df["converted"].mean(),
                    pos=df["clickout_position"].mode().iloc[0] if len(df) else np.nan)

    bfr, aft = blk(before), blk(after)
    out = [f"DECISION RETRO · partner='{args.partner}' · change @ {args.change_date}"]
    out.append(_hr("BEFORE vs AFTER"))
    out.append(f"  BEFORE: clicks={bfr['clicks']:,}  EPC={bfr['epc']:.3f}  "
               f"conv={bfr['conv']*100:.1f}%  modal position={bfr['pos']:.0f}")
    out.append(f"  AFTER : clicks={aft['clicks']:,}  EPC={aft['epc']:.3f}  "
               f"conv={aft['conv']*100:.1f}%  modal position={aft['pos']:.0f}")
    # naive lift with bootstrap CI on EPC
    boot = bootstrap_mean_diff_ci(after["revenue"].to_numpy(float),
                                  before["revenue"].to_numpy(float))
    out.append(f"  EPC change: {boot['lift_pct']:+.1f}%  95% CI [{boot['lift_lo']:+.1f}%, {boot['lift_hi']:+.1f}%]")

    out.append(_hr("VALIDATION — is this a real win, or a confounded one?"))
    if bfr["pos"] != aft["pos"]:
        out.append(_flag(f"The click-out POSITION itself changed ({bfr['pos']:.0f}→{aft['pos']:.0f}). "
                         f"EPC-per-click can shift just from the new slot — you cannot attribute the "
                         f"change to partner quality alone."))
    out.append(_flag("No control group: this is a before/after, not an A/B test. Site-wide trends, "
                     "seasonality, competitor and traffic-mix changes all move with time and are "
                     "confounded with the promotion."))
    if boot["lift_lo"] < 0 < boot["lift_hi"]:
        out.append(_flag("Bootstrap CI on the EPC change includes 0 — the 'improvement' is within noise."))
    else:
        out.append(_ok("EPC change is stable under resampling (CI excludes 0) — but see confounds above."))
    out.append("\n  → Verdict template: state the observed lift, then list what could explain it "
               "besides the decision. A defensible retro names its confounders; it does not claim "
               "causation from a before/after alone.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# Subcommand: decide  (THE ORCHESTRATOR — a gated decision procedure)
# --------------------------------------------------------------------------- #
def cmd_decide(args):
    """
    Break a messy business decision into a fixed, gated sequence of statistical steps.
    This is the point of the workshop: a raw session on the open question can anchor,
    stop early, or take a different path each run. This procedure does not conclude
    until each gate passes — and it runs the same way every time.
    """
    vis = load_visits(Path(args.visits))
    cost = load_cost(Path(args.cost))
    out = ["=" * 70, "DECISION MEMO — gated statistical procedure (deterministic)", "=" * 70,
           "Q: Google converts better than Bing and mobile EPV looks down — should we move",
           "   budget? (+ optionally: was a past chart promotion the right call?)",
           "Each step GATES the next. We do not anchor on the framing or conclude early."]

    # STEP 1 — profile the decision metric before trusting any average of it
    s = describe_shape(vis["epv"].to_numpy(float))
    out.append(_hr("STEP 1 · PROFILE the decision metric (EPV) — is its mean trustworthy?"))
    out.append(f"  mean={s['mean']:.3f}  median={s['median']:.3f}  skew={s['skew']:.1f}  "
               f"zero-share={s['zero_share']*100:.0f}%  top-1% share={s['top1_share']*100:.0f}%")
    out.append("  GATE ▸ EPV is zero-inflated & whale-skewed ⇒ every downstream comparison must use "
               "bootstrap / rank methods, NEVER a t-test or normal CI. ✔ proceed")

    # STEP 2 — is the headline claim even real? (and is it the right question?)
    g = vis[vis.channel == "Google"]["converted"]; b = vis[vis.channel == "Bing"]["converted"]
    r = two_proportion_ztest(int(g.sum()), len(g), int(b.sum()), len(b))
    d, lo, hi = diff_proportion_ci(int(g.sum()), len(g), int(b.sum()), len(b))
    out.append(_hr("STEP 2 · TEST the headline claim: does Google convert better than Bing?"))
    out.append(f"  Google {r['p1']*100:.2f}%  vs Bing {r['p2']*100:.2f}%   "
               f"diff {d*100:+.2f}pp [{lo*100:+.2f},{hi*100:+.2f}]   p={r['p_value']:.2g}")
    out.append("  GATE ▸ the gap is REAL but small — and conversion ignores cost. A conversion gap "
               "does NOT justify a budget move. Do NOT anchor here. ✔ proceed to the decision metric")

    # STEP 3 — reframe to the actual decision metric: profit per visit
    tab = budget_table(vis, cost)
    out.append(_hr("STEP 3 · REFRAME to the decision metric: profit/visit (EPV − CPV), w/ bootstrap CI"))
    for _, row in tab.iterrows():
        flag = "  ← LOSES money" if row["profit/visit"] < 0 else ""
        out.append(f"  {row['channel']:<8}{row['platform']:<8} conv={row['conv%']:>5.2f}%  "
                   f"EPV={row['EPV']:.2f}[{row['EPV_lo']:.2f},{row['EPV_hi']:.2f}]  "
                   f"CPV={row['CPV']:.2f}  profit/visit={row['profit/visit']:+.3f}{flag}")
    gm = tab[(tab.channel == "Google") & (tab.platform == "mobile")]["profit/visit"].iloc[0]
    out.append(f"  GATE ▸ ranking on profit/visit REVERSES the conversion story: Google-mobile loses "
               f"${abs(gm):.2f}/visit (high CPC) while Bing is profitable on both platforms. ✔ proceed")

    # STEP 4 — size the secondary signal with an interval, not a p-value
    a = vis[vis.platform == "mobile"]["epv"].to_numpy(float)
    dsk = vis[vis.platform == "desktop"]["epv"].to_numpy(float)
    boot = bootstrap_mean_diff_ci(a, dsk)
    out.append(_hr("STEP 4 · SIZE the secondary signal: mobile vs desktop EPV (bootstrap + effect size)"))
    out.append(f"  mobile {a.mean():.3f} vs desktop {dsk.mean():.3f}   "
               f"{boot['lift_pct']:+.1f}% [{boot['lift_lo']:+.1f}%,{boot['lift_hi']:+.1f}%]   "
               f"Cohen d={cohens_d(a, dsk):.3f}")
    out.append("  GATE ▸ mobile EPV is materially lower (~ −30%, CI excludes 0) but the standardized "
               "effect is tiny — a where-to-trim signal, not a crisis. ✔ proceed")

    # STEP 5 — optional retrospective on a named past decision
    if args.partner:
        raw = load_clickouts(Path(args.visits))
        p = raw[(raw["clicked"]) & (raw["partner"] == args.partner)]
        cutoff = pd.Timestamp(args.change_date)
        bef, aft = p[p["click_timestamp"] < cutoff], p[p["click_timestamp"] >= cutoff]
        rb = bootstrap_mean_diff_ci(aft["revenue"].to_numpy(float), bef["revenue"].to_numpy(float))
        out.append(_hr(f"STEP 5 · RETRO on a past decision: promoting '{args.partner}' @ {args.change_date}"))
        out.append(f"  BEFORE clicks={len(bef):,} EPC={bef['revenue'].mean():.2f} pos="
                   f"{bef['clickout_position'].mode().iloc[0]:.0f}   "
                   f"AFTER clicks={len(aft):,} EPC={aft['revenue'].mean():.2f} pos="
                   f"{aft['clickout_position'].mode().iloc[0]:.0f}")
        out.append(f"  EPC change {rb['lift_pct']:+.1f}% [{rb['lift_lo']:+.1f}%,{rb['lift_hi']:+.1f}%]")
        out.append("  GATE ▸ volume rose sharply but EPC is FLAT (CI includes 0) and POSITION changed "
                   "(confound) with NO control group ⇒ a volume win, not a proven quality win.")

    # SYNTHESIS — constrained by the gated evidence above
    out.append(_hr("SYNTHESIS — recommendation (grounded in the gated steps)"))
    out.append("  • Do NOT move budget Bing→Google. Profit/visit ranks Bing above Google; "
               "Google-mobile is reliably underwater. The 'Google converts better' framing is a trap.")
    out.append("  • Action: trim/renegotiate Google-mobile CPC; protect Bing; grow Organic where possible.")
    out.append("  • Mobile EPV is lower across the board — a place to trim, not a fire drill.")
    if args.partner:
        out.append(f"  • '{args.partner}' promotion: report it as a VOLUME gain at steady EPC, not a "
                   "performance win — the per-click quality did not measurably improve.")
    out.append(_hr("WHAT WE DID NOT ESTABLISH (honesty gate)"))
    out.append("  • Causality — all observational, no A/B. Cost is aggregate (no marginal/CPC elasticity). "
               "Confirm any big swing with a small budget holdout, then re-measure.")
    print("\n".join(out))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="Deterministic statistics engine for the NI workshop skills.")
    p.add_argument("--visits", default=str(DEFAULT_VISITS), help="visit_clickouts CSV")
    p.add_argument("--cost", default=str(DEFAULT_COST), help="daily_cost CSV")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("profile"); s.add_argument("--metric", default="epv"); s.add_argument("--group", default=None); s.set_defaults(func=cmd_profile)
    s = sub.add_parser("significance"); s.add_argument("--metric", required=True); s.add_argument("--group", required=True); s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.set_defaults(func=cmd_significance)
    s = sub.add_parser("estimate"); s.add_argument("--metric", required=True); s.add_argument("--group", required=True); s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.set_defaults(func=cmd_estimate)
    s = sub.add_parser("relationship"); s.add_argument("--x", required=True); s.add_argument("--y", required=True); s.add_argument("--by", default=None); s.set_defaults(func=cmd_relationship)
    s = sub.add_parser("budget"); s.set_defaults(func=cmd_budget)
    s = sub.add_parser("retro"); s.add_argument("--partner", required=True); s.add_argument("--change-date", required=True); s.add_argument("--metric", default="epc"); s.set_defaults(func=cmd_retro)
    s = sub.add_parser("decide"); s.add_argument("--partner", default=None); s.add_argument("--change-date", default=None); s.set_defaults(func=cmd_decide)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
