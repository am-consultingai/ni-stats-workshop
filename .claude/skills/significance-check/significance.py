#!/usr/bin/env python3
"""significance.py — Module 3. Is a difference between two groups REAL, or an
artifact of thin data / skew / clustering? Auto-selects the right test for the
shape and, at click grain, uses the CLUSTER bootstrap.

    # the drill-down: is partner B really better than A in Bing?
    python significance.py --metric revenue --group partner \
        --a "Summit Direct Business" --b "Cedar Business Bank" \
        --grain click --slice channel=Bing --cluster
    # a rate at visit grain
    python significance.py --metric converted --group channel --a Google --b Bing
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np
from scipy import stats


def run(args):
    if args.grain == "click":
        df = C.load_clickouts(Path(args.visits))
        df = df[df["clicked"]]
        metric = "revenue" if args.metric in ("epv", "revenue") else args.metric
    else:
        df = C.load_visits(Path(args.visits))
        metric = args.metric
    df = C.apply_slice(df, args.slice)
    A = df[df[args.group].astype(str) == args.a]
    B = df[df[args.group].astype(str) == args.b]
    a = A[metric].to_numpy(float)
    b = B[metric].to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        sys.exit(f"ERROR: no rows for {args.group}={args.a!r} ({len(a)}) or {args.b!r} ({len(b)}).")

    out = [f"SIGNIFICANCE · '{metric}' · {args.group}: {args.a} vs {args.b}"
           + (f" · slice={args.slice}" if args.slice else "")]
    binary = set(np.unique(np.concatenate([a, b]))).issubset({0.0, 1.0})

    if binary:
        k1, n1, k2, n2 = int(a.sum()), len(a), int(b.sum()), len(b)
        r = C.two_proportion_ztest(k1, n1, k2, n2)
        d, lo, hi = C.diff_proportion_ci(k1, n1, k2, n2)
        pw = C.power_two_proportions(r["p1"], r["p2"], min(n1, n2))
        out.append(C.hr("TEST CHOSEN: two-proportion z-test (binary metric)"))
        out.append(f"  {args.a}: {r['p1']*100:.2f}% (n={n1:,})   {args.b}: {r['p2']*100:.2f}% (n={n2:,})")
        out.append(f"  diff = {d*100:+.2f} pp   95% CI [{lo*100:+.2f}, {hi*100:+.2f}] pp   "
                   f"rel lift {(r['p1']/r['p2']-1)*100:+.1f}%")
        out.append(f"  z = {r['z']:.2f}   p = {r['p_value']:.4g}")
        out.append(C.hr("VALIDATION"))
        out.append((C.ok if r["p_value"] < 0.05 else C.flag)(
            f"p={r['p_value']:.4g} → {'significant' if r['p_value']<0.05 else 'NOT significant'} at α=0.05."))
        if lo < 0 < hi:
            out.append(C.flag("95% CI for the difference straddles 0 — direction itself is uncertain."))
        out.append((C.ok if pw >= 0.8 else C.flag)(
            f"Approx power ≈ {pw*100:.0f}%. {'Adequate.' if pw>=0.8 else 'Underpowered — a null may be too little data.'}"))
        out.append("  ⚠️  n-trap: significance scales with n. Lead with the effect size + CI, not p.")
    else:
        sa, sb = C.describe_shape(a), C.describe_shape(b)
        skewed = (abs(sa["skew"]) > 1) or (abs(sb["skew"]) > 1)
        t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        clustered = args.cluster and args.grain == "click"
        if clustered:
            boot = C.cluster_bootstrap_diff_ci(a, A["visit_iid"].to_numpy(), b, B["visit_iid"].to_numpy())
            method = "Mann-Whitney U + CLUSTER bootstrap (resampling visits)"
        else:
            boot = C.bootstrap_mean_diff_ci(a, b)
            method = "Mann-Whitney U + bootstrap" if skewed else "Welch's t-test"
        out.append(C.hr(f"TEST CHOSEN: {method}"))
        out.append(f"  {args.a}: mean={a.mean():.3f} median={np.median(a):.3f} skew={sa['skew']:.1f} (n={len(a):,})")
        out.append(f"  {args.b}: mean={b.mean():.3f} median={np.median(b):.3f} skew={sb['skew']:.1f} (n={len(b):,})")
        out.append(f"  mean diff = {boot['diff']:+.3f}   95% CI [{boot['diff_lo']:+.3f}, {boot['diff_hi']:+.3f}]")
        out.append(f"  Welch t p = {t_p:.4g}   Mann-Whitney p = {u_p:.4g}   Cohen d = {C.cohens_d(a, b):.3f}")
        out.append(C.hr("VALIDATION"))
        if skewed:
            out.append(C.flag(f"Heavy-tailed (skew {sa['skew']:.1f}/{sb['skew']:.1f}) → t-test invalid. Primary "
                              f"test is Mann-Whitney; the interval is a BOOTSTRAP. If a tool gave only a "
                              f"t-test p here, that is the #1 pitfall."))
        if clustered:
            de = C.design_effect(df[df[args.group].astype(str).isin([args.a, args.b])][metric].to_numpy(float),
                                 df[df[args.group].astype(str).isin([args.a, args.b])]["visit_iid"].to_numpy())
            out.append(C.ok(f"Clustered CI used (deff≈{de['deff']:.2f}); rows share visits, so the row-level "
                            f"interval would be too narrow."))
        if boot["diff_lo"] < 0 < boot["diff_hi"]:
            out.append(C.flag("Bootstrap CI for the difference INCLUDES 0 — not a reliable difference. "
                              "On a thin slice this is the honest verdict: coin-flip."))
        else:
            out.append(C.ok("Bootstrap CI excludes 0 — the difference is stable under resampling."))
        out.append("  ⚠️  n-trap: with large n almost any gap is 'significant'. Lead with effect size + CI.")
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--metric", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--grain", choices=["visit", "click"], default="visit")
    p.add_argument("--slice", action="append", default=[])
    p.add_argument("--cluster", action="store_true")
    run(p.parse_args())
