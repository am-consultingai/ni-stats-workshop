#!/usr/bin/env python3
"""relationship.py — Module 3 (confound). Measure a relationship with BOTH
Pearson and Spearman, flag when they disagree, and run a Simpson's / confound
check within a chosen dimension.

    python relationship.py --x clickout_position --y revenue --by segment
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np
import pandas as pd
from scipy import stats


def run(args):
    raw = C.load_clickouts(Path(args.visits))
    df = raw[raw["clicked"]] if "clickout_position" in (args.x, args.y) else raw
    df = C.apply_slice(df, args.slice)
    x = pd.to_numeric(df[args.x], errors="coerce")
    y = pd.to_numeric(df[args.y], errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    out = [f"RELATIONSHIP · {args.x} vs {args.y}  (n={len(x):,})"]
    out.append(C.hr("CORRELATION"))
    out.append(f"  Pearson  r = {pear.statistic:+.3f}  (p={pear.pvalue:.3g})  ← linear, outlier-sensitive")
    out.append(f"  Spearman ρ = {spear.statistic:+.3f}  (p={spear.pvalue:.3g})  ← monotonic, robust")
    out.append(C.hr("VALIDATION"))
    if abs(pear.statistic - spear.statistic) > 0.1:
        out.append(C.flag(f"Pearson and Spearman disagree ({pear.statistic:+.3f} vs {spear.statistic:+.3f}) — "
                          f"outliers or non-linearity are bending Pearson; trust Spearman."))
    else:
        out.append(C.ok("Pearson ≈ Spearman; roughly linear and monotonic."))

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
        out.append(C.hr(f"CONFOUND CHECK — does it hold within each '{args.by}'?"))
        for gname, rho in within:
            mark = "  ⟲ SIGN FLIPS" if (np.sign(rho) != np.sign(overall) and abs(rho) > 0.05) else ""
            out.append(f"    {args.by}={gname}: ρ={rho:+.3f}{mark}")
        if flips:
            out.append(C.flag(f"Simpson's paradox risk: the overall correlation ({overall:+.3f}) reverses in "
                              f"{len(flips)} subgroup(s) — a MIX artifact, not a real driver. Report the "
                              f"within-group story."))
        else:
            out.append(C.ok(f"Sign is stable within every '{args.by}' — survives the confound check."))
    else:
        out.append("\n  (Tip: pass --by <dimension> to test for a confound / Simpson's paradox.)")
    out.append("\n  ⚠️  Correlation ≠ causation. Every correlation is a QUESTION, not a conclusion.")
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--by", default=None)
    p.add_argument("--slice", action="append", default=[])
    run(p.parse_args())
