#!/usr/bin/env python3
"""profile.py — Module 1 + 2. Profile a metric's distribution AND how much
independent data is really in the slice (effective-n). Run this FIRST, before
trusting any average or test.

    python profile.py --metric revenue --grain click --slice channel=Bing
    python profile.py --metric epv --group platform
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np


def run(args):
    if args.grain == "click":
        df = C.load_clickouts(Path(args.visits))
        df = df[df["clicked"]]
        col = "revenue" if args.metric in ("epv", "revenue") else args.metric
    else:
        df = C.load_visits(Path(args.visits))
        col = args.metric
    df = C.apply_slice(df, args.slice)
    out = [f"PROFILE · metric='{col}' · grain={args.grain}"
           + (f" · slice={args.slice}" if args.slice else "")
           + (f" · grouped by '{args.group}'" if args.group else "")]

    groups = [(None, df)] if not args.group else list(df.groupby(args.group))
    for gname, g in groups:
        s = C.describe_shape(g[col].to_numpy(float))
        head = f"{args.group}={gname}" if gname is not None else "whole slice"
        out.append(C.hr(head))
        out.append(f"  n={s['n']:,}  mean={s['mean']:.3f}  median={s['median']:.3f}  "
                   f"trimmed(10%)={s['trimmed_mean_10']:.3f}")
        out.append(f"  skew={s['skew']:.1f}  zero-share={s['zero_share']*100:.1f}%  "
                   f"top-1% share={s['top1_share']*100:.1f}%  max={s['max']:.1f}")

    s = C.describe_shape(df[col].to_numpy(float))
    out.append(C.hr("VALIDATION — can you trust an average of this?"))
    if abs(s["skew"]) > 1:
        out.append(C.flag(f"Heavy skew (skew={s['skew']:.1f}). The MEAN is not a typical value. "
                          f"Do NOT use a t-test / normal CI — use bootstrap or rank methods "
                          f"(see /significance-check)."))
    else:
        out.append(C.ok(f"Skew is mild ({s['skew']:.1f}); the mean is a reasonable summary."))
    if s["top1_share"] > 0.10:
        out.append(C.flag(f"Whale-driven: the top 1% of rows hold {s['top1_share']*100:.0f}% of all {col}. "
                          f"A few conversions can invent a difference — plot the distribution first."))
    if s["zero_share"] > 0.5:
        out.append(C.flag(f"{s['zero_share']*100:.0f}% of rows are exactly 0 (zero-inflated); the mean "
                          f"blends 'did it convert' with 'how much'."))

    # Effective-n: the i.i.d. gate (Module 2)
    out.append(C.hr("EFFECTIVE-N — is every row independent?"))
    if args.grain == "click":
        de = C.design_effect(df[col].to_numpy(float), df["visit_iid"].to_numpy())
        out.append(f"  rows={de['n']:,}  visits(clusters)={de['k']:,}  "
                   f"mean rows/visit={de['mbar']:.2f}  ICC={de['icc']:.3f}")
        out.append(f"  design effect deff={de['deff']:.3f}  →  effective n ≈ {de['n_eff']:,.0f} "
                   f"({de['n_eff']/de['n']*100:.0f}% of nominal)")
        if de["deff"] > 1.05:
            out.append(C.flag(f"Click-outs share visits → NOT i.i.d. Treat n as ~{de['n_eff']:,.0f}, "
                              f"and use the CLUSTER bootstrap (resample visits, not rows) for any CI."))
        else:
            out.append(C.ok("Clustering is negligible here; row-level inference is fine."))
    else:
        out.append(C.ok("Visit grain: one independent row per visit → i.i.d. holds; deff≈1."))
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--metric", default="revenue")
    p.add_argument("--grain", choices=["visit", "click"], default="visit")
    p.add_argument("--group", default=None)
    p.add_argument("--slice", action="append", default=[])
    run(p.parse_args())
