#!/usr/bin/env python3
"""estimate.py — how big + how sure. A point estimate with the RIGHT 95%
interval (Wilson for rates, bootstrap for skewed revenue/EPV), plus width-vs-n
intuition and three-audience phrasing.

    python estimate.py --metric converted --group channel --a Google --b Bing
    python estimate.py --metric epv --group platform --a mobile --b desktop
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np


def run(args):
    vis = C.load_visits(Path(args.visits))
    vis = C.apply_slice(vis, args.slice)
    a = vis.loc[vis[args.group].astype(str) == args.a, args.metric].to_numpy(float)
    b = vis.loc[vis[args.group].astype(str) == args.b, args.metric].to_numpy(float)
    out = [f"ESTIMATE · '{args.metric}' · {args.group}: {args.a} vs {args.b}"]
    binary = set(np.unique(np.concatenate([a, b]))).issubset({0.0, 1.0})

    if binary:
        k1, n1, k2, n2 = int(a.sum()), len(a), int(b.sum()), len(b)
        p1, l1, h1 = C.wilson_ci(k1, n1)
        p2, l2, h2 = C.wilson_ci(k2, n2)
        d, lo, hi = C.diff_proportion_ci(k1, n1, k2, n2)
        out.append(C.hr("POINT ESTIMATE + 95% INTERVAL (Wilson for rates)"))
        out.append(f"  {args.a}: {p1*100:.2f}%  [{l1*100:.2f}, {h1*100:.2f}]")
        out.append(f"  {args.b}: {p2*100:.2f}%  [{l2*100:.2f}, {h2*100:.2f}]")
        out.append(f"  difference: {d*100:+.2f} pp  [{lo*100:+.2f}, {hi*100:+.2f}]")
        width = (hi - lo) * 100
    else:
        ma, la, ha = C.bootstrap_mean_ci(a)
        mb, lb, hb = C.bootstrap_mean_ci(b)
        boot = C.bootstrap_mean_diff_ci(a, b)
        out.append(C.hr("POINT ESTIMATE + 95% INTERVAL (bootstrap — metric is skewed)"))
        out.append(f"  {args.a}: {ma:.3f}  [{la:.3f}, {ha:.3f}]")
        out.append(f"  {args.b}: {mb:.3f}  [{lb:.3f}, {hb:.3f}]")
        out.append(f"  difference: {boot['diff']:+.3f}  [{boot['diff_lo']:+.3f}, {boot['diff_hi']:+.3f}]")
        out.append(f"  relative:   {boot['lift_pct']:+.1f}%  [{boot['lift_lo']:+.1f}%, {boot['lift_hi']:+.1f}%]")
        width = boot["diff_hi"] - boot["diff_lo"]

    out.append(C.hr("SAY IT THREE WAYS"))
    out.append("  • Executive: the point estimate with a plain-English range.")
    out.append("  • Analyst:   the 95% interval + the method (Wilson / bootstrap).")
    out.append("  • Skeptic:   'the interval is wide because n is small' — width shrinks ~1/√n.")
    out.append(C.hr("VALIDATION"))
    out.append(C.flag("Lead with the INTERVAL, not the point. A single number with no interval hides the "
                      "uncertainty — ask for the CI."))
    out.append(f"  Interval width ≈ {width:.3f}. To halve it you need ~4× the data.")
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--metric", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--slice", action="append", default=[])
    run(p.parse_args())
