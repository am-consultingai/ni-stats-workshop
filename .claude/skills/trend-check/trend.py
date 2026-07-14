#!/usr/bin/env python3
"""trend.py — Module 5. "CTR was down yesterday — is that meaningful?" Put a
day-over-day move in perspective: compare the latest day against the month-wide
band and its own day-of-week baseline, and run a light regime-break test.

    python trend.py --metric ctr --slice channel=Bing
    python trend.py --metric epv
    python trend.py --metric epc --slice partner="Summit Direct"
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np


def run(args):
    clk = C.load_clickouts(Path(args.visits))
    if args.metric == "ctr":
        df, col, how, label = clk, "clicked_i", "rate", "CTR"
        df["clicked_i"] = df["clicked"].astype(int)
    elif args.metric == "epc":
        df, col, how, label = clk[clk["clicked"]].copy(), "revenue", "mean", "EPC"
    elif args.metric in ("epv", "revenue"):
        df, col, how, label = C.load_visits(Path(args.visits)), "revenue", "mean", "EPV"
    else:
        df, col, how, label = C.load_visits(Path(args.visits)), args.metric, "mean", args.metric
    df = C.apply_slice(df, args.slice)

    t = C.trend_check(df, col, how=how, recent_days=args.recent_days)
    out = [f"TREND-CHECK · {label}" + (f" · slice={args.slice}" if args.slice else "")]
    if t.get("insufficient"):
        out.append(C.hr("VALIDATION"))
        out.append(C.flag(f"Only {t['n_days']} day(s) of data — cannot judge a trend."))
        C.emit(out); return

    scale = 100 if how == "rate" else 1
    unit = "%" if how == "rate" else ""
    out.append(C.hr(f"LATEST DAY vs NORMAL VARIATION"))
    out.append(f"  latest ({t['latest_date']}, {t['dow']}): {t['latest']*scale:.2f}{unit}")
    out.append(f"  month mean: {t['month_mean']*scale:.2f}{unit}   "
               f"normal band (±2σ): [{t['band_lo']*scale:.2f}, {t['band_hi']*scale:.2f}]{unit}")
    if np.isfinite(t["dow_mean"]):
        out.append(f"  {t['dow']} baseline: {t['dow_mean']*scale:.2f}{unit} "
                   f"(±2σ = ±{2*t['dow_std']*scale:.2f})")
    out.append(f"  last {args.recent_days}d mean {t['recent_mean']*scale:.2f} vs prior "
               f"{t['prior_mean']*scale:.2f}{unit}  (regime z={t['regime_z']:.2f})")

    out.append(C.hr("VALIDATION — is the move real?"))
    if t["within_band"]:
        out.append(C.ok(f"Latest is INSIDE the normal monthly band — nothing to react to; this is "
                        f"ordinary day-to-day variation."))
    else:
        out.append(C.flag("Latest is OUTSIDE the ±2σ monthly band — a genuine outlier day, worth a look."))
    if t["dow_within"] is False:
        out.append(C.flag(f"Also unusual for a {t['dow']} specifically (day-of-week adjusted)."))
    elif t["dow_within"]:
        out.append(C.ok(f"Normal for a {t['dow']} — remember weekdays differ; don't compare across days."))
    if t["regime_break"]:
        out.append(C.flag(f"REGIME BREAK: the last {args.recent_days} days differ from the prior period "
                          f"(z={t['regime_z']:.1f}). A historical prior may no longer be valid — "
                          f"treat /bayesian-update results with caution."))
    else:
        out.append(C.ok("No regime break: recent period is consistent with history — a prior is safe to use."))
    out.append("  ⚠️  One low day is not a trend. Always place it against seasonality and the monthly band "
               "before escalating.")
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--metric", default="ctr")
    p.add_argument("--slice", action="append", default=[])
    p.add_argument("--recent-days", dest="recent_days", type=int, default=7)
    run(p.parse_args())
