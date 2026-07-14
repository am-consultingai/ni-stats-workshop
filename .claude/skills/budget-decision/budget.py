#!/usr/bin/env python3
"""budget.py — Module 6. Answer "should we move budget?" the right way: build
the funnel by channel × platform, join the aggregate cost table to get CPV, and
rank by PROFIT PER VISIT (EPV − CPV), not the misleading conversion rate.

    python budget.py
    python budget.py --by channel platform
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C


def run(args):
    vis = C.load_visits(Path(args.visits))
    cost = C.load_cost(Path(args.cost))
    tab = C.budget_table(vis, cost, by=tuple(args.by))
    out = [f"BUDGET DECISION · profit per visit by {' × '.join(args.by)}"]
    out.append(C.hr("FUNNEL (EPV interval is bootstrap 95%)"))
    for _, r in tab.iterrows():
        dims = "  ".join(f"{r[d]}" for d in args.by)
        flag = "  ← LOSES money" if r["profit/visit"] < 0 else ""
        out.append(f"  {dims:<20} visits={int(r['visits']):>7,}  conv={r['conv%']:.2f}%  "
                   f"EPV={r['EPV']:.3f}[{r['EPV_lo']:.2f},{r['EPV_hi']:.2f}]  "
                   f"CPV={r['CPV']:.3f}  profit/visit={r['profit/visit']:+.3f}{flag}")

    out.append(C.hr("VALIDATION — answer honestly"))
    if {"channel"}.issubset(vis.columns):
        g = vis[vis.channel == "Google"]["converted"]
        b = vis[vis.channel == "Bing"]["converted"]
        r = C.two_proportion_ztest(int(g.sum()), len(g), int(b.sum()), len(b))
        out.append(f"  Conversion: Google {r['p1']*100:.2f}% vs Bing {r['p2']*100:.2f}% "
                   f"(diff {r['diff']*100:+.2f}pp, p={r['p_value']:.3g}).")
    out.append(C.flag("Conversion rate is the WRONG yardstick — it ignores cost and revenue size. Decide on "
                      "PROFIT PER VISIT (EPV − CPV), which the table ranks."))
    out.append("  ⚠️  Cost is aggregate (no per-user cost) → CPV is an average; you cannot attribute cost to "
               "a single visit. Confirm any big reallocation with a small holdout before scaling.")
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--cost", default=str(C.DEFAULT_COST))
    p.add_argument("--by", nargs="+", default=["channel", "platform"])
    run(p.parse_args())
