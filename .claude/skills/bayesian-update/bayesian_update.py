#!/usr/bin/env python3
"""bayesian_update.py — Module 4: BAYESIAN INFERENCE. When a slice is too thin to
call (significance = coin-flip), don't stop at "we can't tell" — bring in an
informative PRIOR from history (the partner's behaviour elsewhere / earlier),
combine it with the thin-slice likelihood via a conjugate normal update, and
report the POSTERIOR, P(better) and expected loss — a decision, not a binary
p-value. (Anchoring on a prior this way is the sense in which we "borrow
strength" — partial pooling — but the method is Bayesian inference.)

    python bayesian_update.py --group partner \
        --a "Summit Direct Business" --b "Cedar Business Bank" \
        --slice channel=Bing --prior other-slice --grain click

Prior sources:
    other-slice  : same partner, rows OUTSIDE the slice (e.g. other channels)
    prior-month  : same partner+slice, click_timestamp before --before
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np
import pandas as pd


def _prior_rows(full, partner, args):
    sub = full[full[args.group].astype(str) == partner]
    if args.prior == "prior-month":
        sub = C.apply_slice(sub, args.slice)
        return sub[sub["click_timestamp"] < pd.Timestamp(args.before)]
    # other-slice: complement of the slice filter, within the same partner
    for spec in args.slice:
        col, _, val = spec.partition("=")
        sub = sub[sub[col].astype(str) != val]
    return sub


def _estimate(vals, clusters, grain):
    m, se2, n = C.mean_and_se2(vals)
    if grain == "click" and n > 1 and len(vals):
        de = C.design_effect(vals, clusters)      # clustering widens the likelihood
        se2 = se2 * de["deff"]
    return m, se2, n


def run(args):
    raw = C.load_clickouts(Path(args.visits))
    full = raw[raw["clicked"]] if args.grain == "click" else C.load_visits(Path(args.visits))
    metric = "revenue" if args.metric in ("epv", "revenue") else args.metric
    slice_df = C.apply_slice(full, args.slice)

    out = [f"BAYESIAN INFERENCE · '{metric}' · {args.a} vs {args.b} · slice={args.slice} · prior={args.prior}"]
    post = {}
    for who in (args.a, args.b):
        s = slice_df[slice_df[args.group].astype(str) == who]
        pri = _prior_rows(full, who, args)
        dm, dse2, dn = _estimate(s[metric].to_numpy(float), s["visit_iid"].to_numpy() if args.grain=="click" else s.index.to_numpy(), args.grain)
        pm, pse2, pn = _estimate(pri[metric].to_numpy(float), pri["visit_iid"].to_numpy() if args.grain=="click" else pri.index.to_numpy(), args.grain)
        po = C.posterior_from_prior(pm, pse2, dm, dse2)
        post[who] = dict(dm=dm, dn=dn, pm=pm, pn=pn, po=po)
        out.append(C.hr(who))
        out.append(f"  thin slice : {dm:.2f}  (n={dn:,})")
        out.append(f"  prior      : {pm:.2f}  (n={pn:,})  ← {args.prior}")
        out.append(f"  POSTERIOR  : {po['mean']:.2f}  95% CI [{po['lo']:.2f}, {po['hi']:.2f}]")

    A, B = post[args.a], post[args.b]
    raw_better = C.p_better(A["dm"], max(A["po"]["var"], 1e-9), B["dm"], max(B["po"]["var"], 1e-9))
    pb = C.p_better(A["po"]["mean"], A["po"]["var"], B["po"]["mean"], B["po"]["var"])
    loss = C.expected_loss_choosing(A["po"]["mean"], A["po"]["var"], B["po"]["mean"], B["po"]["var"])
    out.append(C.hr("DECISION (posterior)"))
    out.append(f"  raw slice ordering : {args.a} {'>' if A['dm']>B['dm'] else '<'} {args.b}  "
               f"({A['dm']:.2f} vs {B['dm']:.2f})")
    out.append(f"  posterior ordering : {args.a} {'>' if A['po']['mean']>B['po']['mean'] else '<'} {args.b}  "
               f"({A['po']['mean']:.2f} vs {B['po']['mean']:.2f})")
    out.append(f"  P({args.a} > {args.b}) = {pb*100:.0f}%   "
               f"expected loss if we pick {args.a}: {loss['choose_a']:.3f}/click")

    out.append(C.hr("VALIDATION"))
    flipped = (A["dm"] > B["dm"]) != (A["po"]["mean"] > B["po"]["mean"])
    if flipped:
        out.append(C.flag(f"The slice ordering REVERSES once anchored on history — the apparent win was "
                          f"regression to the mean. Do NOT reorder on this slice alone."))
    elif 0.4 < pb < 0.6:
        out.append(C.flag(f"Even with the prior, P(better) ≈ {pb*100:.0f}% — still a coin-flip. Hold."))
    else:
        out.append(C.ok(f"History REINFORCES the slice: P(better) = {pb*100:.0f}%. A defensible case to act."))
    out.append("  ⚠️  A prior only helps if the past is representative (stationarity). If a regime break is "
               "suspected, validate with /trend-check before trusting it — else the prior biases you.")
    C.emit(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--metric", default="revenue")
    p.add_argument("--group", default="partner")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--grain", choices=["visit", "click"], default="click")
    p.add_argument("--slice", action="append", default=[])
    p.add_argument("--prior", choices=["other-slice", "prior-month"], default="other-slice")
    p.add_argument("--before", default="2026-05-01")
    run(p.parse_args())
