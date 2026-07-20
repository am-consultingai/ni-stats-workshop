#!/usr/bin/env python3
"""decide.py — THE ORCHESTRATOR (Module 7). Takes the messy drill-down question
and runs a FIXED, GATED sequence of the specialist skills — profile → effective-n
→ significance → bayesian-update → trend → budget → synthesis — refusing to
conclude early or anchor on the framing. Same procedure, same order, every run.

    python decide.py \
        --a "Summit Direct Business" --b "Cedar Business Bank" --slice channel=Bing

The business asks: "In Bing, partner B beats partner A on EPC — reorder the chart?"
The orchestrator answers whether that is real enough to act on.
"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import ni_core as C
import numpy as np

SKILLS = Path(__file__).resolve().parents[1]


def _run(folder, script, extra):
    cmd = [sys.executable, str(SKILLS / folder / script)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def run(args):
    A, B, sl = args.a, args.b, (args.slice or ["channel=Bing", "month=2026-05"])
    slice_args = []
    for s in sl:
        slice_args += ["--slice", s]
    # month= names the drill-down window; the trend/stationarity step must see the
    # WHOLE history of the slice (that's what validates the prior), so strip it there.
    trend_slice_args = []
    for s in sl:
        if not s.startswith("month="):
            trend_slice_args += ["--slice", s]
    months = [s.split("=", 1)[1] for s in sl if s.startswith("month=")]
    cutoff = f"{months[-1]}-01" if months else "2026-05-01"

    print("=" * 72)
    print("DECISION MEMO — gated statistical procedure (deterministic)")
    print("=" * 72)
    print(f"Q: In {sl}, does '{A}' beat '{B}' on EPC enough to reorder the chart?")
    print("Each step GATES the next. We do not anchor on the framing or conclude early.")

    # ---- inline facts that drive the gates (computed once, from ni_core) ----
    clk = C.load_clickouts(Path(args.visits))
    cl = C.apply_slice(clk[clk["clicked"]], sl)
    da = cl[cl["partner"].astype(str) == A]
    db = cl[cl["partner"].astype(str) == B]
    cb = C.cluster_bootstrap_diff_ci(da.revenue.values, da.visit_iid.values,
                                     db.revenue.values, db.visit_iid.values)
    coin_flip = cb["diff_lo"] < 0 < cb["diff_hi"]
    # borrow: posterior ordering vs raw ordering (prior = same slice, pre-cutoff)
    full = clk[clk["clicked"]]
    import pandas as pd

    def _post(who):
        mine = full[full.partner.astype(str) == who]
        s = C.apply_slice(mine, sl)
        pri = C.apply_slice(mine, [x for x in sl if not x.startswith("month=")])
        pri = pri[pri["click_timestamp"] < pd.Timestamp(cutoff)]
        dm, dse2, _ = C.mean_and_se2(s.revenue.values)
        dse2 *= C.design_effect(s.revenue.values, s.visit_iid.values)["deff"]
        pm, pse2, _ = C.mean_and_se2(pri.revenue.values)
        pse2 *= C.design_effect(pri.revenue.values, pri.visit_iid.values)["deff"]
        return dm, C.posterior_from_prior(pm, pse2, dm, dse2)
    dmA, poA = _post(A)
    dmB, poB = _post(B)
    flipped = (dmA > dmB) != (poA["mean"] > poB["mean"])
    pb = C.p_better(poA["mean"], poA["var"], poB["mean"], poB["var"])
    decisive = pb >= 0.9 or pb <= 0.1

    steps = [
        ("STEP 1 · PROFILE the metric — is its mean trustworthy?",
         "profile-data", "profile.py",
         ["--metric", "revenue", "--grain", "click"] + slice_args,
         "GATE ▸ EPC is zero-inflated & whale-skewed ⇒ use bootstrap/rank, never a t-test."),
        ("STEP 2 · TEST the claim (cluster-aware) — is the difference real?",
         "significance-check", "significance.py",
         ["--metric", "revenue", "--group", "partner", "--a", A, "--b", B,
          "--grain", "click", "--cluster"] + slice_args,
         "GATE ▸ " + ("CI straddles 0 → COIN-FLIP on this slice. Do not conclude. Bring in a prior."
                      if coin_flip else "CI excludes 0 → a real slice difference; still size it.")),
        ("STEP 3 · BAYESIAN INFERENCE — does the flip survive an informative prior?",
         "bayesian-update", "bayesian_update.py",
         ["--group", "partner", "--a", A, "--b", B, "--prior", "prior-month"] + slice_args,
         "GATE ▸ " + (f"posterior REVERSES the slice ordering decisively (P({A} better) = {pb*100:.0f}%) "
                      f"→ the flip was regression to the mean."
                      if flipped and decisive else
                      f"posterior P({A} better) = {pb*100:.0f}% — still not decisive. Hold."
                      if not decisive else
                      f"posterior preserves the ordering (P = {pb*100:.0f}%) → a defensible case.")),
        ("STEP 4 · TREND / STATIONARITY — is the prior even valid?",
         "trend-check", "trend.py",
         ["--metric", "epc"] + trend_slice_args,
         "GATE ▸ no regime break ⇒ the historical prior is safe to lean on."),
        ("STEP 5 · REFRAME to the decision metric — profit/visit, not EPC alone",
         "budget-decision", "budget.py", [],
         "GATE ▸ rank on profit/visit (EPV−CPV); a per-click EPC gap is not a budget mandate."),
    ]
    for title, folder, script, extra, gate in steps:
        print(C.hr(title))
        print(_run(folder, script, extra).rstrip())
        print("  " + gate)

    print(C.hr("SYNTHESIS — recommendation (grounded in the gated steps)"))
    if coin_flip and flipped and decisive:
        print(f"  • Do NOT reorder the chart for {sl}. The apparent '{A} > {B}' is a thin-slice coin-flip "
              f"(CI includes 0), and anchoring on the slice's own history REVERSES it decisively "
              f"(P({A} better) = {pb*100:.0f}%) — regression to the mean.")
    elif coin_flip and not decisive:
        print(f"  • The slice difference is a coin-flip (CI includes 0) and even with history "
              f"P({A} better) = {pb*100:.0f}% — not decisive. Hold; gather more data.")
    elif coin_flip:
        print(f"  • The slice alone is a coin-flip, but history makes it decisive "
              f"(P({A} better) = {pb*100:.0f}%). Act on the posterior ordering, size the impact first.")
    else:
        print(f"  • The slice difference is real; size the profit impact before acting.")
    print("  • Decide on profit/visit, not EPC or conversion in a single slice.")
    print(C.hr("WHAT WE DID NOT ESTABLISH (honesty gate)"))
    print("  • Causality — observational, no A/B; cost is aggregate (no per-visit cost).")
    print("  • Stability — a slice ordering next month may differ. Confirm any real move with a holdout.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--visits", default=str(C.DEFAULT_VISITS))
    p.add_argument("--a", default="Summit Direct Business")
    p.add_argument("--b", default="Cedar Business Bank")
    p.add_argument("--slice", action="append", default=None)
    run(p.parse_args())
