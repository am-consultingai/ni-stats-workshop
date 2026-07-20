#!/usr/bin/env python3
"""Build + execute Module 0 — Framing & the trap (Act 1: the wrong tool decides)."""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(NB, "00_framing_and_the_trap.ipynb")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Module 0 · Framing & the trap

**The recurring question (from `business_articulate.md`):**

> Overall our chart ranks **Cedar Business Bank above Summit Direct Business** by EPC.
> But drilling into **Bing, this past month**, Summit is way ahead — *"let's reorder the chart
> for Bing."*

The analyst's real job is **not** to confirm the reorder — it's to decide whether that apparent
difference is **real enough to act on**, or an artifact of thin data / seasonality / noise.

This module walks into the trap on purpose. We will run the reflex test, get a **confident,
publishable answer** — and that answer will be **wrong**. Everything after this module is the
machinery for noticing.""")

md("""## The drill-down, exactly as it lands on the desk

One channel (**Bing**), one month (**May**), two partners. The reflex tool for "is A bigger than
B?" is **Welch's t-test**, so that is what we run.""")

code("""import sys, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path as _Path
for _c in [_Path.cwd(), *_Path.cwd().parents]:          # portable: find the repo root
    if (_c / '.claude' / 'skills' / '_lib' / 'ni_core.py').exists():
        sys.path.insert(0, str(_c / '.claude' / 'skills' / '_lib')); break
import ni_core as C

clk = C.load_clickouts()
cl  = clk[clk["clicked"]]

# the drill-down slice: Bing, May, Business Checkings
slice_ = cl[(cl["channel"] == "Bing") & (cl["month"] == "2026-05")
            & (cl["segment"] == "Business Checkings")]
A = slice_[slice_["partner"] == "Summit Direct Business"]["revenue"].to_numpy(float)
B = slice_[slice_["partner"] == "Cedar Business Bank"]["revenue"].to_numpy(float)

t, p = stats.ttest_ind(A, B, equal_var=False)
print(f"Summit Direct Business  EPC = {A.mean():6.2f}   (n = {len(A)} click-outs)")
print(f"Cedar Business Bank     EPC = {B.mean():6.2f}   (n = {len(B)} click-outs)")
print(f"gap = {A.mean() - B.mean():+.2f} EPC")
print()
print(f"Welch t-test:  t = {t:.2f},  p = {p:.4f}")
print()
print("READING:  p < 0.05 -> 'the difference is statistically significant'")
print("DECISION: 'Summit earns ~$5 more per click in Bing. Reorder the chart, promote Summit.'")

assert p < 0.05, "Module 0 depends on the naive test firing"
""")

md("""## That decision is wrong.

Not "imprecise" — **wrong**. This dataset is generated, so we know the truth the test was trying
to estimate. Here it is:""")

code("""truth = pd.read_csv(C.ROOT / "data" / "ground_truth_epc.csv")
tb = truth[(truth.channel == "Bing") &
           (truth.partner.isin(["Summit Direct Business", "Cedar Business Bank"]))]
print(tb[["partner", "channel", "true_epc"]].to_string(index=False))
print()
print("TRUE ordering in Bing:  Cedar 10.5  >  Summit 7.8   (and in every other channel too)")
print("The t-test just told us to promote the WORSE partner, at p < 0.05.")

assert float(tb[tb.partner == "Cedar Business Bank"].true_epc.iloc[0]) > \\
       float(tb[tb.partner == "Summit Direct Business"].true_epc.iloc[0])
""")

md("""So the tool did not merely fail to help — it **manufactured confidence in the wrong
direction**. Two things went wrong at once, and each gets its own module.""")

md("""## Failure 1 — a few whale visits manufactured the GAP (Module 1)

EPC is **zero-inflated and whale-driven**: most click-outs earn nothing, and a handful carry
enormous payouts. A t-test compares *means* and assumes they behave nicely. Look at the shape it
was handed:""")

code("""pooled = np.concatenate([A, B])
s = C.describe_shape(pooled)
print(f"zero-share  = {s['zero_share']*100:4.0f}%   (most click-outs earn $0)")
print(f"skew        = {s['skew']:4.1f}    (a normal distribution has skew 0)")
print(f"top-1% of click-outs hold {s['top1_share']*100:.0f}% of all revenue")
print(f"median = {np.median(pooled):.2f}   vs   mean = {pooled.mean():.2f}")

assert s['zero_share'] > 0.5 and s['skew'] > 2
print()
print("-> The 'average' is a number almost no click-out resembles.")
""")

md("""## Failure 2 — clustering manufactured the CONFIDENCE (Module 2)

The t-test believed it had ~237 and ~522 **independent** observations. It did not. Click-outs
arrive in **visits**: one comparison-shopper opens three partners in a row, and those rows move
together. The real unit of independent evidence is the *visit*:""")

code("""for name, d in (("Summit", slice_[slice_.partner == "Summit Direct Business"]),
                ("Cedar ", slice_[slice_.partner == "Cedar Business Bank"])):
    print(f"{name}: {len(d):4d} click-outs  but only  {d.visit_iid.nunique():4d} visits")

de = C.design_effect(pooled, pd.concat([
        slice_[slice_.partner == "Summit Direct Business"],
        slice_[slice_.partner == "Cedar Business Bank"]])["visit_iid"].to_numpy())
print()
print(f"design effect = {de['deff']:.2f}   (ICC = {de['icc']:.2f})")
print(f"n = {de['n']} rows  ->  effective n = {de['n_eff']:.0f}")
print()
print(f"-> The test's standard error was ~{np.sqrt(de['deff']):.2f}x too small.")
print("   That inflation is what pushed p under 0.05.")

assert de['deff'] > 1.5
""")

md("""## The trap, part 2 — non-reproducibility

If you just *ask an LLM* "is Summit better in Bing?", you may get a **different analysis path each
run** — sometimes a t-test, sometimes a mean comparison, sometimes it catches the skew, sometimes
not. You cannot build a team process on an answer that changes when you re-ask.

**The workshop's thesis:** when a question recurs, don't re-derive the analysis each time. Build a
**reviewed, seeded, deterministic test once** — the agent *runs* it (byte-identical every time),
and you spend your judgment on the *question*, not the arithmetic.""")

md("""## The plan — a gated procedure, taught one beat at a time

Every module follows the same **4-beat loop**: do it **by hand** → **extract** the deterministic
function → **wrap** it as a Claude skill the agent runs → **validate** it with a simulation.

| Module | Question | Skill |
|---|---|---|
| 1 | Can I trust this average? | `/profile-data` |
| 2 | How much data is *really* here? (i.i.d.) | `/profile-data` (effective-n) |
| 3 | Is the difference real? | `/significance-check`, `/relationship` |
| 4 | Bayesian inference | `/bayesian-update` |
| 5 | Is the shift real over time? | `/trend-check` |
| 6 | The decision metric (profit/visit) | `/budget-decision` |
| 7 | Compose the multi-agent flow | `/decide` |

**The arc in three beats.** Module 0 was Act 1: the wrong tool, decisive and wrong.
Module 3 is **Act 2** — the honest tool, which will say *"we cannot call this; it's a coin-flip."*
Module 4 is **Act 3** — bring in evidence the drill-down never looked at (the same slice's own
history) and get a decision that is both **confident and correct**.""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": NB}})
nbf.write(nb, OUT)
print("wrote + executed", OUT, "cells:", len(nb.cells))
