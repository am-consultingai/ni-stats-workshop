#!/usr/bin/env python3
"""Build + execute Module 0 — Framing & the trap."""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(os.path.dirname(NB), ".claude", "skills", "_lib")
OUT = os.path.join(NB, "00_framing_and_the_trap.ipynb")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Module 0 · Framing & the trap

**The recurring question (from `business_articulate.md`):**

> The business does a drill-down and says: *"Overall our chart ranks A > B > C by EPC. But in
> **Bing** specifically, partner B ($9) beats partner A ($8) — let's reorder the chart for Bing."*

The analyst's real job is **not** to confirm the reorder — it's to decide whether that apparent
difference is **real enough to act on**, or an artifact of thin data / seasonality / noise.

This workshop builds a **deterministic, gated procedure** that answers this the same way every
time. This module shows *why the naive move fails* — and motivates the whole flow.""")

md("""## The trap, part 1 — the wrong test looks fine

Let's take the real Bing slice and compare the two partners' EPC (revenue per click-out) with a
plain **t-test**, the reflex tool.""")

code("""import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, r"%s")
import ni_core as C

clk = C.load_clickouts()
cl = clk[clk["clicked"]]
bing = cl[cl["channel"] == "Bing"]
A = bing[bing["partner"] == "Summit Direct Business"]["revenue"].to_numpy(float)
B = bing[bing["partner"] == "Cedar Business Bank"]["revenue"].to_numpy(float)

t, p = stats.ttest_ind(A, B, equal_var=False)
print(f"Summit Direct Business EPC = {A.mean():.2f} (n={len(A)})")
print(f"Cedar Business Bank    EPC = {B.mean():.2f} (n={len(B)})")
print(f"Welch t-test: t={t:.2f}, p={p:.3f}")
print("Naive reading: 'Summit is $2.30 higher — promote it in Bing.'")""" % LIB)

md("""## Why that reading is wrong

The metric is **brutally skewed and zero-inflated** — a t-test's assumptions don't hold, and a
handful of whale conversions in a small slice can invent a $2 gap. Look at the shape:""")

code("""s = C.describe_shape(np.concatenate([A, B]))
print(f"zero-share = {s['zero_share']*100:.0f}%   skew = {s['skew']:.1f}   "
      f"top-1% of clicks hold {s['top1_share']*100:.0f}% of revenue")
print(f"median EPC = {np.median(np.concatenate([A,B])):.2f}  (vs mean {np.concatenate([A,B]).mean():.2f})")
assert s['zero_share'] > 0.5 and s['skew'] > 5
print("\\n→ The mean is not a typical value; the t-test p-value is not trustworthy here.")""")

md("""## The trap, part 2 — non-reproducibility

If you just *ask an LLM* "is Bing better?", you may get a **different analysis path each run** —
sometimes a t-test, sometimes a mean comparison, sometimes it catches the skew, sometimes not.
That is not something you can build a team process on.

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

By Module 7 the orchestrator runs all of this as one gated procedure — and answers our Bing
question honestly. Spoiler: **the apparent flip is a coin-flip, and history reverses it.**""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": NB}})
nbf.write(nb, OUT)
print("wrote + executed", OUT, "cells:", len(nb.cells))
