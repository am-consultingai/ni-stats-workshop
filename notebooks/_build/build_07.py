#!/usr/bin/env python3
"""Build + execute Module 7 — Compose the multi-agent flow (capstone)."""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS = os.path.join(os.path.dirname(NB), ".claude", "skills")
OUT = os.path.join(NB, "07_capstone_orchestrator.ipynb")

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Module 7 · Compose the multi-agent flow

We built six deterministic specialist skills. Now we **compose** them into one orchestrator,
`/decide`, that takes the messy drill-down question and runs a **fixed, gated sequence** —
refusing to conclude early or anchor on the framing:

```
profile → effective-n → significance(cluster) → bayesian-update → trend → budget → SYNTHESIS
```

The individual statistics a strong model can already do. What it will **not** reproduce from a
terse prompt is this **procedure** — which tests, in what order, with which gates. That procedure
is the institutional knowledge the skill encodes. The analyst supplies the question; the flow
guarantees the floor.""")

md("""## Run the orchestrator on our Bing question
Each step below is a real specialist skill (its own reviewed, seeded script) invoked in order.""")

code("""import sys, subprocess, os
DECIDE = r"%s"
out = subprocess.run([sys.executable, os.path.join(DECIDE, "decide", "decide.py"),
                      "--a", "Summit Direct Business", "--b", "Cedar Business Bank",
                      "--slice", "channel=Bing"], capture_output=True, text=True)
print(out.stdout)""" % SKILLS)

md("""## The payoff — why deterministic composition beats re-asking

**Reproducibility:** the whole procedure is seeded. Run it twice → **byte-identical** output. You
can't build a team process on an analysis that wanders each run.""")

code("""r1 = subprocess.run([sys.executable, os.path.join(DECIDE, "decide", "decide.py")],
                    capture_output=True, text=True).stdout
r2 = subprocess.run([sys.executable, os.path.join(DECIDE, "decide", "decide.py")],
                    capture_output=True, text=True).stdout
print("Two independent runs identical:", r1 == r2)
assert r1 == r2""")

md("""## What the flow established (and what it refused to)

- **The honest verdict:** in Bing, *Summit Direct Business > Cedar* is a **thin-slice coin-flip**
  (cluster-bootstrap CI includes 0), and anchoring on history **reverses** it — regression to the
  mean. **Do not reorder the chart for Bing.**
- **Decide on profit/visit**, not a single-slice EPC or conversion gap.
- **What we did NOT establish:** causality (observational, no A/B), per-visit cost (aggregate only),
  and stability over time. Confirm any real move with a small holdout.

**The value the orchestrator adds over re-asking an LLM:**

| Property | Ad-hoc LLM analysis | Deterministic `/decide` |
|---|---|---|
| Same path every run | ✗ varies | ✓ fixed, gated |
| Byte-identical output | ✗ | ✓ seeded |
| Right test for the shape | sometimes | ✓ always |
| Respects clustering / effective-n | rarely | ✓ built-in |
| Cost per run | re-reasons every time | ~free compute |
| Auditable | ✗ | ✓ short reviewed code |

The analyst guarantees the **right question**; the tool guarantees the **floor**. That is the
workshop.""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": NB}})
nbf.write(nb, OUT)
print("wrote + executed", OUT, "cells:", len(nb.cells))
