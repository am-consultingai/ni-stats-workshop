#!/usr/bin/env python3
"""Reorganize the reused (synthetic-data) notebooks into the v2 spiral:
- rename the 3 core reused notebooks into the module numbering
- add a spine header cell + a skill-pointer footer cell to each
- move the remaining reference notebooks into notebooks/reference/
Idempotent: safe to re-run. Does not touch old_workshop.
"""
import os
import nbformat

NB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../new_workshop/notebooks
REF = os.path.join(NB, "reference")
os.makedirs(REF, exist_ok=True)

# reused core notebooks: old name -> (new name, module title, skill pointer)
CORE = {
    "02_outliers_skewness.ipynb": (
        "01_trust_the_average.ipynb",
        "# Module 1 · Can I trust this average?",
        "**Module 1 of the spiral — the drill-down question starts here:** before we compare "
        "partner B vs A in a slice, can we even trust the *average* of a metric like EPV/EPC? "
        "This module profiles the shape (skew, zeros, whales). *Runs on the synthetic teaching "
        "dataset; the real `online_banking` data behaves the same way.*",
        "**→ The skill that automates this:** `/profile-data`. Next: **Module 2 — how much data "
        "is really here?** (i.i.d. / effective-n).",
    ),
    "04_hypothesis_testing.ipynb": (
        "03_is_it_real.ipynb",
        "# Module 3 · Is the difference real?",
        "**Module 3 of the spiral:** the drill-down claim — *in Bing, partner B beats A on EPC* — "
        "is a two-group difference. Pick the right test for the shape, respect the n-trap, and "
        "check for a confound (Simpson's). *Synthetic teaching dataset.*",
        "**→ The skills that automate this:** `/significance-check` (cluster-aware) and "
        "`/relationship` (confound / Simpson's). On a thin real slice the CI includes 0 — a "
        "coin-flip — which sets up **Module 4 (borrow strength)**.",
    ),
    "06_integrated_case.ipynb": (
        "06_profit_decision.ipynb",
        "# Module 6 · The decision metric — profit per visit",
        "**Module 6 of the spiral:** conversion rate is the wrong yardstick. Reframe to "
        "**profit per visit = EPV − CPV** via the cost-table join. *Synthetic teaching dataset; "
        "the real skill runs on `online_banking`.*",
        "**→ The skill that automates this:** `/budget-decision`. Next: **Module 7 — compose the "
        "multi-agent flow** that runs all of this as one gated procedure.",
    ),
}

REFERENCE = [
    "01_foundations_tool_selection.ipynb",
    "03_correlation.ipynb",
    "05_confidence_intervals.ipynb",
]

for old, (new, header, _, footer) in CORE.items():
    src = os.path.join(NB, old)
    dst = os.path.join(NB, new)
    if not os.path.exists(src):
        print(f"skip (already done): {old}")
        continue
    nb = nbformat.read(src, as_version=4)
    # header cell first, footer cell last (only if not already added)
    if not (nb.cells and nb.cells[0].get("source", "").startswith(header.split("·")[0].strip())):
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(header + "\n\n" + CORE[old][2]))
    nb.cells.append(nbformat.v4.new_markdown_cell(footer))
    nbformat.write(nb, dst)
    os.remove(src)
    print(f"reused: {old} -> {new}  (+header/footer)")

for r in REFERENCE:
    src = os.path.join(NB, r)
    dst = os.path.join(REF, r)
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"reference: {r} -> reference/")
    else:
        print(f"skip reference (already moved): {r}")

print("\nfinal notebooks/:")
for f in sorted(os.listdir(NB)):
    print("  ", f)
