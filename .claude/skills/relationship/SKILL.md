---
name: relationship
description: Measure the relationship between two NI variables with BOTH Pearson and Spearman, flag when they disagree (outliers/non-linearity), and run a confound / Simpson's-paradox check within a chosen dimension. Use whenever someone claims "X correlates with Y" (e.g. clickout position vs revenue, competition rank vs CTR). Answers Module 3 (confound).
---

# /relationship — is that correlation real, or a mix artifact?

**When to use:** any "X correlates with Y" claim — especially before believing a driver behind a drill-down.

## How to run
```bash
python relationship.py --x clickout_position --y revenue --by segment
```

## What to report back
Show **Pearson vs Spearman** (disagreement → outliers/non-linearity, trust Spearman) and the **confound check**: does the sign hold within each level of `--by`? If it flips inside subgroups, the headline correlation is Simpson's paradox — report the within-group story. Always: correlation ≠ causation.
