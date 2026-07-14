---
name: budget-decision
description: Answer "should we move budget between channels?" the right way — build the funnel by channel × platform, join the aggregate cost table to get cost-per-visit, and rank cells by PROFIT PER VISIT (EPV − CPV) with bootstrap intervals, instead of by the misleading conversion rate. Use for any budget-reallocation / "which channel is better?" decision. Answers Module 6.
---

# /budget-decision — decide on profit per visit, not conversion

**When to use:** any "shift budget / which channel is better?" question.

## How to run
```bash
python budget.py
python budget.py --by channel platform
```

## What to report back
Lead with the **profit/visit** ranking (EPV − CPV via the cost join), not conversion. Surface the reversal — a channel can convert better yet lose money on high CPC. Carry the caveat: cost is aggregate (no per-visit cost), so CPV is an average; confirm any big reallocation with a small holdout before scaling.
