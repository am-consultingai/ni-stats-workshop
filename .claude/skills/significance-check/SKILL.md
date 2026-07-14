---
name: significance-check
description: Test whether a difference between two NI groups is real or an artifact of thin data / skew / clustering — auto-selects the right test (two-proportion z for rates; Mann-Whitney + bootstrap for skewed revenue/EPC) and uses the CLUSTER bootstrap at click grain. Use whenever someone asks "is this difference significant / real / just noise?" (Google vs Bing, or partner B vs A in a slice). Answers Module 3.
---

# /significance-check — is the difference real?

**When to use:** any "is X really different from Y?" — especially the drill-down (*"in Bing, partner B beats A on EPC — real?"*). Catches the #1 error: the wrong test for the shape, and the #2 error: treating clustered rows as independent.

## How to run
```bash
# the drill-down (click grain, clustered on visit)
python significance.py --metric revenue --group partner \
    --a "Summit Direct Business" --b "Cedar Business Bank" \
    --grain click --slice channel=Bing --cluster
# a rate at visit grain
python significance.py --metric converted --group channel --a Google --b Bing
```

## What to report back
Surface the `TEST CHOSEN` line and the full **VALIDATION** block. Lead with the **effect size + CI**, then the p-value. If the bootstrap CI includes 0 on a thin slice, say so plainly — the honest verdict is "coin-flip", and the next move is `/bayesian-update`, not a bigger claim.
