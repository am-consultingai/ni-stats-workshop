---
name: trend-check
description: "\"CTR was down yesterday — is that meaningful?\" Put a day-over-day move in perspective: compare the latest day against the month-wide band and its own day-of-week baseline, and run a light regime-break test. Use for any \"is this recent drop/spike real or just seasonality/noise?\" question, and to validate whether a historical prior is still safe to use (stationarity). Answers Module 5."
---

# /trend-check — is the recent move real, or just seasonality?

**When to use:** someone panics that a metric (CTR, EPV, EPC) moved yesterday/this week. Also run it before trusting a historical prior in `/bayesian-update` — a regime break invalidates the prior.

## How to run
```bash
python trend.py --metric ctr --slice channel=Bing
python trend.py --metric epv
python trend.py --metric epc --slice partner="Summit Direct"
```

## What to report back
Show the latest day vs the **monthly ±2σ band** and vs its **day-of-week baseline**, plus the **regime-break** verdict. Lead with the honest read: is the move inside normal variation (→ don't react), or a genuine outlier/break (→ investigate; and a prior may no longer hold). Always remind: one low day is not a trend, and weekdays differ — don't compare across days.
