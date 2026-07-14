---
name: profile-data
description: Profile an NI metric's distribution (shape, skew, whales, zero-inflation) AND how much independent data is really in a slice (effective-n via the design effect). Use FIRST, before trusting any average or running any test — whenever someone reports a mean of EPV/revenue/EPC or asks "can I trust this number?". Answers Modules 1–2.
---

# /profile-data — can you trust this number, and how much data is really here?

**When to use:** the first move on any slice. Before a mean or a test is trusted, check the *shape* (is the mean even a typical value?) and the *effective n* (are the rows independent, or do click-outs share visits?).

## How to run
```bash
python profile.py --metric revenue --grain click --slice channel=Bing
python profile.py --metric epv --group platform
```
- `--grain click` profiles click-out rows (partner/EPC questions) and reports the **design effect**; `--grain visit` (default) is one independent row per visit.
- `--slice col=val` (repeatable) narrows to the drill-down slice.

## What to report back
Surface the shape line and the **VALIDATION** + **EFFECTIVE-N** blocks. Lead with: is the metric skewed (→ bootstrap/rank, never a t-test)? and what is the *effective* n after clustering? These two gates decide whether any downstream test is even valid.
