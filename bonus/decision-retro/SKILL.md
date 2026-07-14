---
name: decision-retro
description: Retrospectively audit a past NI decision — "was promoting this partner up the chart the right call?" — by comparing before/after on EPC, conversion and volume with a bootstrap interval, then explicitly listing the confounders (position changed, no control group, seasonality, traffic mix) that block a causal claim. Use for any "did our change actually work?" / post-mortem question. The workshop capstone.
---

# /decision-retro — was it the right call?

**When to use:** the retrospective question Maor asked for — an analyst has an hour and wants
to check, historically, whether a past decision (e.g. **promoting a partner to click-out
position 1**) actually worked. This is the capstone that ties the whole toolkit to NI's
real workflow.

## The one rule this skill enforces
> A before/after is **not** an A/B test. You can measure what changed, but you cannot claim
> the decision *caused* it until you have named — and ruled out — the confounders.

## How to run
```bash
python .claude/skills/ni-stats/ni_stats.py retro --partner "Summit Direct" --change-date 2026-04-01
```

- `--partner`     : the partner whose position changed
- `--change-date` : when the decision took effect (YYYY-MM-DD)

## What to report back
State the observed before/after (EPC, conversion, click volume, modal position) and the
bootstrap CI on the change. Then surface **every** flag in the `VALIDATION` block — the
confounders are the deliverable, not an afterthought. End with the verdict template:
*"Observed X; but Y, Z could equally explain it, so we cannot attribute it to the promotion."*

## The validation checklist this skill embodies
- [ ] Did a **confounder move with the decision** (position, traffic mix, competitor entry)?
- [ ] Is there a **control group**? If not, this is observational — hedge accordingly.
- [ ] Does the change survive a **bootstrap CI** (or is it within noise)?
- [ ] Are you separating **per-click quality (EPC)** from **volume**? They tell different stories.

**NI anchor (why this case is perfect):** Summit Direct was promoted from position 2 to 1.
Naively it looks like a win — click volume ~4×'d. But its **EPC per click stayed flat**
(≈ −0.3%, CI includes 0) and its **position changed**, so the per-click "improvement" is
unproven and confounded. The real, defensible story is a *volume* gain at steady EPC — a
much more honest thing to tell the business than "the promotion boosted performance."
