---
name: decide
description: Orchestrate the messy drill-down decision ("in Bing, partner B beats A on EPC — reorder the chart?") into a FIXED, gated sequence of the specialist skills — profile → effective-n → significance (cluster-aware) → bayesian-update → trend → budget → synthesis — refusing to conclude early or anchor on the framing. The workshop's hero artifact (Module 7).
---

# /decide — the decision orchestrator

**When to use:** an open, loaded drill-down question arrives (*"in Bing, partner B looks best — should we reorder the chart?"*). Asking that raw risks anchoring on the framing, stopping early, or taking a different path each run. This skill runs the *same rigorous procedure every time*.

## Why this is the point of the workshop
The individual statistics a strong model can already do. What it will **not** reliably reproduce from a terse prompt is the **decision procedure** — which tests, in what order, with which gates. That procedure is the institutional knowledge this skill encodes. The analyst supplies the question; the flow guarantees the floor.

## The gated sequence it runs
```
profile → effective-n → significance(cluster) → bayesian-update → trend → budget → SYNTHESIS
```
Each step **gates** the next; it never concludes before the gates pass, and always ends with "what we did NOT establish".

## How to run
```bash
python decide.py --a "Summit Direct Business" --b "Cedar Business Bank" --slice channel=Bing
```

## What to report back
Present the **steps in order** — the gates are the teaching. Lead the recommendation with the honest verdict (on a thin slice the difference is a coin-flip, and a historical prior reverses it → do NOT reorder), and always include the **"what we did NOT establish"** block. Do not collapse it to the final answer; the auditable path is the deliverable.
