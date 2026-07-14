---
name: bayesian-update
description: Module 4 — Bayesian inference. When a slice is too thin to call (a significance test returns a CI that includes 0 = coin-flip), bring in an informative PRIOR from history, combine it with the thin-slice likelihood via a conjugate normal update, and report the POSTERIOR, P(better) and expected loss. Converts "we can't tell" into a probability, and often reveals an apparent slice win as regression to the mean (partial pooling / "borrowing strength"). Use right after /significance-check returns a coin-flip.
---

# /bayesian-update — Bayesian inference: does the thin-slice signal survive a prior?

**When to use:** immediately after `/significance-check` returns a CI that includes 0 on a small
slice. Instead of "we can't tell", do a **Bayesian update** — anchor the estimate on prior evidence
and produce a decision-grade answer: **P(A > B)** and the expected loss of acting.

**The idea:** posterior ∝ prior × likelihood. The prior is what the partner does elsewhere / earlier;
the likelihood is the thin slice. The posterior is narrower than either, and "borrowing strength"
across slices (partial pooling) is exactly what pulls a noisy small-sample estimate toward reality.

## How to run
```bash
python bayesian_update.py --group partner \
    --a "Summit Direct Business" --b "Cedar Business Bank" \
    --slice channel=Bing --prior other-slice
```
- `--prior other-slice` : prior = same partner OUTSIDE the slice (e.g. other channels).
- `--prior prior-month --before 2026-05-01` : prior = same partner+slice, earlier.

## What to report back
Show each arm's **thin slice → prior → posterior**, then the **posterior ordering**, **P(better)**,
and expected loss. If the ordering *reverses* under the prior, the slice "win" was regression to the
mean — do not act. Always carry the stationarity caveat: a prior only helps if the past is
representative — validate with `/trend-check` first.
