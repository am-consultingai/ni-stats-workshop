# Deterministic Analysis for Analysts — Facilitator Run-of-Show

_Natural Intelligence · ~5 hours, hands-on. Built by AM Consulting (Avishay Meron)._

## The one idea
When a question recurs, don't make an agent re-analyze the data every time. Build a **reviewed,
seeded, deterministic test once**; the agent **runs** it (byte-identical every run). Analysts learn
the statistics so they can **build and validate** those tests — and supply the question the tool
can't. The whole workshop builds one such procedure for the question in `business_articulate.md`.

## The spine (one question, all the way through)
> *"In Bing, partner B beats partner A on EPC — should we reorder the chart?"*

We answer it honestly by Module 7: on a thin slice the gap is a **coin-flip**, and anchoring on
history **reverses** it → don't reorder. Decide on **profit/visit**, not a single-slice EPC gap.

## The 4-beat loop (every module)
1. **By hand** (notebook) — understand the statistic on real-shaped data.
2. **Extract** — pull the reusable computation into a deterministic script (`.claude/skills/_lib/ni_core.py`).
3. **Wrap** — expose it as a Claude skill (`SKILL.md` → the script) the agent runs.
4. **Validate** — a simulation (A/A calibration, coverage, shrinkage-recovers-truth) *is* the test
   that proves the skill is right — how you trust the agent's output.

## Run-of-show (~5h, flexible)

| # | Module | Notebook | Skill(s) | Time |
|---|---|---|---|---|
| 0 | Framing & the trap | `00_framing_and_the_trap` | — | 20m |
| 1 | Can I trust this average? | `01_trust_the_average` | `/profile-data` | 35m |
| 2 | How much data is *really* here? (i.i.d.) | `02_how_much_data` | `/profile-data` (effective-n) | 30m |
| 3 | Is the difference real? | `03_is_it_real` | `/significance-check`, `/relationship` | 45m |
| — | **break** | | | 10m |
| 4 | Bayesian inference (borrow strength from history) | `04_bayesian_inference` | `/bayesian-update` | 45m |
| 5 | Is the shift real over time? | `05_trend_seasonality` | `/trend-check` | 30m |
| 6 | The decision metric (profit/visit) | `06_profit_decision` | `/budget-decision` | 30m |
| 7 | Compose the multi-agent flow | `07_capstone_orchestrator` | `/decide` | 45m |

`reference/` holds the deeper background notebooks (foundations/CLT, correlation, confidence
intervals) for anyone who wants them; `bonus/` holds the retrospective skill (`/decision-retro`),
demoted from the core per the client's 08/07 steer.

## Facilitation notes per module
- **M0 — the trap.** Run a plain t-test on the Bing slice; it "works" and gives a clean number.
  Then show the 79%-zeros / skew-12 shape and the non-reproducibility of ad-hoc LLM analysis. This
  is the "why" for everything that follows.
- **M1 — trust the average.** Distribution shape, skew, whales, mean vs median. Gate: is the mean
  even a typical value? (If not → bootstrap/rank, never a t-test.)
- **M2 — how much data.** Click-outs share visits → not i.i.d. ICC, design effect, effective-n,
  and an A/A sim showing the naive test over-fires while the cluster bootstrap is calibrated.
- **M3 — is it real.** Right test for the shape, the n-trap, effect size + CI, and the confound /
  Simpson's check. On the thin Bing slice the CI includes 0 — the honest verdict is *coin-flip*.
- **M4 — Bayesian inference.** Combine the thin slice (likelihood) with the partner's history (a
  prior) into a posterior — it narrows and the Bing flip **reverses** (regression to the mean).
  Output P(better) + expected loss. "Borrowing strength" / partial pooling is the intuition.
- **M5 — trend.** "CTR down yesterday?" → put it against the monthly band and day-of-week baseline;
  a regime break is what would invalidate M4's prior.
- **M6 — profit/visit.** Conversion is the wrong yardstick; rank on EPV − CPV via the cost join.
- **M7 — compose.** Run `/decide`; show it's byte-identical across runs; contrast with re-asking.

## How to run (VS Code / Cursor + Claude Code, or plain Jupyter)
```bash
# from this folder
python -m jupyter lab            # open the notebooks 00 → 07 in order
# skills run from .claude/skills/<skill>/<script>.py, e.g.:
python .claude/skills/decide/decide.py --a "Summit Direct Business" --b "Cedar Business Bank" --slice channel=Bing
```
Open **this folder** (`new_workshop/`) as the project root so Claude Code discovers the skills in
`.claude/skills/`. Every skill prints a `VALIDATION` block and is seeded → **two runs are
byte-identical**.

## Data
- Real NI mock data: `data/online_banking_visit_clickouts.csv` (+ `..._daily_cost.csv`).
  Skills and the new notebooks (M2/M4/M5/M7) run on this.
- Synthetic teaching data: `data/visits.csv` — used by the reused concept notebooks (M1/M3/M6) for
  now; re-basing them onto `online_banking` is a planned later step.
