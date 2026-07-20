# Deterministic Analysis for Analysts (v2)

A hands-on statistics workshop for Natural Intelligence analysts. **Thesis:** when a question
recurs, build a reviewed, seeded, **deterministic** test once and have the agent *run* it — instead
of re-analyzing the data on every prompt. You learn the statistics so you can **build and validate**
those tests.

Everything is anchored on one recurring question (`business_articulate.md`):
*"In Bing, partner B beats partner A on EPC — should we reorder the chart?"* — and the honest answer
the workshop reaches is **no** (thin-slice coin-flip; history reverses it).

## Layout
```
new_workshop/
├── business_articulate.md      the canonical question
├── WORKSHOP_PLAN.md            facilitator run-of-show (modules, timings, notes)
├── notebooks/                  the 8-module spiral, 00 → 07
│   ├── 00_framing_and_the_trap
│   ├── 01_trust_the_average          (M1 · /profile-data)
│   ├── 02_how_much_data              (M2 · i.i.d. / effective-n)
│   ├── 03_is_it_real                 (M3 · /significance-check, /relationship)
│   ├── 04_bayesian_inference            (M4 · Bayesian / /bayesian-update)
│   ├── 05_trend_seasonality          (M5 · /trend-check)
│   ├── 06_profit_decision            (M6 · /budget-decision)
│   ├── 07_capstone_orchestrator      (M7 · /decide)
│   └── reference/                    deeper background notebooks
├── .claude/skills/             the deterministic engine + skills
│   ├── _lib/ni_core.py               shared primitives (the heart)
│   └── <skill>/{<script>.py, SKILL.md}
├── data/                       real NI mock data (online_banking_*.csv) — one dataset for all of M0–M7
├── src/                        notebook chart style (ni_style.py); data + stats live in ni_core
└── bonus/                      /decision-retro (demoted from core)
```

## Setup (first time)
One command builds a self-contained `.venv`, installs the pinned dependencies, registers a Jupyter
kernel, and runs a smoke test. Works on macOS and Ubuntu/Linux (needs Python 3.11–3.13):
```bash
./setup.sh                                 # or: bash setup.sh
source .venv/bin/activate                  # macOS / Linux
```

## Run
Open **this folder** as the project root (VS Code / Cursor + Claude Code) so the skills in
`.claude/skills/` are discovered. Then, with the venv active:
```bash
jupyter lab                                # work through notebooks 00 → 07
# or run a skill directly (every run is byte-identical):
python .claude/skills/decide/decide.py --a "Summit Direct Business" --b "Cedar Business Bank" --slice channel=Bing
```

## The skills (each = one deterministic script + a SKILL.md wrapper)
`/profile-data` · `/significance-check` · `/estimate-effect` · `/relationship` ·
`/bayesian-update` · `/trend-check` · `/budget-decision` · `/decide` (orchestrator).

Each prints a `VALIDATION` block and is seeded — running it twice yields identical output.
