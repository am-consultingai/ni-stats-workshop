---
name: estimate-effect
description: Quantify how big an effect is and how sure you are — a point estimate with the correct 95% interval (Wilson for rates, bootstrap for skewed EPV/revenue), plus width-vs-n intuition and three-audience phrasing. Use when the question is "how much?" / "how confident?" rather than a yes/no significance test.
---

# /estimate-effect — how big, and how sure?

**When to use:** the question is magnitude, not a yes/no — "how much lower is mobile EPV, and how confident are we?"

## How to run
```bash
python estimate.py --metric converted --group channel --a Google --b Bing
python estimate.py --metric epv --group platform --a mobile --b desktop
```

## What to report back
Lead with the **interval**, not the point. Give the executive/analyst/skeptic phrasing, and the width-vs-n note (halving the interval needs ~4× the data). A single number with no interval hides the uncertainty.
