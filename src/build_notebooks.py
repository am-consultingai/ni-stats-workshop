"""
build_notebooks.py — generate the six workshop Jupyter notebooks.

Each notebook loads the mock NI data via ni_style and walks analysts from raw
numbers -> the right statistical tool -> a decision, with branded charts.

Run:  python src/build_notebooks.py   (then execute with nbconvert)
"""
from pathlib import Path
import nbformat as nbf

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"
NBDIR.mkdir(exist_ok=True)


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


def build(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "ni-workshop", "display_name": "Python (NI workshop)"},
        "language_info": {"name": "python"},
    }
    path = NBDIR / name
    nbf.write(nb, path)
    print("wrote", path)


# Shared bootstrap cell -------------------------------------------------------
BOOT = r'''
import sys
from pathlib import Path
_here = Path.cwd()
for _c in [_here, *_here.parents]:
    if (_c / "src" / "ni_style.py").exists():
        sys.path.insert(0, str(_c / "src")); break
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import ni_style as ni
ni.set_style()

visits = ni.load_visits()
print(f"Loaded {len(visits):,} visits | {visits['date'].min().date()} -> {visits['date'].max().date()}")
visits.head()
'''


# =========================================================================== #
# MODULE 1 — Foundations: selecting the right statistical tool
# =========================================================================== #
def module1():
    cells = [
        md(r'''
# Module 1 — From a Business Question to a Statistical Question
### Practical Statistics for Analysts @ Natural Intelligence

> **💬 The Slack message that starts our day** (and runs through the whole workshop):
>
> *"Google's converting way better than Bing this week — 45% vs 35% — mobile EPV looks down, and weekend EPV is crashing. Should we pull budget out of Bing and the weekend?"*

We *could* jump straight to tests. Instead we build the foundation first, **then** inspect the
data, **then** choose the method — because the right tool *follows* from the theory:

**Part 1 · Theory** — distributions (the *shape* of data) and the Central Limit Theorem (why
*averages* are trustworthy). → **Part 2 · Inspect the data** — the NI visit grain and its KPIs. →
**Part 3 · Choose the tool** — the right method follows from the distribution.
'''),
        code(BOOT),

        # ------------------------------------------------------------------ #
        # PART 1 — THEORY: DISTRIBUTIONS
        # ------------------------------------------------------------------ #
        md(r'''
# Part 1 · Theoretical background

We'll *illustrate* the theory with NI's visit-level data loaded above — each row is a visit,
`revenue` is its EPV contribution (\$0 unless it converted), and `converted` is a click-out (0/1).
We formalise the grain and KPIs in **Part 2**; first, the two ideas that decide everything else:
**distributions** and the **Central Limit Theorem**.

## Distributions: the shape of your data (and why it rules everything)

A **distribution** is just *how often each value occurs* — the **shape** of your data. Before
any statistic, ask *"what shape is this?"*, because the shape decides:

- which **summary** is honest (mean vs median),
- which **test** is valid,
- whether a few **outliers** secretly run the show.

Four shapes you'll meet constantly — two textbook, two straight from NI's own data:
'''),
        code(r'''
# A distribution = "how often each value occurs" = its SHAPE.
# We show four families: two synthetic (to see the ideal shape) + two from real NI data.
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

# 1) NORMAL (symmetric "bell"): values cluster around a centre; rare extremes on BOTH sides.
#    Where it shows up: measurement noise, and — crucially — AVERAGES of many things (Step 2).
normal = rng.normal(100, 15, 5000)
ax[0].hist(normal, bins=40, color=ni.BLUE, alpha=.85)
ni.titlebox(ax[0], "Normal (symmetric)", "mean ≈ median; ±SD intervals just work")

# 2) RIGHT-SKEWED / LOG-NORMAL: a wall on the left, a long tail to the right.
#    Where it shows up: session length, revenue per converting visit, prices.
skewed = rng.lognormal(3, 0.9, 5000)
ax[1].hist(skewed, bins=40, color=ni.TEAL, alpha=.85)
ni.titlebox(ax[1], "Right-skewed (log-normal)", "mean is pulled right by the tail")

# 3) BINARY / BERNOULLI: only two outcomes, 0 or 1. Its 'shape' is two bars,
#    and its MEAN is literally the rate. Where it shows up: converted yes/no.
conv = visits["converted"].values
ax[2].bar([0, 1], [np.mean(conv == 0), np.mean(conv == 1)], width=.6, color=ni.GREEN)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["no (0)", "yes (1)"])
ni.titlebox(ax[2], "Binary / Bernoulli", f"mean = the rate itself ({conv.mean():.0%})")

# 4) HEAVY-TAILED: like skewed but FAR more extreme — a few values dwarf the rest (whales).
#    This is NI's revenue-per-visit. The tail, not the bulk, drives the average.
rev = visits["revenue"].values
ax[3].hist(np.clip(rev, 0, 80), bins=40, color=ni.ORANGE, alpha=.85)  # clip only for display
ni.titlebox(ax[3], "Heavy-tailed (real EPV)", "rare whales dominate the mean")

fig.suptitle("Four distribution shapes every analyst must recognise",
             fontsize=15, fontweight="bold", color=ni.NAVY, y=1.06)
fig.tight_layout(); ni.savefig(fig, "m1_distribution_zoo"); plt.show()
'''),
        md(r'''
### Why the shape matters — three consequences

1. **It picks your summary.** On a *symmetric* shape, mean ≈ median. On a *skewed/heavy* shape,
   the mean is dragged toward the tail and can misrepresent the "typical" visit.
2. **It picks your test.** Binary → proportion tests; symmetric-continuous → t-test;
   skewed/heavy → rank tests or bootstrap; a ratio → delta/bootstrap. *(That's Step 3.)*
3. **It decides whether outliers run the show.** In heavy tails the top 1% can be most of the
   total — so "the average" is really a story about a few whales.

Consequence #1 in numbers — watch the mean/median gap widen as the shape gets heavier:
'''),
        code(r'''
# Diagnostic: the ratio  mean / median.
#   ~1.0  -> symmetric: the mean is a safe, honest summary.
#   >>1.0 -> right-skewed/heavy: the mean is inflated by the tail; prefer the median / be careful.
rows = []
for name, data in [("Normal (symmetric)", normal),
                   ("Right-skewed",        skewed),
                   ("Heavy-tailed EPV*",   rev[rev > 0])]:   # * converting visits only
    m, med = np.mean(data), np.median(data)
    rows.append([name, m, med, m / med])

summary = pd.DataFrame(rows, columns=["distribution", "mean", "median", "mean / median"])
display(summary.round(2))
print("Top -> bottom the shape gets heavier and mean/median climbs far above 1.0:")
print("the mean stops describing a 'typical' visit and starts describing the whales.")
'''),

        # ------------------------------------------------------------------ #
        # STEP 2 — CENTRAL LIMIT THEOREM
        # ------------------------------------------------------------------ #
        md(r'''
## The Central Limit Theorem: why we can still trust averages

Shapes like EPV are ugly — so how can we ever put a clean "±1.96·SE" interval around *mean*
EPV? The **Central Limit Theorem (CLT)**: the **average** of many independent observations has a
*sampling distribution* that approaches a **normal** curve as n grows — *whatever the raw shape*
— with spread (standard error) = σ/√n.

The key idea: **the CLT acts on the *mean*, not on the raw values.** Watch the sampling
distribution of mean EPV go from badly skewed to bell-shaped as n grows:
'''),
        code(r'''
from scipy.stats import skew, norm
rng = np.random.default_rng(0)
pop = visits["revenue"].values            # the raw, heavy-tailed population (EPV per visit)
true_mu = pop.mean()
ns = [5, 30, 200, 2000]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, n in zip(axes, ns):
    # Draw 4000 independent samples of size n; take each sample's MEAN.
    # The histogram of those 4000 means IS the sampling distribution of the mean at this n.
    means = rng.choice(pop, size=(4000, n)).mean(axis=1)
    ax.hist(means, bins=45, density=True, color=ni.LIGHT, edgecolor=ni.SKY)
    # Overlay the normal curve the CLT predicts — the fit improves as n grows.
    xs = np.linspace(means.min(), means.max(), 200)
    ax.plot(xs, norm.pdf(xs, means.mean(), means.std()), color=ni.RED, lw=2, label="normal fit")
    ax.axvline(true_mu, color=ni.NAVY, lw=1.6, ls="--")
    ni.titlebox(ax, f"n = {n}", f"skew of sample-mean = {skew(means):.2f}")
    ax.set_yticks([])
axes[0].legend()
fig.suptitle("Central Limit Theorem: the mean's distribution becomes normal as n grows",
             fontsize=15, fontweight="bold", color=ni.NAVY, y=1.07)
fig.text(0.5, -0.04, "Raw EPV is wildly skewed, yet by n=2000 the AVERAGE is nearly normal — "
         "that's why large-sample means are trustworthy.", ha="center", color=ni.GREY, fontsize=11)
fig.tight_layout(); ni.savefig(fig, "m1_clt_in_action"); plt.show()
'''),
        md(r'''
### The leap: the two promises that turn "it's normal" into a confidence interval

Watching the means go *normal* still isn't enough to build an interval — for that we need two
**numbers** about that distribution of means. The CLT (plus the basic algebra of averages) pins
both down:

1. **Center.** The middle of the distribution-of-means is the **true population mean μ**.
   → a single sample mean is an *unbiased* estimate of μ (it doesn't systematically run high or low).
2. **Width.** The standard deviation of the distribution-of-means is **σ/√n** — and *that quantity is
   the Standard Error (SE)*. → uncertainty shrinks like 1/√n: quadruple the data, halve the spread.

Those two numbers ARE the interval: **μ̂ ± 1.96 × (σ/√n)**.

> ℹ️ *Precise version:* the center (= μ) and width (= σ/√n) hold at **any** n — they're facts about
> averages, not the CLT. The CLT's special gift is the **normal *shape*** at large n, which is what
> lets us use the "1.96" (a normal-distribution quantile). Shape **+** center **+** width = the interval.

Let's verify the two numbers directly, by measuring the thousands of sample means themselves:
'''),
        code(r'''
# We GENERATE the "distribution of many means" and then measure its CENTRE and WIDTH,
# checking each against the CLT's prediction.
from scipy.stats import norm
rng = np.random.default_rng(0)
pop = visits["revenue"].values
mu_true = pop.mean()      # the true population mean μ (known here — it's mock data)
sigma   = pop.std()       # the population standard deviation σ

# --- Promise check across several sample sizes ---
rows = []
for n in [30, 100, 300, 1000, 3000]:
    means = rng.choice(pop, size=(8000, n)).mean(axis=1)     # 8000 sample-means of size n
    rows.append([n,
                 means.mean(),          # CENTRE of the means (empirical)  -> should equal μ
                 mu_true,               # μ for comparison
                 means.std(),           # WIDTH of the means (empirical)   -> should equal σ/√n
                 sigma/np.sqrt(n)])     # σ/√n the CLT predicts
check = pd.DataFrame(rows, columns=["n", "centre of means", "μ (true)",
                                    "width of means", "σ/√n (predicted)"])
display(check.round(3))
print("Promise 1 — 'centre of means' ≈ μ at every n  →  the sample mean is unbiased.")
print("Promise 2 — 'width of means' ≈ σ/√n at every n →  that's literally why SE = σ/√n.")

# --- Visual proof at n = 200: centred on μ, spread by exactly σ/√n ---
n = 200
means = rng.choice(pop, size=(8000, n)).mean(axis=1)
se = sigma / np.sqrt(n)                                        # the Standard Error = σ/√n

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(means, bins=50, density=True, color=ni.LIGHT, edgecolor=ni.SKY)
ax.axvspan(mu_true - se, mu_true + se, color=ni.BLUE, alpha=0.12, label="μ ± 1 SE   (SE = σ/√n)")
ax.axvline(mu_true, color=ni.NAVY, lw=2.2, label=f"centre = μ = ${mu_true:.2f}")   # promise 1
xs = np.linspace(means.min(), means.max(), 200)
ax.plot(xs, norm.pdf(xs, mu_true, se), color=ni.RED, lw=2.2,                        # promise 2 (width=σ/√n)
        label=f"Normal(μ, σ/√n),  SE = ${se:.2f}")
ax.set_xlabel(f"mean of a sample of n={n} visits ($)"); ax.set_yticks([]); ax.legend()
ni.titlebox(ax, "The CLT's two promises, made concrete (n = 200)",
            "the 8,000 sample means centre on μ and spread by exactly σ/√n — that pair IS the interval")
fig.tight_layout(); ni.savefig(fig, "m1_clt_two_promises"); plt.show()
'''),
        md(r'''
### How much data do you actually need? (the heavy-tail tax)

Promise 2 has a sharp business consequence. The mean's **relative** uncertainty is **CV / √n**, where
**CV = σ/μ** is the coefficient of variation — *how heavy the tail is relative to the average*. EPV's
CV ≈ 4.5 (σ ≈ 35 on a mean of ≈ 8), so EPV is **expensive to measure**: you need a surprising amount
of traffic before the average is pinned down. The curve below turns that into a planning rule.
'''),
        code(r'''
# Relative margin of error on mean EPV = CV / sqrt(n).  Lower CV (tamer data) -> cheaper to measure.
mu, sigma = pop.mean(), pop.std()
cv = sigma / mu                                   # coefficient of variation (how heavy the tail is)
ns = np.logspace(1.5, 5.5, 150)                   # ~30 ... ~300,000 visits
rel_se = cv / np.sqrt(ns) * 100                   # margin of error as a % of the mean

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(ns, rel_se, color=ni.BLUE, lw=2.4)
for target in [10, 5, 1]:                         # how many visits to hit ±10%, ±5%, ±1%?
    n_need = (cv / (target/100))**2               # invert CV/sqrt(n) = target
    ax.scatter([n_need], [target], color=ni.RED, zorder=5)
    ax.annotate(f"±{target}%  →  {n_need:,.0f} visits", (n_need, target),
                textcoords="offset points", xytext=(10, 8), color=ni.NAVY, fontweight="bold")
ax.set_xscale("log"); ax.set_ylim(0, 50)
ax.set_xlabel("sample size n (log)"); ax.set_ylabel("margin of error on mean EPV (% of the mean)")
ni.titlebox(ax, "The heavy-tail tax: EPV is expensive to measure precisely",
            f"CV = σ/μ ≈ {cv:.1f}  →  pinning mean EPV to ±1% needs ~{(cv/0.01)**2:,.0f} visits")
fig.tight_layout(); ni.savefig(fig, "m1_data_hunger"); plt.show()

print(f"σ={sigma:.1f},  μ={mu:.2f},  CV={cv:.2f}")
for target in [10, 5, 2, 1]:
    print(f"  ±{target:>2}% precision on mean EPV  ->  ~{(cv/(target/100))**2:>9,.0f} visits")
print("\nRule of thumb: a tame metric (CV≈1) hits ±1% at ~10k; EPV (CV≈4.5) needs ~20x more.")
'''),
        md(r'''
### CLT at work: from a skewed metric to a valid decision

We've now watched the means go normal **and** verified the CLT's two promises (centre = μ,
width = σ/√n). Time to *spend* them: two experiments turn that theory into an actual business
call — and, crucially, *check that it's allowed to*.

#### 🔬 Experiment 1 — Can a simple CLT interval drive a real decision?

**The claim being tested.** EPV is heavy-tailed, so any *single* visit is unpredictable — yet the
CLT says the *mean* of a large sample is near-normal, so we should be able to attach a trustworthy
"± margin of error" to mean EPV and act on it.

**How the experiment works.** We take the media buyer's real question — *"Is desktop EPV higher
than mobile, and by how much?"* — compute each device's mean EPV from tens of thousands of visits,
wrap each mean in a textbook CLT interval (`mean ± 1.96·SE`, where `SE = σ/√n`), then form the
**difference** of the two means and its interval.

**What proves the point.** If the CLT is doing its job, the *difference* interval will sit cleanly
away from \$0 — a margin of error tight enough to act on, produced by a one-line formula on
otherwise-ugly data.
'''),
        code(r'''
# --- CLT AT WORK: a real decision the media buyer needs ---
# Q: is DESKTOP EPV higher than MOBILE EPV, and by how much?
# With tens of thousands of visits per group the CLT applies, so we can use the textbook
# "mean ± 1.96 * SE" interval on the MEAN even though EPV itself is heavy-tailed.

def clt_ci(x, z=1.96):
    n = len(x); m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)          # standard error = sigma / sqrt(n)  <- straight from the CLT
    return n, m, se, m - z*se, m + z*se       # 95% interval the CLT licenses

stats_by_dev = {}
for dev in ["desktop", "mobile"]:
    n, m, se, lo, hi = clt_ci(visits.loc[visits.device == dev, "revenue"].values)
    stats_by_dev[dev] = (n, m, se, lo, hi)
    print(f"{dev:>7}: EPV=${m:.3f}  (n={n:,}, SE=${se:.3f})  95% CI [${lo:.3f}, ${hi:.3f}]")

# Difference of two INDEPENDENT means: variances add, so the SEs combine in quadrature.
(nd, md_, sed, *_), (nm, mm, sem, *_) = stats_by_dev["desktop"], stats_by_dev["mobile"]
diff = md_ - mm
se_diff = np.sqrt(sed**2 + sem**2)
dlo, dhi = diff - 1.96*se_diff, diff + 1.96*se_diff
print(f"\nDesktop − Mobile EPV = ${diff:.3f}   95% CI [${dlo:.3f}, ${dhi:.3f}]")
print("The interval is entirely above $0 → the gap is real → safe to act (lean bids to desktop).")

# Visualise the two means with their CLT-based 95% intervals
fig, ax = plt.subplots(figsize=(8.5, 5))
for i, dev in enumerate(["desktop", "mobile"]):
    n, m, se, lo, hi = stats_by_dev[dev]
    ax.bar(i, m, 0.55, color=ni.DEVICE_COLORS[dev])
    ax.errorbar(i, m, yerr=[[m-lo], [hi-m]], color=ni.NAVY, capsize=10, lw=2)   # CLT 95% CI
    ax.text(i, hi + 0.05, f"${m:.2f}", ha="center", fontweight="bold", color=ni.NAVY)
ax.set_xticks([0, 1]); ax.set_xticklabels(["desktop", "mobile"]); ax.set_ylabel("EPV ($)")
ni.titlebox(ax, "CLT in practice: mean EPV with a valid 95% interval",
            f"non-overlapping intervals → desktop EPV is ${diff:.2f} higher (real, not noise)")
fig.tight_layout(); ni.savefig(fig, "m1_clt_decision"); plt.show()
'''),
        md(r'''
**📊 What Experiment 1 showed.** Desktop EPV ≈ \$8.40 and mobile ≈ \$7.53, and the **difference is
≈ \$0.87 with a 95% interval of about [\$0.52, \$1.22]** — entirely above \$0. The whales make
individual visits unpredictable, yet the CLT produced a precise, actionable margin of error on the
*average*. **Point proven:** large-n means are decision-ready even when the raw data is not.

---

#### 🔬 Experiment 2 — Is that CLT interval actually trustworthy at this n?

**The challenge.** The interval in Experiment 1 *assumes* the sample mean is near-normal. How do we
know that assumption has kicked in here, instead of just hoping it has?

**How the experiment works.** We re-estimate the desktop-EPV interval with the **bootstrap** — which
makes **no** distributional assumption: it rebuilds the sampling distribution empirically by
resampling the data 4,000 times and taking each resample's mean.

**What proves the point.** If the assumption-free bootstrap interval lands in the *same place* as
the CLT interval (and the bootstrap distribution sits under the CLT's predicted normal curve), then
the mean really is normal at this n — so the cheap formula is valid. If they disagreed, the CLT
interval could **not** be trusted.
'''),
        code(r'''
# --- Validate the CLT interval against the assumption-free bootstrap ---
# If the two 95% CIs agree, the sample mean really is ~normal at this n (CLT holds).
from scipy.stats import norm
rng = np.random.default_rng(1)
desk = visits.loc[visits.device == "desktop", "revenue"].values

# Bootstrap: resample desktop visits WITH replacement 4000 times; record each resample's mean.
boot_means = np.array([rng.choice(desk, len(desk)).mean() for _ in range(4000)])
b_lo, b_hi = np.percentile(boot_means, [2.5, 97.5])          # bootstrap 95% CI (no normality assumed)
_, _, _, c_lo, c_hi = stats_by_dev["desktop"]                # CLT 95% CI from the previous cell

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(boot_means, bins=50, density=True, color=ni.LIGHT, edgecolor=ni.SKY, label="bootstrap means")
xs = np.linspace(boot_means.min(), boot_means.max(), 200)
ax.plot(xs, norm.pdf(xs, desk.mean(), desk.std(ddof=1)/np.sqrt(len(desk))),
        color=ni.RED, lw=2.2, label="normal predicted by the CLT")
ax.set_xlabel("mean desktop EPV ($)"); ax.set_yticks([]); ax.legend(loc="upper right")
ni.titlebox(ax, "The CLT's prediction matches reality",
            "the bootstrap distribution of the mean is bell-shaped and sits right under the CLT curve")
fig.tight_layout(); ni.savefig(fig, "m1_clt_validates"); plt.show()

print(f"CLT       95% CI: [${c_lo:.3f}, ${c_hi:.3f}]")
print(f"Bootstrap 95% CI: [${b_lo:.3f}, ${b_hi:.3f}]")
print("They nearly coincide → at this n the CLT holds → the simple interval is valid,")
print("and we can make the desktop-vs-mobile call with confidence.")
'''),
        md(r'''
**📊 What Experiment 2 showed.** The CLT interval (≈ [\$8.14, \$8.67]) and the bootstrap interval
(≈ [\$8.15, \$8.68]) are almost identical, and the bootstrap distribution of the mean is a clean
bell sitting right under the CLT's predicted normal curve. **Point proven:** the CLT has genuinely
converged at this n — we didn't *assume* the interval was valid, we *checked* it.

> ✅ **The decision, earned validly:** desktop EPV is reliably higher than mobile (interval clear of
> \$0), and the method behind that interval is verified — so we can lean bids toward desktop with a
> one-line calculation instead of a simulation.
>
> **This is the CLT paying rent:** it converts an unpredictable, whale-ridden metric into a
> *trustworthy average with a valid margin of error* — exactly what a decision needs. The catch is
> the phrase "at this n"… which is precisely where it can go wrong (next).
'''),
        md(r'''
### When the CLT *doesn't* save you — and what breaks

The CLT is **asymptotic**: a promise about "large enough" n. It fails to help when:

| Condition | Why it breaks | NI example |
|---|---|---|
| **n too small for the skew** | Convergence is *slow* for heavy tails — "large enough" can be thousands | one vertical × device × a single day |
| **Infinite / undefined variance** | If σ² isn't finite (extreme power-law tails) the CLT doesn't apply *at all* | a runaway whale distribution |
| **Strong dependence** | Observations aren't independent, so the *effective* n is far smaller | a bot surge or one campaign spike |

**What actually breaks:** your "95% confidence interval" stops containing the truth 95% of the
time — it **under-covers** — and your p-values are mis-calibrated. You *think* you're being
rigorous; you're quietly being overconfident. Let's *measure* it.
'''),
        md(r'''
> **The assumption behind it all: i.i.d.** The CLT wants observations that are **i**ndependent **and
> i**dentically **d**istributed — two *separate* requirements:
> - **Identically distributed** = every observation is a draw from the *same* random variable. Pooling
>   verticals breaks this: a visit's EPV comes from the mortgages distribution (~\$65) or the dating
>   one (~\$4) depending on the row — a **mixture**, not one variable. Slicing to *one vertical × device
>   × day* is how you *buy back* "identically distributed" — but it shrinks n (row 1). You trade a
>   **mixture** problem for a **small-sample** problem.
> - **Independent** = no observation leans on another. Slicing does **not** fix this — repeat users,
>   bot bursts and campaign waves still violate it (row 3, and the demo at the end of this part).
'''),
        code(r'''
# WHY pooling breaks "identically distributed": the pooled data is a MIXTURE of per-vertical
# distributions with very different centres, so its variance is inflated by the BETWEEN-vertical
# spread. Law of total variance:  Var(pooled) = E[within-vertical var] + Var[vertical means]
conv = visits[visits.converted == 1]                                   # converting visits (revenue > 0)
N = len(conv); mu_all = conv["revenue"].mean()
g_var  = conv.groupby("vertical", observed=True)["revenue"].var(ddof=0)  # spread INSIDE each vertical
g_mean = conv.groupby("vertical", observed=True)["revenue"].mean()       # the differing centres
w      = conv.groupby("vertical", observed=True).size() / N             # vertical weights

E_within = float((w * g_var).sum())                       # avg within-vertical variance
Between  = float((w * (g_mean - mu_all)**2).sum())        # variance OF the vertical means (the mixture)
pooled   = conv["revenue"].var(ddof=0)

print("Decomposing the variance of pooled revenue|converted:")
print(f"  pooled variance                  : {pooled:11,.0f}")
print(f"  = avg WITHIN-vertical variance    : {E_within:11,.0f}")
print(f"  + BETWEEN-vertical variance (mix) : {Between:11,.0f}")
print(f"\n  the BETWEEN term is {Between/pooled:.0%} of the total — pure mixture, not visit-to-visit noise.")
print(f"\nHomogeneous slices are far tighter than the pooled mixture:")
print(f"  std POOLED revenue|converted     : ${conv['revenue'].std():9,.1f}")
print(f"  std WITHIN 'dating' only         : ${conv.loc[conv.vertical=='dating','revenue'].std():9,.1f}")
print(f"  std WITHIN 'mortgages' only      : ${conv.loc[conv.vertical=='mortgages','revenue'].std():9,.1f}")
print("\n→ Conditioning on vertical removes the between-group chunk → an (almost) identically")
print("  distributed sample — at the cost of the smaller n inside each cell.")
'''),
        code(r'''
from scipy import stats
rng = np.random.default_rng(7)

def coverage(population, ns, reps=2000, conf=0.95):
    """Empirical coverage: of many textbook 95% t-intervals, what fraction actually contain
    the TRUE mean? If the CLT has kicked in it should be ~95%; if not, it UNDER-covers."""
    mu = population.mean(); out = []
    for n in ns:
        s = rng.choice(population, size=(reps, n))          # 'reps' fresh samples of size n
        m, sd = s.mean(axis=1), s.std(axis=1, ddof=1)       # each sample's mean and SD
        se = sd / np.sqrt(n)                                 # standard error of the mean
        tcrit = stats.t.ppf(0.5 + conf/2, n - 1)             # textbook t critical value
        lo, hi = m - tcrit*se, m + tcrit*se                  # each sample's "95%" CI
        out.append(np.mean((lo <= mu) & (mu <= hi)))         # fraction that caught the truth
    return out

ns = [10, 30, 100, 300, 1000, 5000]
cov_epv  = coverage(pop, ns)                                       # heavy-tailed EPV
tame     = rng.normal(pop.mean(), pop.std()/3, size=200_000)       # a well-behaved control group
cov_tame = coverage(tame, ns)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(ns, cov_epv,  "-o", color=ni.RED,   lw=2.4, label="heavy-tailed EPV")
ax.plot(ns, cov_tame, "-o", color=ni.GREEN, lw=2.4, label="tame (normal) data")
ax.axhline(0.95, color=ni.NAVY, ls="--", lw=1.6, label="nominal 95%")
ax.set_xscale("log"); ax.set_ylim(0.60, 1.0)
ax.set_xlabel("sample size n (log)"); ax.set_ylabel("actual coverage of a '95%' CI"); ax.legend()
ni.titlebox(ax, "A '95%' interval isn't 95% when the CLT hasn't kicked in",
            "on heavy-tailed EPV a small-n CI can cover as little as ~65% — overconfident by design")
fig.tight_layout(); ni.savefig(fig, "m1_clt_coverage"); plt.show()

for n, c in zip(ns, cov_epv):
    print(f"  EPV  n={n:>4}:  a nominal-95% CI actually covers {c:.1%}")
'''),
        md(r'''
> **The lesson from the breakdown:** at small n on EPV, a "95%" CI may really cover only ~65–88%
> — you'd reject true nulls and ship false findings far more often than you think. The fixes are
> exactly the branch-2/3 tools: **get more data**, use the **bootstrap** (which re-derives the
> sampling distribution empirically instead of assuming normality), or a **rank test** (which
> doesn't depend on the mean at all). This is *why* Module 2 obsesses over EPV's shape, and
> Module 5 bootstraps its intervals.
'''),
        md(r'''
### Adtech reality check: when to distrust the CLT before betting budget

Two situations should make you suspicious of *any* `σ/√n` interval — both everywhere in adtech:

**1 · Too little data behind a slice** *(the breakdown above).* A niche vertical × device × a single
day can be a few hundred heavy-tailed visits, so the mean and its interval are unreliable and the
"95%" CI is really ~75–85%. *Decision risk:* you "spot a winner" in a thin segment and move bids on
pure noise.

**2 · The observations aren't independent** — the failure analysts miss most, because the math still
*runs*; it just lies. `σ/√n` assumes independent visits. In adtech they rarely are:

- **Intra-user correlation** — one visitor generates 10 sessions; those 10 EPVs move together, so
  they are nowhere near 10 *independent* facts (randomization unit = user, analysis unit = visit;
  Deng et al. 2018, Kohavi et al. 2020).
- **Campaign / creative bursts** — one creative or a budget spike floods you with similar traffic:
  thousands of rows, little independent information.
- **Bot / fraud surges & time-of-day waves** — correlated clumps that inflate counts without adding
  signal.
- **Ratio metrics (EPV, CTR, ROAS)** — a ratio's variance is *not* a mean's SE at all.

When observations cluster, the **true** standard error is *larger* than `σ/√n` — often several times
larger — so the naive interval is too narrow and you over-declare "significant" wins. Let's measure it
on a classic adtech pattern: many visits per user.
'''),
        code(r'''
# Adtech CLUSTERING: 2,000 users x 10 visits each = 20,000 visits, where each user's visits
# share a user-level effect -> visits within a user are CORRELATED, not independent.
rng = np.random.default_rng(3)
n_users, k = 2000, 10

def make_clustered():
    user_effect = rng.normal(0, 4.0, n_users)        # shared by all of a user's visits (the correlation)
    noise       = rng.normal(0, 3.0, (n_users, k))   # independent visit-level noise
    return (user_effect[:, None] + noise).ravel()    # 20,000 visit-level values

# TRUE spread of the dataset mean under clustering: simulate many whole datasets, look at the means.
true_means = np.array([make_clustered().mean() for _ in range(3000)])
true_se = true_means.std()

# NAIVE SE: take ONE dataset and pretend all 20,000 visits are independent (the usual mistake).
one = make_clustered()
naive_se = one.std(ddof=1) / np.sqrt(len(one))

inflation = true_se / naive_se
print(f"naive  SE (assumes 20,000 independent visits): {naive_se:.4f}")
print(f"true   SE (accounts for the 2,000 user clusters): {true_se:.4f}")
print(f"-> real uncertainty is {inflation:.1f}x larger than the naive CLT interval admits")
print(f"-> a naive '95%' CI is ~{inflation:.1f}x too NARROW: you'd call noise a 'significant' win")

from scipy.stats import norm
fig, ax = plt.subplots(figsize=(10.5, 5))
ax.hist(true_means, bins=45, density=True, color=ni.LIGHT, edgecolor=ni.SKY, label="true means (clustered)")
xs = np.linspace(true_means.min(), true_means.max(), 200)
ax.plot(xs, norm.pdf(xs, true_means.mean(), naive_se), color=ni.RED, lw=2.2,
        label="what NAIVE σ/√n predicts (too narrow)")
ax.plot(xs, norm.pdf(xs, true_means.mean(), true_se), color=ni.NAVY, lw=2.2, label="the TRUE spread")
ax.set_yticks([]); ax.set_xlabel("dataset mean EPV"); ax.legend()
ni.titlebox(ax, "Dependence breaks σ/√n: the naive interval is far too narrow",
            f"correlated visits → true SE is {inflation:.1f}× the naive one (overconfidence by design)")
fig.tight_layout(); ni.savefig(fig, "m1_clustering"); plt.show()
'''),
        md(r'''
### Taking the wrong decision on dependent data — an A/B test that ships noise

The inflated SE isn't academic — it flips real decisions. Here's the classic adtech mistake: we A/B
test a new landing page **B** against control **A** on EPV. **In truth B does nothing** (A and B are
identical). But visits cluster by user, and we analyse at the **visit** level with a t-test that
assumes every visit is independent. Repeat this *null* experiment 1,000 times and ask how often each
method falsely declares a winner — a correct test should be fooled only ~5% of the time (that's α).
'''),
        code(r'''
from scipy import stats
rng = np.random.default_rng(20)
n_users, k = 300, 20                        # users are the randomization unit; k visits each, per arm

def one_null_experiment():
    # NULL world: A and B identical. Each user's k visits share a user effect -> correlated (dependent).
    def arm():
        u = rng.normal(0, 2.0, n_users)                          # user-level effect = the dependence
        return u[:, None] + rng.normal(0, 4.0, (n_users, k))     # (users x visits), pure normal noise
    A, B = arm(), arm()
    p_naive = stats.ttest_ind(A.ravel(), B.ravel(), equal_var=False).pvalue   # WRONG: visits 'independent'
    p_user  = stats.ttest_ind(A.mean(1),  B.mean(1),  equal_var=False).pvalue # RIGHT: one value per user
    return p_naive, p_user

res = np.array([one_null_experiment() for _ in range(1000)])
fp_naive = (res[:, 0] < 0.05).mean()
fp_user  = (res[:, 1] < 0.05).mean()
print("False-positive rate over 1,000 NULL A/B tests (truth: A and B identical):")
print(f"  NAIVE visit-level t-test : {fp_naive:5.0%}    <- should be 5%")
print(f"  CORRECT user-level t-test: {fp_user:5.0%}    <- ~5%, as intended")
print(f"\n→ the naive test cries 'significant winner!' about {fp_naive/0.05:.0f}x too often.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for a, col, title, fp, color in [
        (ax[0], 0, "NAIVE visit-level (visits 'independent')", fp_naive, ni.RED),
        (ax[1], 1, "CORRECT user-level (one value per user)",  fp_user,  ni.GREEN)]:
    a.hist(res[:, col], bins=20, range=(0, 1), color=color, alpha=.85, edgecolor="white")
    a.axvline(0.05, color=ni.NAVY, ls="--", lw=1.6)
    ni.titlebox(a, title, f"false-positive rate = {fp:.0%}   (target 5%)")
    a.set_xlabel("p-value under the NULL (no real effect)")
ax[0].set_ylabel("# of A/B tests")
fig.suptitle("Dependent data ships noise: naive p-values pile up below 0.05",
             fontsize=14, fontweight="bold", color=ni.NAVY, y=1.04)
fig.tight_layout(); ni.savefig(fig, "m1_dependence_decision"); plt.show()
'''),
        md(r'''
> **The wrong decision, concretely:** on dependent data the naive visit-level test declares a "winner"
> far more than 5% of the time — so you'd routinely report *"landing page B lifts EPV, p<0.05 — ship it
> and shift budget,"* when **B does nothing**. The maths was fine; the *inputs weren't independent*. The
> correct user-level test stays at ~5%, as it should.
'''),
        md(r'''
> **What this means for the business:** that dataset has 20,000 rows, but because the visits cluster
> into 2,000 users it carries the *information* of far fewer independent observations. Trust the naive
> `σ/√n` interval and you'd announce a "statistically significant" lift and reallocate budget — when
> the true uncertainty is ~2–3× wider and the "win" is noise.
>
> **Be suspicious whenever:** the slice is thin; a few users / campaigns / creatives dominate the rows;
> traffic arrived in bursts; or the metric is a ratio (EPV, CTR, ROAS). **The fixes:** resample or
> cluster by **user** (not visit), use the **delta method** for ratio metrics (Module 5), or collect
> more *independent* units — **more users, not more visits from the same users.**

> 🧭 **CLT trust checklist** — before acting on a `mean ± 1.96·SE` interval, confirm: (a) enough data
> behind *this* slice; (b) observations are roughly independent (no user/campaign/bot clustering);
> (c) finite variance / not a raw ratio. Fail any one → use the bootstrap or delta method instead.
'''),

        # ------------------------------------------------------------------ #
        # PART 2 — INSPECT THE DATA
        # ------------------------------------------------------------------ #
        md(r'''
# Part 2 · Inspecting the NI data

We've used EPV and conversions to *illustrate* the theory. Now, before picking tools, let's lay out
the actual table we'll analyse — the **grain** and the **KPIs** computed from it.

## One row per **visit**

Dimensions describe each visit; `converted` / `revenue` / `cost` are its outcomes. The KPIs are just
**averages over this grain**: `conversion_rate = mean(converted)` and `EPV = mean(revenue)`.
'''),
        code(r'''
# Each row is ONE visit. Dimensions describe it; converted/revenue/cost are the outcomes.
print("Grain check — one row per visit:")
display(visits[["date", "engine", "device", "vertical", "converted", "revenue", "cost"]].head())

# The two headline KPIs are just column averages over the grain:
#   conversion_rate = mean of the 0/1 'converted' column  -> a PROPORTION
#   EPV             = mean of the 'revenue' column         -> a heavy-tailed MEAN
print("\nHeadline metrics, computed from the grain:")
print(f"  overall conversion rate : {visits['converted'].mean():.1%}")
print(f"  overall EPV             : ${visits['revenue'].mean():.2f}")
print(f"  overall CPV             : ${visits['cost'].mean():.2f}")
print(f"  profit / visit          : ${visits['profit'].mean():.2f}")
'''),

        # ------------------------------------------------------------------ #
        # PART 3 — TOOL SELECTION
        # ------------------------------------------------------------------ #
        md(r'''
# Part 3 · Choosing the tool: it *follows* from the distribution

Now the payoff. Because the distribution fixes the **variance structure**, it fixes the right
method. Route any request with three questions:

1. **What is the unit?** (here: a visit)
2. **What *type* is the outcome?** binary (proportion) · continuous-skewed (mean) ·
   category × category (association) · a ratio
3. **What am I comparing or relating?** two groups · a trend · a relationship

The same NI dataset wears three statistical "faces" — each needs a different tool:
'''),
        code(r'''
# The SAME data, shown as the three outcome TYPES that drive tool choice.
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (A) BINARY outcome -> a PROPORTION (conversion rate by engine)
cr = visits.groupby("engine", observed=True)["converted"].mean().reindex(["Google","Bing","Organic","Social"])
ax[0].bar(cr.index, cr.values, color=[ni.ENGINE_COLORS[e] for e in cr.index])
ax[0].set_ylim(0, 0.55); ax[0].set_ylabel("conversion rate")
for i, v in enumerate(cr.values):
    ax[0].text(i, v + 0.012, f"{v:.0%}", ha="center", fontweight="bold", color=ni.NAVY)
ni.titlebox(ax[0], "A · Binary outcome", "convert: yes/no  →  a PROPORTION")

# (B) CONTINUOUS skewed outcome -> a heavy-tailed MEAN (EPV / revenue)
rev = visits.loc[visits.converted == 1, "revenue"]
ax[1].hist(rev.clip(upper=120), bins=60, color=ni.BLUE, alpha=0.85)
ax[1].axvline(rev.mean(), color=ni.RED, lw=2, label=f"mean ${rev.mean():.0f}")
ax[1].axvline(rev.median(), color=ni.GREEN, lw=2, label=f"median ${rev.median():.0f}")
ax[1].set_xlabel("revenue per converting visit ($)"); ax[1].legend()
ni.titlebox(ax[1], "B · Continuous & skewed", "EPV / revenue  →  a heavy-tailed MEAN")

# (C) CATEGORY x CATEGORY -> an ASSOCIATION (engine x device conversion grid)
piv = visits.pivot_table(index="device", columns="engine", values="converted", aggfunc="mean")
piv = piv.reindex(index=["desktop","mobile","tablet"], columns=["Google","Bing","Organic","Social"])
im = ax[2].imshow(piv.values, cmap="Blues", aspect="auto", vmin=0.25, vmax=0.47)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(piv.columns, rotation=20)
ax[2].set_yticks(range(3)); ax[2].set_yticklabels(piv.index)
for r in range(piv.shape[0]):
    for c in range(piv.shape[1]):
        ax[2].text(c, r, f"{piv.values[r,c]:.0%}", ha="center", va="center",
                   color="white" if piv.values[r,c] > 0.4 else ni.NAVY, fontweight="bold")
ax[2].grid(False)
ni.titlebox(ax[2], "C · Category × Category", "engine × device  →  an ASSOCIATION")

fig.suptitle("One dataset, three statistical faces — each picks a different tool", fontsize=15,
             fontweight="bold", color=ni.NAVY, x=0.5, y=1.06)
fig.tight_layout(); ni.savefig(fig, "m1_variable_types"); plt.show()
'''),
        md(r'''
### The tool-selection decision tree

Pin this to your monitor. Start at the top with *"what type is my outcome?"*
'''),
        code(r'''
# This cell just DRAWS the flowchart (boxes + arrows) — no statistics here.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, xy, w, h, text, fc, tc="white", fs=11):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", wrap=True)

def arrow(ax, p0, p1, label=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=ni.GREY, shrinkA=2, shrinkB=2))
    if label:
        ax.text((p0[0]+p1[0])/2, (p0[1]+p1[1])/2 + 0.015, label, ha="center",
                fontsize=9.5, color=ni.NAVY, style="italic")

fig, ax = plt.subplots(figsize=(13, 6.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

box(ax, (6, 8.3), 5.4, 0.9, "What is my OUTCOME variable?", ni.NAVY, fs=12.5)
box(ax, (2.2, 6.2), 3.4, 1.0, "Binary\n(converted yes/no)", ni.BLUE)
box(ax, (6, 6.2), 3.4, 1.0, "Continuous\n(EPV / revenue)", ni.TEAL)
box(ax, (9.8, 6.2), 3.4, 1.0, "Two categories\n(engine × device)", ni.ORANGE)
arrow(ax, (4.6, 7.85), (2.6, 6.7))
arrow(ax, (6, 7.85), (6, 6.7))
arrow(ax, (7.4, 7.85), (9.4, 6.7))

box(ax, (2.2, 4.0), 3.6, 1.5,
    "Compare 2 groups → 2-proportion z-test\nUncertainty → Wilson CI\nMany groups → chi-square", ni.LIGHT, tc=ni.NAVY, fs=9.5)
arrow(ax, (2.2, 5.7), (2.2, 4.75))
box(ax, (6, 4.0), 3.6, 1.5,
    "Skewed? → Mann-Whitney /\nbootstrap / log-transform\nUncertainty → bootstrap CI", ni.LIGHT, tc=ni.NAVY, fs=9.5)
arrow(ax, (6, 5.7), (6, 4.75))
box(ax, (9.8, 4.0), 3.6, 1.5,
    "Association → chi-square test\nStrength → Cramér's V\nTrend → rank correlation", ni.LIGHT, tc=ni.NAVY, fs=9.5)
arrow(ax, (9.8, 5.7), (9.8, 4.75))

box(ax, (6, 1.6), 9.5, 1.1,
    "ALWAYS first: plot the distribution · check sample size & power · ask \"how many tests did I run?\"",
    ni.GOLD, tc=ni.NAVY, fs=11)
for x0 in (2.2, 6, 9.8):
    arrow(ax, (x0, 3.25), (x0 if x0==6 else 6, 2.18))

ax.text(0, 8.9, "NI Analyst's Tool-Selection Tree", fontsize=15, fontweight="bold", color=ni.NAVY)
ni.savefig(fig, "m1_decision_tree"); plt.show()
'''),
        md(r'''
### Routing the five workshop use cases

Every use case from the syllabus maps cleanly onto the tree:
'''),
        code(r'''
# Map each real NI question -> its outcome TYPE -> the right tool -> the module that covers it.
mapping = pd.DataFrame([
    ["Google vs Bing conversion (45% vs 35%)", "Binary proportion", "2-proportion z-test + Wilson CI", "Module 4 & 5"],
    ["EPV vs day-of-week", "Continuous (skewed) + trend", "Spearman correlation (+ control for mix)", "Module 3"],
    ["Outliers / whales distorting EPV", "Continuous, heavy-tailed", "Distribution + median/transform", "Module 2"],
    ["Mobile vs desktop EPV gap", "Continuous ratio", "Bootstrap CI for the ratio", "Module 5"],
    ["Mobile EPV CI [-20%, -5%]", "Interval estimate", "Confidence interval (communication)", "Module 5"],
    ["Should we move budget? (all of it)", "Decision", "End-to-end workflow", "Module 6"],
], columns=["NI use case", "Variable type", "Right tool", "Covered in"])

mapping.style.hide(axis="index").set_caption(
    "From business question → statistical tool").set_properties(**{"text-align":"left"})
'''),
        md(r'''
## A worked guide: business decision → tool → *why that tool*

The table is the index; here is the *reasoning* — for each real NI decision, the tool to reach for
and **why it beats the obvious alternative.**

**1 · "Bing's clicks are cheaper — shift budget from Google to Bing?"**
→ testable sub-question: *do the engines convert at different rates?*
- **Outcome:** binary (converted 0/1) = a proportion. **Tool:** two-proportion **z-test + Wilson CI**.
- **Why:** a 0/1 outcome's variance is *fixed by its mean* (p(1−p)), so it needs a proportion test, not a
  t-test (which assumes a free-floating continuous spread). Wilson over the textbook Wald interval
  because Wald mis-covers. *Caveat:* conversion rate alone never decides budget — pair it with #5.

**2 · "Is mobile EPV really lower than desktop — trim mobile bids?"**
- **Outcome:** continuous, heavy-tailed (EPV). **Tool:** **Welch's t-test** on the means (n huge → CLT
  valid), *confirmed* with **Mann–Whitney** and a **bootstrap**.
- **Why:** you're comparing *means of a continuous metric*, so a proportion test doesn't apply. Welch
  (not Student's) because the groups' variances differ. You confirm with the rank test + bootstrap
  because whales make you distrust the mean's normality on any single slice — if all three agree, act.

**3 · "By *how much* is mobile EPV down — enough to matter?"**
- **Outcome:** a **ratio** (EPV = revenue ÷ visits); you want a magnitude **+ interval**. **Tool:**
  **bootstrap CI for the ratio** (or the **delta method**).
- **Why:** a ratio of random quantities has *no* plain mean's standard error, so naive ±1.96·SE is the
  wrong width. And an *interval* — not a p-value — is what tells you the effect is big enough to act on.

**4 · "Does EPV swing by day of week — bid differently on weekends?"**
- **Outcome:** a relationship on skewed data, at risk of a confound. **Tool:** **Spearman correlation**,
  then **condition on vertical**.
- **Why:** Spearman (rank) over Pearson because whales + non-linearity wreck Pearson. And you control
  for vertical because the raw "day effect" is a traffic-**mix** confound (Module 3) — tool *plus* design.

**5 · "So — do we move the budget at all?"** *(the decision that actually matters)*
- **Outcome:** **profit per visit** (= EPV − CPV) per segment, with uncertainty. **Tool:** per-segment
  **profit/visit with bootstrap CIs**; act only where the interval clears your threshold.
- **Why:** budget follows *profit*, not conversion rate or EPV alone; the **interval**, not the point,
  sets the decision rule (Module 6).

**6 · "The A/B test says the new page wins — ship it?"** *(dependence!)*
- **Outcome:** a mean comparison on data **clustered by user**. **Tool:** test at the **randomization
  unit** (one value per user), or a **cluster bootstrap** / **delta method** — *never* a visit-level t-test.
- **Why:** visits within a user are dependent, so a visit-level test underestimates variance and fires
  false winners ~8× too often (we measured it). *Pick which:* user-level t-test when you can aggregate
  cleanly; cluster bootstrap when the metric is a ratio or the aggregation is messy.

**7 · "We sliced 40 segments and 3 look significant — real?"**
- **Tool:** a **multiple-comparison correction** (Benjamini–Hochberg FDR, or Bonferroni if few tests).
- **Why:** 40 null tests at α=0.05 expect ~2 false hits; correction stops you chasing noise (Module 4).

**8 · "Before launch, how much traffic to detect a 2% EPV lift?"**
- **Tool:** a **power / sample-size calculation** (done *before* the test).
- **Why:** so you're neither underpowered (missing real effects) nor tempted to peek (inflating false
  positives). Recall the heavy-tail tax: small effects on EPV need a lot of traffic.

> **The pattern:** binary→proportion test · continuous→mean test (rank/bootstrap if skewed or thin) ·
> ratio→bootstrap/delta · relationship→rank correlation · dependent→cluster/unit-level · many tests→
> correct · and *every* decision ends on an **interval on profit-per-visit**, not a bare p-value.
'''),
        md(r'''
## Why these tools? The reasoning — and the evidence

Tool choice isn't taste; it follows from the **variance structure** of the outcome, which is
itself fixed by the variable's *measurement scale* (Stevens 1946). Each branch of the tree
above has a documented justification:

- **Binary outcome → proportion methods.** A 0/1 mean *is* a proportion; its variance is
  p(1−p), set by the mean. Use the two-proportion z-test / chi-square, and the **Wilson**
  interval — the textbook **Wald** interval has famously erratic coverage (Brown, Cai &
  DasGupta 2001).
- **Continuous outcome → means via the CLT, but watch the shape.** In *large* samples a
  mean-based t-test is valid for *any* distribution (Lumley et al. 2002) — skew alone is not a
  reason to abandon it. The reasons to switch to **rank tests** (Mann-Whitney 1947 / Wilcoxon
  1945) or the **bootstrap** (Efron 1979) are *small slices* and *whales that destroy power*,
  not validity.
- **Ratio metric (EPV) → delta method / bootstrap.** EPV is ΣRevenue/ΣVisits — a ratio of
  random quantities whose variance is NOT a mean's standard error (Deng et al. 2018; Kohavi,
  Tang & Xu 2020).
- **Relationship → rank correlation.** Pearson is linear and outlier-sensitive; Spearman
  (1904) / Kendall (1938) use ranks and resist whales.
- **Always:** correct for **multiple comparisons** (Benjamini & Hochberg 1995) and check
  **power** (Cohen 1988).

📚 **Sources** — Stevens (1946) *Science* 103:677–680 · Brown, Cai & DasGupta (2001)
*Statistical Science* 16:101–133 · Lumley et al. (2002) *Annu. Rev. Public Health* 23:151–169 ·
Mann & Whitney (1947) *Ann. Math. Stat.* 18:50–60 · Wilcoxon (1945) *Biometrics Bull.* 1:80–83 ·
Efron (1979) *Ann. Stat.* 7:1–26 · Deng, Knoblich & Lu (2018) *KDD* (arXiv:1803.06336) ·
Kohavi, Tang & Xu (2020) *Trustworthy Online Controlled Experiments*, Cambridge ·
Spearman (1904) *Am. J. Psychol.* 15:72–101 · Kendall (1938) *Biometrika* 30:81–93 ·
Benjamini & Hochberg (1995) *JRSS-B* 57:289–300 · Cohen (1988) *Statistical Power Analysis*, 2nd ed.
'''),
        md(r'''
### ✅ Takeaway

> **Shape first, then theory, then tool.**
> 1. Identify the **distribution** (Part 1) — it tells you whether the mean is even trustworthy.
> 2. Lean on the **CLT** (Part 1) — big-n averages are reliable; small-n heavy-tailed ones are not.
> 3. Pick the **tool from the variable type** (Part 3), not from habit.
>
> The Slack question is secretly **two** questions — a *proportion* comparison (Google vs Bing)
> and a *ratio* comparison (mobile vs desktop EPV) — answered across the next five modules.

**Next:** zoom into EPV's shape and the whales that make its mean lie. → *Module 2.*
'''),
    ]
    build("01_foundations_tool_selection.ipynb", cells)


# =========================================================================== #
# MODULE 2 — Outliers, skewness, transformations
# =========================================================================== #
def module2():
    cells = [
        md(r'''
# Module 2 — Look Before You Leap: Outliers, Skewness & Transformations
### Practical Statistics for Analysts @ Natural Intelligence

EPV is the most important number at NI — and the most **dangerous to average naively**.
Most visits earn **\$0**; a handful of conversions (a mortgage lead!) earn hundreds.
That shape — a spike at zero plus a long right tail — means **the mean is unstable**
and a single "whale" can swing your daily report.

This module: see the shape, find the outliers, fix it responsibly.
'''),
        code(BOOT),
        md(r'''
## 1. The mean lies — see the shape first

Three views of the same revenue data: all visits (the zero spike), converting
visits only (the long tail), and the log-transform that tames it.
'''),
        code(r'''
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# A — all visits: the zero spike dominates
ax[0].hist(visits["revenue"].clip(upper=60), bins=60, color=ni.NAVY, alpha=0.85)
ax[0].set_xlabel("revenue per visit ($)"); ax[0].set_ylabel("# visits")
ni.titlebox(ax[0], "A · All visits", f"{(visits.revenue==0).mean():.0%} earn exactly $0")

# B — converting visits only: heavy right tail, mean >> median
rev = visits.loc[visits.converted == 1, "revenue"]
ax[1].hist(rev.clip(upper=150), bins=60, color=ni.BLUE, alpha=0.85)
ax[1].axvline(rev.mean(), color=ni.RED, lw=2.2, label=f"mean ${rev.mean():.1f}")
ax[1].axvline(rev.median(), color=ni.GREEN, lw=2.2, label=f"median ${rev.median():.1f}")
ax[1].set_xlabel("revenue | converted ($)"); ax[1].legend()
ni.titlebox(ax[1], "B · Converters only", f"skew = {stats.skew(rev):.1f}  (mean is pulled right)")

# C — log1p transform: roughly symmetric, analyzable
ax[2].hist(np.log1p(rev), bins=60, color=ni.TEAL, alpha=0.85)
ax[2].set_xlabel("log(1 + revenue)")
ni.titlebox(ax[2], "C · After log transform", "the tail is tamed → symmetric-ish")

fig.suptitle("EPV's true shape: a wall of zeros + a long tail of whales",
             fontsize=15, fontweight="bold", color=ni.NAVY, y=1.06)
fig.tight_layout(); ni.savefig(fig, "m2_distributions"); plt.show()

print(f"Top 1% of converters earn {rev[rev>=rev.quantile(.99)].sum()/rev.sum():.0%} of ALL revenue.")
'''),
        md(r'''
## 2. Daily EPV is mostly noise — a whale is just the clearest culprit

At ~2,000 visits/day, EPV swings violently from day to day. It is tempting to read a daily spike as
a "trend" and react — but at this resolution the number is dominated by **noise** (with the
occasional whale on top). Two things make this graph honest, and they correct a common misreading:
1. a **7-day rolling mean** shows the *real* signal hiding under the daily noise, and
2. removing each day's single biggest whale **barely calms the line** — proof the problem is
   pervasive small-n noise, not one deletable outlier.
'''),
        code(r'''
daily = visits.groupby("date", observed=True).agg(
    revenue=("revenue", "sum"), visits=("visit_id", "size"), biggest=("revenue", "max")).reset_index()
daily["EPV"] = daily.revenue / daily.visits

# (a) daily EPV with each day's single biggest conversion removed
daily["EPV_ex_whale"] = (daily.revenue - daily.biggest) / (daily.visits - 1)
# (b) the 7-day rolling mean = the stable underlying signal (aggregation beats deletion)
daily["EPV_roll7"] = daily["EPV"].rolling(7, center=True).mean()

def cv(x):                                   # day-to-day variation = std / mean
    x = x.dropna(); return x.std() / x.mean()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily.date, daily.EPV, color=ni.BLUE, lw=1.6, alpha=0.9,
        label=f"daily EPV — as reported   (swings ±{cv(daily.EPV):.0%})")
ax.plot(daily.date, daily.EPV_ex_whale, color=ni.GREY, lw=1.3, ls="--",
        label=f"daily EPV minus the day's biggest whale   (still ±{cv(daily.EPV_ex_whale):.0%} — barely helps)")
ax.plot(daily.date, daily.EPV_roll7, color=ni.NAVY, lw=3,
        label=f"7-day rolling mean — the real signal   (±{cv(daily.EPV_roll7):.0%}, ~flat)")
spike = daily.loc[daily.EPV.idxmax()]
ax.annotate(f"clearest single whale\n(${spike.biggest:,.0f}) — but just ONE\nof many noisy days",
            xy=(spike.date, spike.EPV), xytext=(spike.date, spike.EPV * 1.13),
            ha="center", color=ni.RED, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=ni.RED, lw=1.6))
ax.set_ylabel("EPV ($)"); ax.legend(loc="upper left", fontsize=9)
ni.titlebox(ax, "Don't read EPV at daily resolution — it's noise around a flat trend",
            "each day is ~2,000 visits; deleting the whale barely calms it, but aggregating does")
fig.tight_layout(); ni.savefig(fig, "m2_whale_timeseries"); plt.show()

print(f"day-to-day variation (CV):  raw {cv(daily.EPV):.0%}   |   minus biggest whale "
      f"{cv(daily.EPV_ex_whale):.0%}   |   7-day rolling {cv(daily.EPV_roll7):.0%}")
print("Removing the single biggest whale leaves ~98% of the swing → the noise is pervasive, not one whale.")
'''),
        md(r'''
> **So which spikes do we ignore? All of them.** The marked day is just the one where a *single*
> conversion is the obvious culprit — but the de-whaled line proves the rest of the line is *equally*
> erratic for diffuse reasons (small n + multiple whales + day-of-week mix). This is Module 1's wide
> daily sampling distribution in time-series form. The rule isn't "ignore that one peak" — it's
> **never act on a single day's EPV; read the 7-day trend, or test the difference properly.**
'''),
        md(r'''
## 3. Finding outliers: visual + rule-based

Boxplots on a **log scale** reveal the whales by vertical; the IQR and z-score
rules flag them numerically. (Note: with skewed data the IQR rule is more
trustworthy than the z-score, which is itself distorted by the outliers.)
'''),
        code(r'''
fig, ax = plt.subplots(figsize=(13, 5))
order = visits.groupby("vertical")["revenue"].median().sort_values().index.tolist()
data = [visits.loc[(visits.vertical==v) & (visits.converted==1), "revenue"].values for v in order]
bp = ax.boxplot(data, vert=True, patch_artist=True, labels=order, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.25, markerfacecolor=ni.RED, markeredgecolor="none"))
for patch in bp["boxes"]:
    patch.set(facecolor=ni.LIGHT, edgecolor=ni.NAVY)
for med in bp["medians"]:
    med.set(color=ni.NAVY, linewidth=2)
ax.set_yscale("log"); ax.set_ylabel("revenue | converted ($, log scale)")
plt.setp(ax.get_xticklabels(), rotation=20)
ni.titlebox(ax, "Revenue by vertical — whales live in finance",
            "red points are outliers; finance verticals have the fattest tails")
fig.tight_layout(); ni.savefig(fig, "m2_outliers_box"); plt.show()

# Rule-based counts
def outlier_counts(x):
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    iqr_out = (x > q3 + 1.5*iqr).sum()
    z = (x - x.mean())/x.std(); z_out = (np.abs(z) > 3).sum()
    return iqr_out, z_out
io, zo = outlier_counts(rev.values)
print(f"Converting visits flagged as high outliers:  IQR rule = {io:,}   |   z>3 rule = {zo:,}")
print("→ The IQR rule catches the heavy tail; the z-score under-flags because the whales inflate the SD.")
'''),
        md(r'''
## 4. Does the choice of summary change the decision?

Mean vs median vs 5%-trimmed mean by device — notice the **median and trimmed
mean are far more stable**, and they still preserve the mobile-vs-desktop gap
we care about (without being hostage to whales).
'''),
        code(r'''
def trimmed_mean(x, p=0.05): return stats.trim_mean(x, p)
g = visits.groupby("device", observed=True)["revenue"]
summ = pd.DataFrame({
    "mean (raw EPV)": g.mean(),
    "median": g.median(),
    "5% trimmed mean": g.apply(lambda s: trimmed_mean(s.values)),
}).reindex(["desktop","mobile","tablet"])

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(summ)); w = 0.26
for i, col in enumerate(summ.columns):
    ax.bar(x + (i-1)*w, summ[col], w, label=col, color=[ni.NAVY, ni.BLUE, ni.TEAL][i])
ax.set_xticks(x); ax.set_xticklabels(summ.index); ax.set_ylabel("$ per visit")
ax.legend()
ni.titlebox(ax, "Mean vs median vs trimmed mean (EPV by device)",
            "the mean is inflated by whales; robust summaries still show desktop > mobile")
fig.tight_layout(); ni.savefig(fig, "m2_robust_summaries"); plt.show()
display(summ.round(2))
'''),
        md(r'''
### ✅ Takeaway

> **Always plot the distribution before you quote an average.**
> For EPV at NI: report the **mean** (it's what the business banks) **but** sanity-check
> with the **median**, flag outliers with the **IQR rule**, and consider a
> **log transform** or **bootstrap** for any test. A spike in mean EPV is a *whale
> question* before it's a *trend question*.

**Next:** now that we respect EPV's shape, can we trust a *relationship* —
like "EPV depends on the day of week"? → *Module 3.*
'''),
    ]
    build("02_outliers_skewness.ipynb", cells)


# =========================================================================== #
# MODULE 3 — Correlation: Pearson / Spearman / Kendall + the confound
# =========================================================================== #
def module3():
    cells = [
        md(r'''
# Module 3 — Measuring Relationships: Correlation (and its traps)
### Practical Statistics for Analysts @ Natural Intelligence

"EPV varies by day of week." Maybe — but **correlation is a hypothesis, not a finding.**
Two traps bite NI analysts constantly:
1. **Outliers distort Pearson** correlation → rank-based (Spearman/Kendall) is safer on EPV.
2. **A confound fakes a relationship** → the day-of-week "effect" is really a *traffic-mix* effect.
'''),
        code(BOOT),
        code(r'''
# Daily aggregates we'll relate to each other
daily = visits.groupby("date", observed=True).agg(
    EPV=("revenue","mean"), CPV=("cost","mean"),
    conv=("converted","mean"), visits=("visit_id","size")).reset_index()
daily["dow_idx"] = daily["date"].dt.dayofweek           # 0=Mon..6=Sun
daily["dow"] = pd.Categorical(daily["date"].dt.strftime("%a"), categories=ni.DOW_ORDER, ordered=True)
daily.head()
'''),
        md(r'''
## 1. The raw picture: EPV by day of week

There *is* a striking weekly pattern — weekdays look far richer than weekends.
'''),
        code(r'''
by_dow = visits.groupby("day_of_week", observed=True)["revenue"].mean().reindex(ni.DOW_ORDER)
fig, ax = plt.subplots(figsize=(11, 5))
colors = [ni.BLUE if d not in ("Sat","Sun") else ni.ORANGE for d in by_dow.index]
ax.bar(by_dow.index, by_dow.values, color=colors)
for i, v in enumerate(by_dow.values):
    ax.text(i, v+0.1, f"${v:.1f}", ha="center", fontweight="bold", color=ni.NAVY)
ax.set_ylabel("EPV ($)")
ni.titlebox(ax, "EPV by day of week — a big weekend drop",
            "orange = weekend.  Looks like a strong 'day' effect… is it real?")
fig.tight_layout(); ni.savefig(fig, "m3_epv_by_dow"); plt.show()
'''),
        md(r'''
## 2. Three correlation coefficients — and why Pearson can mislead

**Pearson** measures *linear* association and is sensitive to outliers.
**Spearman** and **Kendall** work on *ranks* — robust to skew and whales.
On NI metrics, **default to Spearman.**
'''),
        code(r'''
# Relate daily EPV to daily cost — outlier days make Pearson and Spearman disagree
x, y = daily["CPV"].values, daily["EPV"].values
pear = stats.pearsonr(x, y); spear = stats.spearmanr(x, y); kend = stats.kendalltau(x, y)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(x, y, s=42, color=ni.BLUE, alpha=0.7, edgecolor="white")
# highlight whale-driven outlier days
hi = y > np.percentile(y, 95)
ax[0].scatter(x[hi], y[hi], s=90, facecolor="none", edgecolor=ni.RED, linewidth=2, label="outlier days (whales)")
ax[0].set_xlabel("daily cost per visit ($)"); ax[0].set_ylabel("daily EPV ($)"); ax[0].legend()
ni.titlebox(ax[0], "Daily EPV vs CPV",
            f"Pearson r={pear.statistic:.2f}  |  Spearman ρ={spear.statistic:.2f}")

methods = ["Pearson\n(linear)", "Spearman\n(rank)", "Kendall\n(rank)"]
vals = [pear.statistic, spear.statistic, kend.statistic]
ax[1].bar(methods, vals, color=[ni.RED, ni.GREEN, ni.TEAL])
for i, v in enumerate(vals):
    ax[1].text(i, v+0.02, f"{v:.2f}", ha="center", fontweight="bold", color=ni.NAVY)
ax[1].set_ylabel("correlation coefficient"); ax[1].set_ylim(0, 1)
ni.titlebox(ax[1], "Same data, three coefficients", "rank methods resist the outlier pull")
fig.tight_layout(); ni.savefig(fig, "m3_three_correlations"); plt.show()
'''),
        md(r'''
### Outliers really do move Pearson

Remove just the top-5% whale days and watch **Pearson jump while Spearman barely
budges** — proof that the rank measure is telling you the stable story.
'''),
        code(r'''
keep = ~hi
p_all, p_trim = stats.pearsonr(x, y).statistic, stats.pearsonr(x[keep], y[keep]).statistic
s_all, s_trim = stats.spearmanr(x, y).statistic, stats.spearmanr(x[keep], y[keep]).statistic

fig, ax = plt.subplots(figsize=(9, 5))
xb = np.arange(2); w = 0.35
ax.bar(xb-w/2, [p_all, s_all], w, label="all days", color=ni.NAVY)
ax.bar(xb+w/2, [p_trim, s_trim], w, label="whale days removed", color=ni.SKY)
ax.set_xticks(xb); ax.set_xticklabels(["Pearson", "Spearman"]); ax.set_ylabel("coefficient"); ax.legend()
for i,(a,b) in enumerate([(p_all,p_trim),(s_all,s_trim)]):
    ax.text(i-w/2, a+0.01, f"{a:.2f}", ha="center", fontweight="bold")
    ax.text(i+w/2, b+0.01, f"{b:.2f}", ha="center", fontweight="bold")
ni.titlebox(ax, "Pearson is fragile, Spearman is stable",
            "removing 5% of days swings Pearson far more than Spearman")
fig.tight_layout(); ni.savefig(fig, "m3_pearson_fragile"); plt.show()
'''),
        md(r'''
## 3. The confound: it was never about the *day*

The weekend EPV drop is real in the raw data — but it's driven by **what kind of
traffic runs on weekends**, not the day itself. NI sends more high-payout finance
traffic on weekdays and more entertainment traffic on weekends. **Control for
vertical and the day effect collapses.**
'''),
        code(r'''
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))

# Left: weekday vertical MIX vs weekend — the real driver
mix = (visits.assign(part=np.where(visits.is_weekend,"Weekend","Weekday"))
       .pivot_table(index="part", columns="vertical", values="visit_id", aggfunc="size"))
mix = mix.div(mix.sum(axis=1), axis=0)[
    ["mortgages","loans","insurance","web_hosting","web_builders","vpn","streaming","dating"]]
bottom = np.zeros(len(mix))
for i, v in enumerate(mix.columns):
    ax[0].bar(mix.index, mix[v], bottom=bottom, label=v, color=ni.SEQ[i % len(ni.SEQ)])
    bottom += mix[v].values
ax[0].set_ylabel("share of traffic"); ax[0].legend(ncol=2, fontsize=8, loc="lower center")
ni.titlebox(ax[0], "WHY: the traffic mix differs by day",
            "weekdays carry more high-payout finance traffic")

# Right: EPV by weekend overall vs WITHIN one vertical (insurance)
parts = ["Weekday","Weekend"]
overall = [visits.loc[~visits.is_weekend,"revenue"].mean(), visits.loc[visits.is_weekend,"revenue"].mean()]
ins = visits[visits.vertical=="insurance"]
within = [ins.loc[~ins.is_weekend,"revenue"].mean(), ins.loc[ins.is_weekend,"revenue"].mean()]
xb = np.arange(2); w=0.36
ax[1].bar(xb-w/2, overall, w, label="overall EPV", color=ni.BLUE)
ax[1].bar(xb+w/2, within, w, label="within 'insurance' only", color=ni.GREEN)
ax[1].set_xticks(xb); ax[1].set_xticklabels(parts); ax[1].set_ylabel("EPV ($)"); ax[1].legend()
ni.titlebox(ax[1], "The 'day effect' disappears when mix is held fixed",
            f"overall drop {overall[1]/overall[0]-1:+.0%}  →  within-vertical {within[1]/within[0]-1:+.0%}")
fig.tight_layout(); ni.savefig(fig, "m3_confound"); plt.show()
'''),
        md(r'''
### ✅ Takeaway

> **On NI data, default to Spearman, and treat every correlation as a *question*.**
> "EPV depends on the day of week" was a **mix artifact** — the day didn't cause
> anything; the *vertical composition* of weekend traffic did. Before acting on
> any correlation, ask: *what third variable changes alongside both of these?*

**Next:** the Slack message claims Google ≠ Bing on conversion. Is that gap real,
or could it be luck? → *Module 4.*
'''),
    ]
    build("03_correlation.ipynb", cells)


# =========================================================================== #
# MODULE 4 — Hypothesis testing
# =========================================================================== #
def module4():
    cells = [
        md(r'''
# Module 4 — Is the Difference Real? Hypothesis Testing
### Practical Statistics for Analysts @ Natural Intelligence

Back to the Slack message: **Google 45% vs Bing 35% conversion.** Real, or noise?
Hypothesis testing is the discipline of asking *"if there were truly no difference,
how surprising is what I see?"* — the **p-value**. We'll also meet the traps that
make analysts over-claim: **sample size, the wrong test for skewed data, power,
and multiple comparisons.**
'''),
        code(BOOT),
        md(r'''
## 1. Two-proportion test: Google vs Bing conversion

- **H₀ (null):** Google and Bing convert at the same rate.
- **H₁:** they differ.

A tiny p-value means *"this gap would be very unlikely if H₀ were true."*
'''),
        code(r'''
g = visits[visits.engine.isin(["Google","Bing"])].groupby("engine", observed=True)["converted"].agg(["sum","size"])
kG, nG = g.loc["Google"]; kB, nB = g.loc["Bing"]
res = ni.two_proportion_ztest(int(kG), int(nG), int(kB), int(nB))
print(f"Google: {res['p1']:.1%}  (n={nG:,})")
print(f"Bing  : {res['p2']:.1%}  (n={nB:,})")
print(f"difference = {res['diff']:+.1%}   z = {res['z']:.1f}   p-value = {res['p_value']:.2e}")

fig, ax = plt.subplots(figsize=(8, 5))
for i, eng in enumerate(["Google","Bing"]):
    k, n = (kG,nG) if eng=="Google" else (kB,nB)
    p, lo, hi = ni.wilson_ci(int(k), int(n))
    ax.bar(i, p, 0.55, color=ni.ENGINE_COLORS[eng])
    ax.errorbar(i, p, yerr=[[p-lo],[hi-p]], color=ni.NAVY, capsize=8, lw=2)
    ax.text(i, hi+0.012, f"{p:.1%}", ha="center", fontweight="bold", color=ni.NAVY)
ax.set_xticks([0,1]); ax.set_xticklabels(["Google","Bing"]); ax.set_ylim(0, 0.55)
ax.set_ylabel("conversion rate (95% Wilson CI)")
ni.titlebox(ax, "Google vs Bing conversion", f"gap = {res['diff']:+.1%},  p = {res['p_value']:.1e}  → not luck")
fig.tight_layout(); ni.savefig(fig, "m4_google_vs_bing"); plt.show()
'''),
        md(r'''
## 2. The #1 trap: significance depends on sample size

The **same 10-point gap** is "insignificant" at small n and "rock-solid" at large n.
A p-value is not an effect size — it's a statement about *evidence*, which grows with data.
Left: p-value shrinks as we feed in more visits. Right: an **A/A test** (no real
difference) produces p-values scattered everywhere — most "findings" at small n are noise.
'''),
        code(r'''
rng = np.random.default_rng(0)
G = visits[visits.engine=="Google"]["converted"].values
B = visits[visits.engine=="Bing"]["converted"].values

sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
pvals = []
for nps in sizes:
    gk = rng.choice(G, nps); bk = rng.choice(B, nps)
    pvals.append(ni.two_proportion_ztest(gk.sum(), nps, bk.sum(), nps)["p_value"])

# A/A null: split Google against itself, repeat
aa = []
for _ in range(400):
    idx = rng.permutation(len(G)); half = len(G)//2
    a, b = G[idx[:half]], G[idx[half:2*half]]
    aa.append(ni.two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))["p_value"])

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(sizes, pvals, "-o", color=ni.BLUE, lw=2)
ax[0].axhline(0.05, color=ni.RED, ls="--", lw=1.6, label="α = 0.05")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("visits per engine (log)"); ax[0].set_ylabel("p-value (log)"); ax[0].legend()
ni.titlebox(ax[0], "Same effect, growing evidence", "the gap only 'becomes significant' once n is big enough")

ax[1].hist(aa, bins=20, color=ni.GREY, alpha=0.85, edgecolor="white")
ax[1].axvline(0.05, color=ni.RED, ls="--", lw=1.6)
fp = np.mean(np.array(aa) < 0.05)
ax[1].set_xlabel("p-value"); ax[1].set_ylabel("# of A/A tests")
ni.titlebox(ax[1], "A/A test: no real difference", f"{fp:.0%} still land p<0.05 by chance (that's α!)")
fig.tight_layout(); ni.savefig(fig, "m4_sample_size"); plt.show()
'''),
        md(r'''
## 3. The right test for *skewed* EPV: t-test vs Mann-Whitney

For mobile-vs-desktop **EPV** the outcome is heavy-tailed, so a plain t-test on the
mean is shaky. Compare it with the **Mann-Whitney** rank test and a **bootstrap** —
when they agree, you can trust the call; when they disagree, the whales are talking.
'''),
        code(r'''
mob = visits[visits.device=="mobile"]["revenue"].values
desk = visits[visits.device=="desktop"]["revenue"].values

t = stats.ttest_ind(mob, desk, equal_var=False)
u = stats.mannwhitneyu(mob, desk, alternative="two-sided")
point, lo, hi, _ = ni.bootstrap_ratio_ci(mob, desk, n_boot=3000, seed=3)
print(f"Welch t-test     : t={t.statistic:.1f},  p={t.pvalue:.2e}")
print(f"Mann-Whitney U   : p={u.pvalue:.2e}   (rank-based, robust to whales)")
print(f"Bootstrap EPV gap: {point:+.1f}%   95% CI [{lo:+.1f}%, {hi:+.1f}%]")
print("→ All three agree mobile EPV is genuinely lower. Confidence is high.")
'''),
        md(r'''
## 4. Chi-square: are engine and device *associated*?

When both variables are categorical (engine × device, outcome = converted),
the **chi-square test** asks whether the pattern of conversions is independent of
the combination — useful before you trust a per-cell number.
'''),
        code(r'''
ct = pd.crosstab(visits.engine, visits.device, values=visits.converted, aggfunc="sum")
nt = pd.crosstab(visits.engine, visits.device)
chi2, p, dof, _ = stats.chi2_contingency(ct.values)
print("Conversions by engine × device:"); display(ct)
print(f"chi-square = {chi2:.1f},  dof = {dof},  p = {p:.2e}")
print("→ Conversion behaviour is NOT independent of the engine/device combo.")
'''),
        md(r'''
## 5. Power & errors — could we *miss* a real effect?

- **Type I error (α):** crying wolf — a difference that isn't there.
- **Type II error (β):** missing a real difference. **Power = 1 − β.**

The curve shows how many visits per engine you need to reliably detect a gap.
A big gap (10 pts) needs little data; a subtle 2-pt gap needs a lot.
'''),
        code(r'''
ns = np.arange(50, 6000, 50)
pow_big = [ni.power_two_proportions(0.45, 0.35, n) for n in ns]   # the real gap
pow_small = [ni.power_two_proportions(0.45, 0.43, n) for n in ns]  # a subtle gap

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(ns, pow_big, color=ni.GREEN, lw=2.4, label="10-pt gap (45% vs 35%)")
ax.plot(ns, pow_small, color=ni.ORANGE, lw=2.4, label="2-pt gap (45% vs 43%)")
ax.axhline(0.8, color=ni.RED, ls="--", lw=1.6, label="80% power target")
ax.set_xlabel("visits per engine"); ax.set_ylabel("power (chance of detecting the gap)")
ax.legend()
ni.titlebox(ax, "How much data do you need?",
            "subtle effects need far bigger samples — plan before you peek")
fig.tight_layout(); ni.savefig(fig, "m4_power_curve"); plt.show()
'''),
        md(r'''
## 6. Multiple comparisons: slice enough and you'll "find" anything

NI analysts slice dozens of verticals × engines × devices daily. If you run 40
*genuinely null* tests at α=0.05, you **expect ~2 false positives**. Here we run
many A/A tests across pretend "segments" — and watch false alarms appear. The
**Bonferroni** line shows the corrected bar.
'''),
        code(r'''
rng = np.random.default_rng(11)
m = 40
ps = []
base = visits["converted"].values
for _ in range(m):
    idx = rng.permutation(len(base)); h = 4000
    a, b = base[idx[:h]], base[idx[h:2*h]]
    ps.append(ni.two_proportion_ztest(a.sum(), h, b.sum(), h)["p_value"])
ps = np.array(ps)
false_pos = (ps < 0.05).sum()
bonf = 0.05 / m

fig, ax = plt.subplots(figsize=(12, 5))
colors = [ni.RED if p < 0.05 else ni.SKY for p in ps]
ax.bar(range(m), ps, color=colors)
ax.axhline(0.05, color=ni.NAVY, ls="--", lw=1.6, label="α = 0.05 (naive)")
ax.axhline(bonf, color=ni.GREEN, ls="--", lw=1.6, label=f"Bonferroni = {bonf:.4f}")
ax.set_xlabel("'segment' # (all genuinely NULL)"); ax.set_ylabel("p-value"); ax.legend()
ni.titlebox(ax, f"{m} null tests → {false_pos} false 'discoveries' at α=0.05",
            "red bars cross the naive line by pure chance; Bonferroni guards against it")
fig.tight_layout(); ni.savefig(fig, "m4_multiple_comparisons"); plt.show()
'''),
        md(r'''
### ✅ Takeaway

> **Significant ≠ important, and significant ≠ true.**
> Check three things every time: (1) is n big enough that the *gap*, not the *size*,
> is driving the p-value? (2) is the test right for the data shape (rank/bootstrap
> for EPV)? (3) how many tests did I run — do I need a correction? The Google>Bing
> conversion gap survives all three. **It's real.**

**Next:** "real" isn't enough to move budget. *How big* is the effect, and *how
sure* are we? → *Module 5.*
'''),
    ]
    build("04_hypothesis_testing.ipynb", cells)


# =========================================================================== #
# MODULE 5 — Confidence intervals
# =========================================================================== #
def module5():
    cells = [
        md(r'''
# Module 5 — How Big and How Sure? Confidence Intervals
### Practical Statistics for Analysts @ Natural Intelligence

A p-value says *"something is there."* A **confidence interval** says *"here's how
big it is, and here's our uncertainty"* — which is what a stakeholder actually needs
to make a budget call. This module turns the Slack question's **mobile EPV CI of
[-20%, -5%]** into something you can compute, plot, and explain.
'''),
        code(BOOT),
        md(r'''
## 1. CIs for proportions — a forest plot of conversion rates

The **Wilson interval** is the reliable choice for conversion rates. Plotting each
engine with its CI ("forest plot") makes overlaps — and real separations — obvious
at a glance. Non-overlapping intervals ≈ a real difference.
'''),
        code(r'''
eng = visits.groupby("engine", observed=True)["converted"].agg(["sum","size"]).reindex(["Google","Organic","Bing","Social"])
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (name, row) in enumerate(eng.iterrows()):
    p, lo, hi = ni.wilson_ci(int(row["sum"]), int(row["size"]))
    ax.plot([lo, hi], [i, i], color=ni.ENGINE_COLORS[name], lw=3)
    ax.plot(p, i, "o", color=ni.ENGINE_COLORS[name], ms=11)
    ax.text(hi+0.004, i, f"{p:.1%}  [{lo:.1%}, {hi:.1%}]", va="center", color=ni.NAVY, fontsize=10)
ax.set_yticks(range(len(eng))); ax.set_yticklabels(eng.index)
ax.set_xlabel("conversion rate (95% Wilson CI)"); ax.set_xlim(0.25, 0.52)
ax.invert_yaxis()
ni.titlebox(ax, "Conversion rate by engine — with uncertainty",
            "tight intervals (huge n) → these separations are real")
fig.tight_layout(); ni.savefig(fig, "m5_forest_proportions"); plt.show()
'''),
        md(r'''
## 2. The headline: a bootstrap CI for the mobile/desktop **EPV ratio**

EPV is a *ratio* (revenue ÷ visits) of skewed data, so we don't trust a textbook
formula — we **bootstrap**: resample the data thousands of times and watch how the
ratio wobbles. The middle 95% of those wobbles is our CI.

Realistically an analyst looks at **a couple of weeks** of one comparison — which
is exactly where the email's **[-20%, -5%]** interval comes from.
'''),
        code(r'''
two_wk = visits[(visits.date >= "2026-05-01") & (visits.date <= "2026-05-14") &
                visits.device.isin(["mobile","desktop"])]
mob = two_wk[two_wk.device=="mobile"]["revenue"].values
desk = two_wk[two_wk.device=="desktop"]["revenue"].values
point, lo, hi, boots = ni.bootstrap_ratio_ci(mob, desk, n_boot=6000, seed=5)

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(boots, bins=50, color=ni.LIGHT, edgecolor=ni.SKY)
ax.axvspan(lo, hi, color=ni.BLUE, alpha=0.12)
for v, c, lab in [(point, ni.NAVY, f"point {point:+.0f}%"), (lo, ni.BLUE, f"{lo:+.0f}%"), (hi, ni.BLUE, f"{hi:+.0f}%")]:
    ax.axvline(v, color=c, lw=2.2, ls="--" if c==ni.BLUE else "-")
    ax.text(v, ax.get_ylim()[1]*0.93, lab, rotation=90, va="top", ha="right", color=c, fontweight="bold")
ax.axvline(0, color=ni.RED, lw=1.4)
ax.set_xlabel("mobile EPV vs desktop (%)"); ax.set_ylabel("# bootstrap resamples")
ni.titlebox(ax, f"Mobile EPV is {point:.0f}% lower — 95% CI [{lo:.0f}%, {hi:.0f}%]",
            "the whole interval sits below 0 → we're confident mobile underperforms")
fig.tight_layout(); ni.savefig(fig, "m5_bootstrap_ratio"); plt.show()
print(f"n(mobile)={len(mob):,}  n(desktop)={len(desk):,}")
'''),
        md(r'''
## 3. Why the interval's *width* matters: n drives certainty

The same true gap looks wildly different depending on how much data you pool.
One day = a uselessly wide interval (whales!); the full quarter = a razor-thin one.
**Always report the interval, not just the point — and know what n is behind it.**
'''),
        code(r'''
windows = {
    "1 day":   ("2026-05-07", "2026-05-07"),
    "1 week":  ("2026-05-04", "2026-05-10"),
    "2 weeks": ("2026-05-01", "2026-05-14"),
    "full quarter": ("2026-03-01", "2026-05-30"),
}
rows = []
for label, (a, b) in windows.items():
    s = visits[(visits.date>=a) & (visits.date<=b) & visits.device.isin(["mobile","desktop"])]
    mo = s[s.device=="mobile"]["revenue"].values; de = s[s.device=="desktop"]["revenue"].values
    pt, l, h, _ = ni.bootstrap_ratio_ci(mo, de, n_boot=3000, seed=9)
    rows.append((label, len(s), pt, l, h))

fig, ax = plt.subplots(figsize=(11, 4.8))
for i, (label, n, pt, l, h) in enumerate(rows):
    ax.plot([l, h], [i, i], color=ni.BLUE, lw=3)
    ax.plot(pt, i, "o", color=ni.NAVY, ms=10)
    ax.text(h+0.6, i, f"n={n:,}", va="center", fontsize=10, color=ni.GREY)
ax.axvline(0, color=ni.RED, lw=1.4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("mobile EPV vs desktop (%, 95% CI)")
ni.titlebox(ax, "More data → tighter interval → firmer decision",
            "one day is hopeless; two weeks already nails the email's [-20%, -5%]")
fig.tight_layout(); ni.savefig(fig, "m5_ci_width_vs_n"); plt.show()
'''),
        md(r'''
## 4. Say it three ways — for three audiences

Same interval, framed for whoever's in the room. **Lead with the interval, not the p-value.**
'''),
        code(r'''
print("FROM THE [-20%, -5%]-STYLE RESULT, COMMUNICATE:\n")
print("👔  Executive:   'Mobile visitors are worth meaningfully less right now — roughly")
print("                 5 to 20 percent below desktop. We're confident it's a real gap,")
print("                 so it's safe to rebalance bids toward desktop.'\n")
print("📊  Analyst:     'Mobile/desktop EPV ratio is -12% (95% bootstrap CI [-20%, -5%], n≈26k).")
print("                 Interval excludes 0 → significant at α=0.05; whale-robust.'\n")
print("🛠️  Action:      'Cut mobile bids ~10-15% on the affected verticals; re-measure in 2 weeks.")
print("                 Don't act on any single day — that interval is too wide to trust.'")
'''),
        md(r'''
### ✅ Takeaway

> **Lead with the interval.** "Mobile EPV is **5–20% lower** (and we're 95% sure it's
> *down*)" drives a decision; "p<0.05" does not. The CI carries **direction +
> magnitude + uncertainty** in one breath — and its *width* tells you whether you
> have enough data to act.

**Next:** put every tool together and finally answer the media buyer. → *Module 6.*
'''),
    ]
    build("05_confidence_intervals.ipynb", cells)


# =========================================================================== #
# MODULE 6 — Integrated case
# =========================================================================== #
def module6():
    cells = [
        md(r'''
# Module 6 — Putting It Together: The Budget Decision
### Practical Statistics for Analysts @ Natural Intelligence

> **💬 Back to the Slack message:** *"Google converts better than Bing, mobile EPV is
> down, weekends are crashing — should we pull budget out of Bing and the weekend?"*

Now we answer it properly, end to end: pick the right metric, respect the
distribution, test the differences, quantify with intervals, and convert it all
into a **recommendation with a decision rule and a projected impact.**

The real objective isn't conversion rate *or* EPV — it's **profit per visit =
EPV − CPV**. That's what we optimize.
'''),
        code(BOOT),
        md(r'''
## 1. The full funnel, by engine × device — with uncertainty

One table that an analyst can actually take to a meeting: volume, conversion (with
CI), EPV (with bootstrap CI), cost, **profit per visit (with CI)**, and ROAS.
'''),
        code(r'''
def segment_stats(df):
    n = len(df); k = int(df.converted.sum())
    cr, cr_lo, cr_hi = ni.wilson_ci(k, n)
    epv, epv_lo, epv_hi = ni.bootstrap_mean_ci(df.revenue.values, n_boot=1500, seed=1)
    cpv = df.cost.mean()
    ppv, ppv_lo, ppv_hi = ni.bootstrap_mean_ci(df.profit.values, n_boot=1500, seed=2)
    return pd.Series({
        "visits": n, "conv_rate": cr, "conv_lo": cr_lo, "conv_hi": cr_hi,
        "EPV": epv, "CPV": cpv, "profit_visit": ppv, "ppv_lo": ppv_lo, "ppv_hi": ppv_hi,
        "ROAS": df.revenue.sum()/df.cost.sum()})

seg = (visits.groupby(["engine","device"], observed=True)
       .apply(segment_stats, include_groups=False).reset_index())
seg = seg[seg.visits > 1500].sort_values("profit_visit", ascending=False)

show = seg.assign(
    conversion=lambda d: d.apply(lambda r: f"{r.conv_rate:.0%} [{r.conv_lo:.0%},{r.conv_hi:.0%}]", axis=1),
    EPV_=lambda d: d.EPV.map("${:.2f}".format),
    CPV_=lambda d: d.CPV.map("${:.2f}".format),
    profit=lambda d: d.apply(lambda r: f"${r.profit_visit:.2f} [${r.ppv_lo:.2f},${r.ppv_hi:.2f}]", axis=1),
    ROAS_=lambda d: d.ROAS.map("{:.2f}x".format),
)[["engine","device","visits","conversion","EPV_","CPV_","profit","ROAS_"]]
show.columns = ["engine","device","visits","conversion (95% CI)","EPV","CPV","profit/visit (95% CI)","ROAS"]
show.style.hide(axis="index").set_caption("NI funnel by engine × device (last quarter)")
'''),
        md(r'''
## 2. Where is the profit? A forest plot of profit-per-visit

Green = profitable to scale, red = losing money per visit. The CIs tell us which
calls are safe (interval clear of \$0) vs which need more data.
'''),
        code(r'''
seg2 = seg.copy()
seg2["label"] = seg2.engine + " · " + seg2.device
seg2 = seg2.sort_values("profit_visit")
fig, ax = plt.subplots(figsize=(11, 6.5))
for i, (_, r) in enumerate(seg2.iterrows()):
    col = ni.GREEN if r.profit_visit > 0 else ni.RED
    ax.plot([r.ppv_lo, r.ppv_hi], [i, i], color=col, lw=3)
    ax.plot(r.profit_visit, i, "o", color=col, ms=9)
ax.axvline(0, color=ni.NAVY, lw=1.6)
ax.set_yticks(range(len(seg2))); ax.set_yticklabels(seg2.label)
ax.set_xlabel("profit per visit ($, 95% CI)")
ni.titlebox(ax, "Profit per visit by segment — what to scale, what to cut",
            "intervals clear of $0 are safe bets; ones straddling $0 need more data")
fig.tight_layout(); ni.savefig(fig, "m6_profit_forest"); plt.show()
'''),
        md(r'''
## 3. A decision rule, and the projected impact

Every segment here is *profitable* — so the decision isn't "cut losers," it's
**"where should the next marginal dollar go?"** We bid only on **paid** engines
(Organic is free SEO — protect it, but it's not a budget lever), so we rank the
paid segments by **profit per visit** and reallocate.

**Rule:** *Scale up* the top third of paid segments; *trim bids* on the bottom third
(their profit/visit CI sits clearly below the leaders — Module 5). Then simulate
shifting 20% of the weakest segments' spend into the strongest.
'''),
        code(r'''
# We control bids on PAID engines only (Organic is free SEO, not a budget lever)
paid = seg[seg.engine != "Organic"].copy().sort_values("profit_visit")
lo_thr, hi_thr = paid.profit_visit.quantile([1/3, 2/3])
paid["verdict"] = np.where(paid.profit_visit >= hi_thr, "SCALE UP",
                   np.where(paid.profit_visit <= lo_thr, "TRIM", "hold / optimize"))
print("Budget decision for PAID segments (we don't bid on Organic):\n")
print(paid[["engine","device","visits","profit_visit","ppv_lo","ppv_hi","verdict"]]
      .round(2).sort_values("profit_visit", ascending=False).to_string(index=False))

# Reallocate 20% of the weakest ('TRIM') segments' traffic into the strongest segment
winners = paid[paid.verdict == "SCALE UP"]
losers  = paid[paid.verdict == "TRIM"]
top = winners.iloc[winners.profit_visit.values.argmax()]
moved_visits = int(0.20 * losers.visits.sum())
l_ppv = (losers.profit_visit * losers.visits).sum() / losers.visits.sum()
delta = moved_visits * (top.profit_visit - l_ppv)
print(f"\nMove ~{moved_visits:,} visits/qtr from TRIM segments (avg ${l_ppv:.2f}/visit) "
      f"to {top.engine}·{top.device} (${top.profit_visit:.2f}/visit)")

current = (seg.profit_visit * seg.visits).sum()
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.bar(["current\n(quarter)","after 20%\nreallocation"], [current, current+delta],
       color=[ni.GREY, ni.GREEN])
for i, v in enumerate([current, current+delta]):
    ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontweight="bold", color=ni.NAVY)
ax.set_ylim(0, (current+delta)*1.12); ax.set_ylabel("modeled total profit ($ / quarter)")
ni.titlebox(ax, "Projected impact of the reallocation",
            f"shift 20% of the weakest paid traffic to the top segment → +${delta:,.0f} / quarter")
fig.tight_layout(); ni.savefig(fig, "m6_reallocation"); plt.show()
'''),
        md(r'''
## 4. The answer to the Slack message

> **"Google converts better than Bing — should we pull budget out of Bing?"**
>
> ⚠️ **Don't kill Bing — conversion rate is the wrong yardstick.** Yes, Google's
> conversion edge is *real* (Module 4), **but budget follows profit per visit, not
> conversion rate.** Bing is still solidly profitable (positive profit/visit, CI
> clear of \$0 — Module 5), and its cheaper clicks partly offset the lower
> conversion. The right move is to **shift *marginal* budget toward the highest
> profit/visit segments (Google desktop leads), trim the weakest, and keep Bing for
> profitable volume** — not a blanket cut.
>
> ✅ **On mobile EPV:** that one *is* a real, quantified problem — mobile EPV is
> **5–20% below desktop** (Module 5). Trim mobile bids on the affected verticals
> and re-measure in two weeks.
>
> 🚫 **Ignore the "weekend EPV crash":** it's a **traffic-mix artifact** (Module 3) —
> weekends carry more low-payout entertainment traffic, not a real day effect.
> Don't touch weekend bidding because of it.
>
> 💡 **Bonus insight the data hands us:** *Organic* traffic earns ~8× ROAS (near-zero
> cost) — by far the most profitable. The biggest lever isn't engine reallocation at
> all; it's **protecting and growing SEO.**
'''),
        md(r'''
## 🧾 The Statistical Decision Checklist (pin this at your desk)

| Step | Ask yourself |
|---|---|
| **1. Frame** | What's the unit, the outcome *type*, and what am I comparing? (Module 1) |
| **2. Look** | Did I plot the distribution? Is it skewed? Are there whales? (Module 2) |
| **3. Relate** | Is this correlation rank-based and confound-checked? (Module 3) |
| **4. Test** | Right test for the data shape? Enough power? How many tests did I run? (Module 4) |
| **5. Quantify** | What's the confidence interval — direction, magnitude, width? (Module 5) |
| **6. Decide** | Does it change **profit per visit**? What's the rule and the projected impact? (Module 6) |

> **The one-liner:** *Significance tells you something is there; the confidence
> interval tells you whether it's worth acting on; profit-per-visit tells you what to do.*

### 🎓 That's the workshop — from a Slack message to a defensible decision.
'''),
    ]
    build("06_integrated_case.ipynb", cells)


if __name__ == "__main__":
    module1(); module2(); module3(); module4(); module5(); module6()
    print("\nAll six notebooks written to", NBDIR)
