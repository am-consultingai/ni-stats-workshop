"""
Natural Intelligence — Workshop mock data generator
===================================================

Generates a SYNTHETIC visit-level dataset that mimics the data an analyst at
Natural Intelligence (Top10.com / BestMoney.com style comparison sites) works
with every day, plus aggregated views built on top of it.

Business model in one line:
    Buy a visit (paid search / SEO / social) -> user clicks out to a brand
    ("conversion") -> advertiser pays a commission ("revenue").
    EPV = Earnings Per Visit = revenue / visits.  Profit/visit = EPV - cost/visit.

The data is deliberately engineered so every workshop demo works out of the box.
PLANTED EFFECTS (the teaching hooks):

  1. ENGINE -> CONVERSION RATE   Google click-out rate ~45% vs Bing ~35%.
                                 A REAL effect (Module 4: two-proportion test).
  2. DEVICE -> EPV               Mobile EPV ~12% below desktop. A REAL effect of
                                 a size that yields a CI near [-20%, -5%]
                                 (Module 5: confidence interval for a ratio).
  3. DAY-OF-WEEK <-> EPV         Weekday EPV looks higher than weekend EPV, BUT it
                                 is a CONFOUND: weekdays carry more high-payout
                                 finance traffic. Controlling for vertical, the
                                 day effect vanishes (Module 3: correlation != cause).
  4. SKEW + WHALES              Revenue is zero for most visits and heavy-tailed
                                 (log-normal) for converters, with rare "whale"
                                 conversions. The mean EPV is unstable
                                 (Module 2: outliers, skewness, transformations).
  5. COST                       Per-visit acquisition cost, so profit/ROI is
                                 computable for the final decision (Module 6).

Run:  python src/generate_data.py
Out:  data/visits.csv                 (visit-level, the raw grain)
      data/agg_daily_engine_device.csv
      data/agg_by_engine.csv
      data/agg_by_device.csv
      data/agg_vertical_by_dow.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reproducibility & scale
# --------------------------------------------------------------------------- #
SEED = 42
RNG = np.random.default_rng(SEED)
N_VISITS = 180_000
START_DATE = pd.Timestamp("2026-03-01")   # 13 full weeks ending late May 2026
N_DAYS = 91

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Dimension configuration
# --------------------------------------------------------------------------- #
ENGINES = ["Google", "Bing", "Organic", "Social"]
ENGINE_SHARE = [0.50, 0.18, 0.22, 0.10]
# Click-out (conversion) base rate by engine -> drives the 45% vs 35% story.
ENGINE_CLICKOUT_BASE = {"Google": 0.45, "Bing": 0.35, "Organic": 0.42, "Social": 0.30}
# Acquisition cost multiplier (Google auctions pricey, organic ~free).
ENGINE_COST_FACTOR = {"Google": 1.20, "Bing": 0.90, "Organic": 0.20, "Social": 0.70}

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_SHARE = [0.58, 0.34, 0.08]
# Device is (almost) neutral on click-out; the device effect lives in REVENUE,
# so the EPV gap is clean. mobile revenue factor 0.88 -> ~ -12% EPV vs desktop.
DEVICE_CLICKOUT_ADJ = {"mobile": -0.01, "desktop": 0.00, "tablet": -0.005}
DEVICE_REV_FACTOR = {"mobile": 0.88, "desktop": 1.00, "tablet": 0.93}

# Verticals: (base traffic share, mean payout $ per click-out, log-normal sigma)
# High-payout finance verticals are rare but lucrative -> heavy tail.
VERTICALS = {
    "mortgages":    dict(share=0.06, payout=65.0, sigma=0.95),
    "loans":        dict(share=0.07, payout=55.0, sigma=0.95),
    "insurance":    dict(share=0.10, payout=45.0, sigma=0.90),
    "web_hosting":  dict(share=0.13, payout=12.0, sigma=0.80),
    "web_builders": dict(share=0.15, payout=9.0,  sigma=0.80),
    "vpn":          dict(share=0.16, payout=6.0,  sigma=0.75),
    "streaming":    dict(share=0.13, payout=4.0,  sigma=0.70),
    "dating":       dict(share=0.20, payout=4.0,  sigma=0.70),
}
VERT_NAMES = list(VERTICALS)
VERT_BASE_SHARE = np.array([VERTICALS[v]["share"] for v in VERT_NAMES])

# Day-of-week CONFOUND: tilt the vertical mix.
# Weekdays -> more finance (high payout); weekends -> more entertainment (low payout).
HIGH_PAYOUT = {"mortgages", "loans", "insurance"}
LOW_PAYOUT = {"streaming", "dating", "vpn"}
weekday_tilt = np.array([1.6 if v in HIGH_PAYOUT else (0.8 if v in LOW_PAYOUT else 1.0)
                         for v in VERT_NAMES])
weekend_tilt = np.array([0.5 if v in HIGH_PAYOUT else (1.4 if v in LOW_PAYOUT else 1.0)
                         for v in VERT_NAMES])
PROB_WEEKDAY = (VERT_BASE_SHARE * weekday_tilt); PROB_WEEKDAY /= PROB_WEEKDAY.sum()
PROB_WEEKEND = (VERT_BASE_SHARE * weekend_tilt); PROB_WEEKEND /= PROB_WEEKEND.sum()

# Whale conversions: rare, enormous payouts (the outliers that break the mean).
WHALE_PROB = 0.004
WHALE_MULT_LOW, WHALE_MULT_HIGH = 8.0, 25.0


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_visits() -> pd.DataFrame:
    n = N_VISITS

    # --- when ---
    day_offset = RNG.integers(0, N_DAYS, n)
    date = START_DATE + pd.to_timedelta(day_offset, unit="D")
    dow = date.dayofweek.values                     # 0=Mon ... 6=Sun
    is_weekend = dow >= 5
    dow_name = pd.Categorical.from_codes(
        dow, categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    # --- who / where ---
    engine = RNG.choice(ENGINES, n, p=ENGINE_SHARE)
    device = RNG.choice(DEVICES, n, p=DEVICE_SHARE)

    # --- vertical (day-conditioned -> the confound) ---
    vert_idx = np.empty(n, dtype=int)
    wd_mask = ~is_weekend
    vert_idx[wd_mask] = RNG.choice(len(VERT_NAMES), wd_mask.sum(), p=PROB_WEEKDAY)
    vert_idx[~wd_mask] = RNG.choice(len(VERT_NAMES), (~wd_mask).sum(), p=PROB_WEEKEND)
    vertical = np.array(VERT_NAMES)[vert_idx]

    # --- conversion (click-out to a brand) ---
    p_click = (np.vectorize(ENGINE_CLICKOUT_BASE.get)(engine)
               + np.vectorize(DEVICE_CLICKOUT_ADJ.get)(device)
               + RNG.normal(0.0, 0.02, n))
    p_click = np.clip(p_click, 0.02, 0.95)
    clicked = RNG.random(n) < p_click

    # --- revenue (heavy-tailed; zero unless click-out) ---
    payout_mu = np.array([np.log(VERTICALS[v]["payout"]) - 0.5 * VERTICALS[v]["sigma"] ** 2
                          for v in VERT_NAMES])[vert_idx]
    payout_sigma = np.array([VERTICALS[v]["sigma"] for v in VERT_NAMES])[vert_idx]
    base_rev = np.exp(RNG.normal(payout_mu, payout_sigma))
    base_rev *= np.vectorize(DEVICE_REV_FACTOR.get)(device)

    whale = clicked & (RNG.random(n) < WHALE_PROB)
    base_rev[whale] *= RNG.uniform(WHALE_MULT_LOW, WHALE_MULT_HIGH, whale.sum())

    revenue = np.where(clicked, base_rev, 0.0)

    # --- acquisition cost (paid on every visit) ---
    vert_payout = np.array([VERTICALS[v]["payout"] for v in VERT_NAMES])[vert_idx]
    cost = (0.22 * vert_payout
            * np.vectorize(ENGINE_COST_FACTOR.get)(engine)
            * RNG.lognormal(0.0, 0.35, n))
    cost = np.round(cost, 4)

    df = pd.DataFrame({
        "date": date,
        "day_of_week": dow_name,
        "is_weekend": is_weekend,
        "engine": engine,
        "device": device,
        "vertical": vertical,
        "converted": clicked.astype(int),     # 1 = clicked out to a brand
        "revenue": np.round(revenue, 4),       # commission earned on this visit
        "cost": cost,                          # what we paid to acquire this visit
    })
    df["profit"] = (df["revenue"] - df["cost"]).round(4)
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "visit_id", np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------- #
# Aggregated views (how an analyst would query the warehouse)
# --------------------------------------------------------------------------- #
def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, observed=True).agg(
        visits=("visit_id", "size"),
        conversions=("converted", "sum"),
        revenue=("revenue", "sum"),
        cost=("cost", "sum"),
    ).reset_index()
    g["conversion_rate"] = (g["conversions"] / g["visits"]).round(4)
    g["EPV"] = (g["revenue"] / g["visits"]).round(4)              # earnings per visit
    g["CPV"] = (g["cost"] / g["visits"]).round(4)                 # cost per visit
    g["profit_per_visit"] = (g["EPV"] - g["CPV"]).round(4)
    g["roas"] = (g["revenue"] / g["cost"]).round(3)               # return on ad spend
    return g


def build_aggregates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "agg_daily_engine_device": _agg(df, ["date", "engine", "device"]),
        "agg_by_engine":           _agg(df, ["engine"]),
        "agg_by_device":           _agg(df, ["device"]),
        "agg_vertical_by_dow":     _agg(df, ["day_of_week", "vertical"]),
    }


# --------------------------------------------------------------------------- #
# Verification — prove the planted effects are present
# --------------------------------------------------------------------------- #
def verify(df: pd.DataFrame) -> None:
    from scipy import stats

    print("\n" + "=" * 68)
    print(f"GENERATED {len(df):,} visits  |  {df['date'].min().date()} -> {df['date'].max().date()}")
    print("=" * 68)

    print("\n[1] ENGINE -> CONVERSION RATE (target Google ~0.45, Bing ~0.35)")
    cr = df.groupby("engine", observed=True)["converted"].mean().round(4)
    print(cr.to_string())

    print("\n[2] DEVICE -> EPV  (target mobile ~ -12% vs desktop)")
    epv = df.groupby("device", observed=True)["revenue"].mean()
    d, m = epv["desktop"], epv["mobile"]
    # bootstrap the mobile/desktop EPV ratio CI on a teaching-sized sample
    sub = df[df["device"].isin(["mobile", "desktop"])]
    rev = sub["revenue"].values
    is_mob = (sub["device"] == "mobile").values
    boot = []
    for _ in range(2000):
        idx = RNG.integers(0, len(sub), len(sub))
        rr, mm = rev[idx], is_mob[idx]
        boot.append(rr[mm].mean() / rr[~mm].mean() - 1.0)
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
    print(f"    EPV desktop=${d:.3f}  mobile=${m:.3f}  point diff={ (m/d-1)*100:+.1f}%")
    print(f"    95% bootstrap CI on (mobile/desktop - 1):  [{lo:+.1f}%, {hi:+.1f}%]")

    print("\n[3] DAY-OF-WEEK <-> EPV  (raw effect that is a vertical-mix CONFOUND)")
    daily = df.groupby("date", observed=True).agg(
        EPV=("revenue", "mean")).reset_index()
    daily["dow"] = daily["date"].dt.dayofweek
    rho_raw, p_raw = stats.spearmanr(daily["dow"], daily["EPV"])
    wk = df[~df["is_weekend"]]["revenue"].mean()
    we = df[df["is_weekend"]]["revenue"].mean()
    print(f"    Raw weekday EPV=${wk:.3f}  weekend EPV=${we:.3f}  ({(we/wk-1)*100:+.1f}%)")
    # within a single vertical the weekday/weekend gap should ~vanish
    ins = df[df["vertical"] == "insurance"]
    wk_i = ins[~ins["is_weekend"]]["revenue"].mean()
    we_i = ins[ins["is_weekend"]]["revenue"].mean()
    print(f"    Within 'insurance' only: weekday=${wk_i:.3f} weekend=${we_i:.3f} "
          f"({(we_i/wk_i-1)*100:+.1f}%)  <- confound controlled")

    print("\n[4] SKEW / WHALES  (mean EPV is unstable; median tells another story)")
    conv = df[df["converted"] == 1]["revenue"]
    print(f"    revenue|converted: mean=${conv.mean():.2f}  median=${conv.median():.2f}  "
          f"max=${conv.max():.0f}  skew={stats.skew(conv):.2f}")
    print(f"    share of revenue from top 1% of converting visits: "
          f"{conv[conv >= conv.quantile(0.99)].sum() / conv.sum() * 100:.1f}%")

    print("\n[5] UNIT ECONOMICS (overall)")
    print(f"    EPV=${df['revenue'].mean():.3f}  CPV=${df['cost'].mean():.3f}  "
          f"profit/visit=${df['profit'].mean():.3f}  ROAS={df['revenue'].sum()/df['cost'].sum():.2f}")
    print("=" * 68 + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    df = generate_visits()
    df.to_csv(DATA_DIR / "visits.csv", index=False)
    print(f"wrote {DATA_DIR/'visits.csv'}  ({len(df):,} rows)")

    for name, adf in build_aggregates(df).items():
        adf.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"wrote {DATA_DIR / (name + '.csv')}  ({len(adf):,} rows)")

    verify(df)


if __name__ == "__main__":
    main()
