#!/usr/bin/env python3
"""generate_data.py — v3 workshop dataset generator (the acts are engineered here).

Produces data/online_banking_visit_clickouts.csv + data/online_banking_daily_cost.csv
(same schema/vocabulary as v2) plus data/ground_truth_epc.csv — with three properties
the teaching arc needs and the v2 data lacked:

  1. REAL CLUSTERING. Business-Checkings visits are comparison-heavy: a clicking
     visit clicks its primary partner 1-4 times, and conversion/payout are driven
     by visit-level latents (buyer status, visit quality). Click-outs from one
     visit are therefore correlated → design effect ≈ 2-3 at click grain.
  2. A CERTIFIED-RANDOM ARTIFACT. In May×Bing the hero pair (Summit Direct
     Business vs Cedar Business Bank) shows an APPARENT flip: Summit's naive
     click-grain t-test fires (p<0.05) while every honest analysis says
     coin-flip. The flip is *found by a seed-hunt*, never hand-planted: ground
     truth is stationary (Cedar better everywhere, incl. Bing) and the hunt
     searches HERO_SEED until an unlucky-draw May produces the pattern.
  3. KNOWN GROUND TRUTH. True EPC per partner×channel is a model constant,
     written to data/ground_truth_epc.csv — so Module 4 can end by revealing
     which method recovered the truth.

Usage:
    python src/generate_data.py                # write CSVs with committed HERO_SEED
    python src/generate_data.py --check        # re-verify gates C1-C10 on written CSVs
    python src/generate_data.py --hunt 500     # search hero seeds (dev only)

Gates certified by the seed (see check_gates):
  C1  naive Welch t (click grain, May Bing) fires: 0.003 < p < 0.05, Summit "ahead"
  C2  cluster bootstrap 95% CI for the diff includes 0        (honest: coin-flip)
  C3  Mann-Whitney p > 0.10                                   (ranks see nothing)
  C4  design effect ≥ 2.0 in the slice                        (the trap mechanism)
  C5  posterior P(Cedar > Summit | May + Mar-Apr prior) ≥ 0.95 (Bayes decisive)
  C6  posterior ordering equals TRUE ordering                  (Bayes right)
  C7  overall chart (all channels, full period): Cedar > Summit, CI excludes 0
  C8  no regime break in Bing EPC (prior is valid; /trend-check will agree)
  C9  visit-grain Welch t p > 0.10                             (simple fix agrees)
  C10 budget beats hold (Google mobile profit < 0; Organic desktop top)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".claude" / "skills" / "_lib"))
import ni_core as C  # noqa: E402  (reviewed primitives; gates reuse them verbatim)

OUT_VISITS = ROOT / "data" / "online_banking_visit_clickouts.csv"
OUT_COST = ROOT / "data" / "online_banking_daily_cost.csv"
OUT_TRUTH = ROOT / "data" / "ground_truth_epc.csv"

# --------------------------------------------------------------------------- #
# Fixed world constants
# --------------------------------------------------------------------------- #
BASE_SEED = 20260301          # everything except the hero cells
HERO_SEED = 4                 # chosen: May conversion rates balanced (16.5% both partners)

# The May artifact, placed deterministically (Route A): payout draws in the hero
# May slice are scaled so the observed means hit these targets exactly.  Scaling
# a lognormal is a lognormal — every value remains a legitimate draw from the
# model's own family; we choose the (unlucky) realization instead of dicing for
# it.  Conversion rates are untouched (seed chosen balanced), so rank tests stay
# silent and the fake gap lives purely in whale-payout magnitude.
HERO_MAY_TARGET = {"Summit Direct Business": 12.6,   # truth 7.8  → lucky whale month
                   "Cedar Business Bank": 7.6}       # truth 10.5 → quiet month
START = pd.Timestamp("2026-03-01")
N_DAYS = 91                   # 2026-03-01 .. 2026-05-30
PRIOR_CUTOFF = pd.Timestamp("2026-05-01")   # drill-down window = May; prior = Mar-Apr

CHANNELS = ["Google", "Bing", "Organic", "Social"]
PLATFORMS = ["desktop", "mobile"]
SEGMENTS = ["Business Checkings", "Checkings", "Savings"]

HERO_CHANNEL, HERO_SEGMENT = "Bing", "Business Checkings"
HERO_A, HERO_B = "Summit Direct Business", "Cedar Business Bank"   # A=apparent May winner; B=truly better

# visits/day per (channel, platform, segment) — magnitudes of the v2 dataset
CELL_RATE = {
    ("Bing", "desktop", "Business Checkings"): 33.5, ("Bing", "desktop", "Checkings"): 76.2,
    ("Bing", "desktop", "Savings"): 84.0, ("Bing", "mobile", "Business Checkings"): 34.3,
    ("Bing", "mobile", "Checkings"): 74.6, ("Bing", "mobile", "Savings"): 81.5,
    ("Google", "desktop", "Business Checkings"): 69.3, ("Google", "desktop", "Checkings"): 155.1,
    ("Google", "desktop", "Savings"): 168.5, ("Google", "mobile", "Business Checkings"): 89.0,
    ("Google", "mobile", "Checkings"): 195.1, ("Google", "mobile", "Savings"): 211.9,
    ("Organic", "desktop", "Business Checkings"): 29.5, ("Organic", "desktop", "Checkings"): 67.3,
    ("Organic", "desktop", "Savings"): 71.7, ("Organic", "mobile", "Business Checkings"): 41.1,
    ("Organic", "mobile", "Checkings"): 91.7, ("Organic", "mobile", "Savings"): 98.5,
    ("Social", "desktop", "Business Checkings"): 10.6, ("Social", "desktop", "Checkings"): 30.0,
    ("Social", "desktop", "Savings"): 26.7, ("Social", "mobile", "Business Checkings"): 38.0,
    ("Social", "mobile", "Checkings"): 105.1, ("Social", "mobile", "Savings"): 95.0,
}
# Bing Business-Checkings spend story: heavy Mar-Apr, cut in May.  This is what
# makes the May drill-down slice thin (and the Mar-Apr prior rich) — visible in
# the cost table too, so the narrative is honest end-to-end.
HERO_VOLUME_MULT = {"mar_apr": 2.2, "may": 0.7}

DOW_MULT = [1.069, 1.084, 1.053, 0.995, 0.962, 0.899, 0.939]   # Mon..Sun
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

CTR = {  # P(visit clicks at least one partner), per (channel, segment)
    ("Bing", "Business Checkings"): 0.238, ("Bing", "Checkings"): 0.444, ("Bing", "Savings"): 0.385,
    ("Google", "Business Checkings"): 0.245, ("Google", "Checkings"): 0.467, ("Google", "Savings"): 0.408,
    ("Organic", "Business Checkings"): 0.236, ("Organic", "Checkings"): 0.463, ("Organic", "Savings"): 0.407,
    ("Social", "Business Checkings"): 0.183, ("Social", "Checkings"): 0.364, ("Social", "Savings"): 0.307,
}

CAMPAIGNS = {
    ("Bing", "Business Checkings", "desktop"): ["bing_business_checking_cash_bonus_desktop", "bing_business_checking_no_fee_desktop", "bing_business_checking_smb_exact_desktop"],
    ("Bing", "Business Checkings", "mobile"): ["bing_business_checking_cash_bonus_mobile", "bing_business_checking_no_fee_mobile", "bing_business_checking_smb_exact_mobile"],
    ("Bing", "Checkings", "desktop"): ["bing_checking_bonus_exact_desktop", "bing_checking_direct_deposit_desktop", "bing_checking_no_fee_desktop"],
    ("Bing", "Checkings", "mobile"): ["bing_checking_bonus_exact_mobile", "bing_checking_mobile_app_mobile", "bing_checking_no_fee_mobile"],
    ("Bing", "Savings", "desktop"): ["bing_savings_best_rates_desktop", "bing_savings_cd_alternative_desktop", "bing_savings_high_yield_exact_desktop"],
    ("Bing", "Savings", "mobile"): ["bing_savings_app_opening_mobile", "bing_savings_best_rates_mobile", "bing_savings_high_yield_exact_mobile"],
    ("Google", "Business Checkings", "desktop"): ["google_business_checking_cash_bonus_desktop", "google_business_checking_no_fee_desktop", "google_business_checking_smb_exact_desktop"],
    ("Google", "Business Checkings", "mobile"): ["google_business_checking_cash_bonus_mobile", "google_business_checking_no_fee_mobile", "google_business_checking_smb_exact_mobile"],
    ("Google", "Checkings", "desktop"): ["google_checking_bonus_exact_desktop", "google_checking_no_fee_desktop", "google_checking_rewards_desktop"],
    ("Google", "Checkings", "mobile"): ["google_checking_bonus_exact_mobile", "google_checking_mobile_app_mobile", "google_checking_no_fee_mobile"],
    ("Google", "Savings", "desktop"): ["google_savings_cd_alternative_desktop", "google_savings_high_yield_exact_desktop", "google_savings_rate_comparison_desktop"],
    ("Google", "Savings", "mobile"): ["google_savings_app_opening_mobile", "google_savings_high_yield_exact_mobile", "google_savings_rate_comparison_mobile"],
    ("Organic", "Business Checkings", "desktop"): ["organic_best_business_checking_accounts_desktop", "organic_business_checking_fees_desktop", "organic_business_checking_reviews_desktop"],
    ("Organic", "Business Checkings", "mobile"): ["organic_best_business_checking_accounts_mobile", "organic_business_checking_fees_mobile", "organic_business_checking_reviews_mobile"],
    ("Organic", "Checkings", "desktop"): ["organic_best_checking_account_bonus_desktop", "organic_checking_account_reviews_desktop", "organic_free_checking_accounts_desktop"],
    ("Organic", "Checkings", "mobile"): ["organic_best_checking_account_bonus_mobile", "organic_checking_account_reviews_mobile", "organic_free_checking_accounts_mobile"],
    ("Organic", "Savings", "desktop"): ["organic_best_high_yield_savings_accounts_desktop", "organic_savings_account_rates_desktop", "organic_savings_account_reviews_desktop"],
    ("Organic", "Savings", "mobile"): ["organic_best_high_yield_savings_accounts_mobile", "organic_savings_account_rates_mobile", "organic_savings_account_reviews_mobile"],
    ("Social", "Business Checkings", "desktop"): ["social_business_checking_cash_bonus_prospecting_desktop", "social_business_checking_fee_guide_desktop", "social_business_checking_smb_retargeting_desktop"],
    ("Social", "Business Checkings", "mobile"): ["social_business_checking_cash_bonus_prospecting_mobile", "social_business_checking_fee_guide_mobile", "social_business_checking_smb_retargeting_mobile"],
    ("Social", "Checkings", "desktop"): ["social_checking_bonus_retargeting_desktop", "social_checking_mobile_app_desktop", "social_checking_rewards_prospecting_desktop"],
    ("Social", "Checkings", "mobile"): ["social_checking_bonus_retargeting_mobile", "social_checking_mobile_app_mobile", "social_checking_rewards_prospecting_mobile"],
    ("Social", "Savings", "desktop"): ["social_savings_app_download_desktop", "social_savings_high_yield_prospecting_desktop", "social_savings_rate_alert_retargeting_desktop"],
    ("Social", "Savings", "mobile"): ["social_savings_app_download_mobile", "social_savings_high_yield_prospecting_mobile", "social_savings_rate_alert_retargeting_mobile"],
}
CAMPAIGN_W = [0.42, 0.33, 0.25]

# --------------------------------------------------------------------------- #
# Segment / partner model.
#   Clicking visit: primary partner ~ share; clicks it R times (R_DIST);
#   BUYER visits (q = conv/c) convert each primary click w.p. c, payout
#   base×u_visit×eps  (u shared across the visit's clicks → clustering).
#   TRUE_EPC per click = conv × base × E[u] × E[eps]  — the ground truth.
# --------------------------------------------------------------------------- #
SEG_CFG = {
    "Business Checkings": dict(
        partners=["Cedar Business Bank", "Summit Direct Business", "Metro SMB Banking", "HarborOne Business"],
        shares=[0.523, 0.265, 0.142, 0.070],
        position={"Cedar Business Bank": 1, "Summit Direct Business": 2, "Metro SMB Banking": 3, "HarborOne Business": 4},
        R_vals=[1, 2, 3, 4], R_probs=[0.22, 0.30, 0.30, 0.18],   # comparison-heavy product
        c_click=0.8, sigma_v=0.6, sigma_e=0.1, sec_prob=0.10,
        conv={"Cedar Business Bank": 0.16, "Summit Direct Business": 0.16,
              "Metro SMB Banking": 0.118, "HarborOne Business": 0.086},
        true_epc={  # ground truth: Cedar > Summit in EVERY channel (incl. Bing)
            "Cedar Business Bank": {"Google": 10.1, "Bing": 10.5, "Organic": 13.0, "Social": 3.4},
            "Summit Direct Business": {"Google": 7.3, "Bing": 7.8, "Organic": 9.0, "Social": 2.6},
            "Metro SMB Banking": {"Google": 11.6, "Bing": 5.7, "Organic": 7.1, "Social": 2.2},
            "HarborOne Business": {"Google": 6.1, "Bing": 5.4, "Organic": 1.6, "Social": 1.3},
        },
    ),
    "Checkings": dict(
        partners=["BlueRiver Checking", "Evergreen Rewards Bank", "NorthStar Bank", "HarborOne Digital"],
        shares=[0.590, 0.242, 0.114, 0.054],
        position={"BlueRiver Checking": 1, "Evergreen Rewards Bank": 2, "NorthStar Bank": 3, "HarborOne Digital": 4},
        R_vals=[1], R_probs=[1.0], c_click=1.0, sigma_v=0.45, sigma_e=0.85, sec_prob=0.21,
        conv={"BlueRiver Checking": 0.245, "Evergreen Rewards Bank": 0.273,
              "NorthStar Bank": 0.211, "HarborOne Digital": 0.210},
        true_epc={
            "BlueRiver Checking": {"Google": 4.59, "Bing": 4.52, "Organic": 4.69, "Social": 3.19},
            "Evergreen Rewards Bank": {"Google": 3.32, "Bing": 3.26, "Organic": 3.57, "Social": 2.60},
            "NorthStar Bank": {"Google": 3.04, "Bing": 2.93, "Organic": 3.15, "Social": 2.18},
            "HarborOne Digital": {"Google": 4.79, "Bing": 4.81, "Organic": 4.63, "Social": 3.34},
        },
    ),
    "Savings": dict(
        partners=["NorthStar Bank", "Summit Direct", "GreenLeaf Savings", "HarborOne Digital"],
        shares=[0.463, 0.358, 0.123, 0.056],
        position={"NorthStar Bank": (1, 2, 0.68), "Summit Direct": (1, 2, 0.70),
                  "GreenLeaf Savings": 3, "HarborOne Digital": 4},
        R_vals=[1], R_probs=[1.0], c_click=1.0, sigma_v=0.45, sigma_e=0.85, sec_prob=0.18,
        conv={"NorthStar Bank": 0.196, "Summit Direct": 0.220,
              "GreenLeaf Savings": 0.176, "HarborOne Digital": 0.165},
        true_epc={
            "NorthStar Bank": {"Google": 3.94, "Bing": 3.55, "Organic": 3.69, "Social": 2.54},
            "Summit Direct": {"Google": 8.71, "Bing": 7.81, "Organic": 8.52, "Social": 4.96},
            "GreenLeaf Savings": {"Google": 4.92, "Bing": 5.07, "Organic": 4.23, "Social": 3.15},
            "HarborOne Digital": {"Google": 4.59, "Bing": 4.48, "Organic": 4.68, "Social": 3.42},
        },
    ),
}

# CPV/EPV ratios per (channel, platform) — reproduces the v2 budget beats
# (Google mobile loses money; Organic essentially free) from realized EPVs.
COST_RATIO = {("Bing", "desktop"): 0.784, ("Bing", "mobile"): 0.810,
              ("Google", "desktop"): 0.951, ("Google", "mobile"): 1.095,
              ("Organic", "desktop"): 0.0163, ("Organic", "mobile"): 0.0209,
              ("Social", "desktop"): 0.846, ("Social", "mobile"): 0.941}


def payout_base(seg_cfg, partner, channel):
    """base so that TRUE_EPC = conv × base × E[u_visit] × E[eps]."""
    lift = np.exp((seg_cfg["sigma_v"] ** 2 + seg_cfg["sigma_e"] ** 2) / 2.0)
    return seg_cfg["true_epc"][partner][channel] / (seg_cfg["conv"][partner] * lift)


# --------------------------------------------------------------------------- #
# Cell generation (one call = one channel×platform×segment, all 91 days)
# --------------------------------------------------------------------------- #
def _position_of(spec, rng):
    return spec if isinstance(spec, int) else (spec[0] if rng.random() < spec[2] else spec[1])


def gen_cell(channel, platform, segment, rng):
    cfg = SEG_CFG[segment]
    partners = cfg["partners"]
    conv = np.array([cfg["conv"][p] for p in partners])
    q_buyer = conv / cfg["c_click"]
    base = np.array([payout_base(cfg, p, channel) for p in partners])

    rate = CELL_RATE[(channel, platform, segment)]
    days = np.arange(N_DAYS)
    dates = START + pd.to_timedelta(days, unit="D")
    mult = np.array([DOW_MULT[d.dayofweek] for d in dates])
    if channel == HERO_CHANNEL and segment == HERO_SEGMENT:
        mult = mult * np.where(dates < PRIOR_CUTOFF, HERO_VOLUME_MULT["mar_apr"], HERO_VOLUME_MULT["may"])
    n_per_day = rng.poisson(rate * mult)
    n_vis = int(n_per_day.sum())

    day_of_visit = np.repeat(days, n_per_day)
    ts = (day_of_visit * 86400 + rng.uniform(7 * 3600, 22.9 * 3600, n_vis)).astype(int)
    camp = rng.choice(CAMPAIGNS[(channel, segment, platform)], size=n_vis, p=CAMPAIGN_W)
    clicked = rng.random(n_vis) < CTR[(channel, segment)]

    rows = {k: [] for k in ("vis", "ts", "partner", "pos", "conv", "rev")}

    def add(vis_i, t, partner_i, converted, revenue):
        rows["vis"].append(vis_i)
        rows["ts"].append(t)
        rows["partner"].append(partners[partner_i])
        rows["pos"].append(_position_of(cfg["position"][partners[partner_i]], rng))
        rows["conv"].append(int(converted))
        rows["rev"].append(round(float(revenue), 4))

    ck = np.flatnonzero(clicked)
    prim = rng.choice(len(partners), size=len(ck), p=cfg["shares"])
    R = rng.choice(cfg["R_vals"], size=len(ck), p=cfg["R_probs"])
    buyer = rng.random(len(ck)) < q_buyer[prim]
    u_v = rng.lognormal(0.0, cfg["sigma_v"], len(ck))
    sec = rng.random(len(ck)) < cfg["sec_prob"]

    for j, vi in enumerate(ck):
        pi, t = int(prim[j]), int(ts[vi])
        for _ in range(int(R[j])):
            t += int(rng.uniform(20, 240))
            cvt = bool(buyer[j]) and (rng.random() < cfg["c_click"])
            rev = base[pi] * u_v[j] * rng.lognormal(0.0, cfg["sigma_e"]) if cvt else 0.0
            add(vi, t, pi, cvt, rev)
        if sec[j]:
            others = [k for k in range(len(partners)) if k != pi]
            w = np.array(cfg["shares"])[others]
            si = int(rng.choice(others, p=w / w.sum()))
            t += int(rng.uniform(20, 240))
            cvt2 = rng.random() < conv[si]
            rev2 = base[si] * rng.lognormal(0.0, cfg["sigma_v"]) * rng.lognormal(0.0, cfg["sigma_e"]) if cvt2 else 0.0
            add(vi, t, si, cvt2, rev2)

    click_df = pd.DataFrame({
        "local_vis": rows["vis"], "ts": rows["ts"],
        "partner": rows["partner"], "clickout_position": rows["pos"],
        "converted": rows["conv"], "revenue": rows["rev"]})
    click_df["campaign"] = camp[click_df["local_vis"].to_numpy()] if len(click_df) else []
    noclick = np.flatnonzero(~clicked)
    noclick_df = pd.DataFrame({
        "local_vis": noclick, "ts": ts[noclick], "partner": None,
        "clickout_position": np.nan, "converted": 0, "revenue": 0.0,
        "campaign": camp[noclick]})
    cell = pd.concat([click_df, noclick_df], ignore_index=True)
    cell["channel"], cell["platform"], cell["segment"] = channel, platform, segment
    return cell


_BASE_CACHE: dict = {}


def build_dataset(hero_seed=HERO_SEED, light=False):
    """All 24 cells. Hero cells (Bing × Business Checkings) draw from hero_seed
    so the seed-hunt varies ONLY the hero draw; the rest of the world is fixed.
    light=True (seed-hunt): reuse cached non-hero cells, keep string visit ids,
    skip the global sort — gate-equivalent, ~7× faster."""
    frames = []
    for ci, (ch, pl, seg) in enumerate(sorted(CELL_RATE)):
        hero = (ch == HERO_CHANNEL and seg == HERO_SEGMENT)
        if hero:
            seed = [hero_seed, ci]
            cell = gen_cell(ch, pl, seg, np.random.default_rng(np.random.SeedSequence(seed)))
            cell["cell"] = ci
        else:
            if ci not in _BASE_CACHE:
                cell = gen_cell(ch, pl, seg, np.random.default_rng(np.random.SeedSequence([BASE_SEED, ci])))
                cell["cell"] = ci
                _BASE_CACHE[ci] = cell
            cell = _BASE_CACHE[ci]
        frames.append(cell)
    df = pd.concat(frames, ignore_index=True)

    key = df["cell"].astype(str) + "_" + df["local_vis"].astype(str)
    if light:
        df["visit_iid"] = key            # cluster label is all the gates need
    else:
        uniq = pd.unique(key)            # global visit ids: shuffled ints, deterministic
        perm = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 999])).permutation(len(uniq))
        df["visit_iid"] = key.map(dict(zip(uniq, perm)))

    df["click_timestamp"] = START + pd.to_timedelta(df["ts"], unit="s")
    if not light:
        df = df.sort_values(["click_timestamp", "visit_iid"], kind="stable").reset_index(drop=True)
    df["clicked"] = df["partner"].notna()
    df["date"] = df["click_timestamp"].dt.normalize()
    df["month"] = df["click_timestamp"].dt.strftime("%Y-%m")
    df["day_of_week"] = df["click_timestamp"].dt.dayofweek.map(lambda i: DOW_NAMES[i])
    _shape_hero_may(df)
    return df


def _shape_hero_may(df):
    """Place the May artifact: scale hero-partner May payouts to HERO_MAY_TARGET."""
    may = (df["clicked"] & (df["channel"] == HERO_CHANNEL)
           & (df["segment"] == HERO_SEGMENT) & (df["click_timestamp"] >= PRIOR_CUTOFF))
    for who, target in HERO_MAY_TARGET.items():
        mine = may & (df["partner"] == who)
        cur = df.loc[mine, "revenue"].mean()
        if cur > 0:
            nz = mine & (df["revenue"] > 0)
            df.loc[nz, "revenue"] = (df.loc[nz, "revenue"] * (target / cur)).round(4)


# --------------------------------------------------------------------------- #
# Gates — the dataset's own regression test (mirrors what the skills compute)
# --------------------------------------------------------------------------- #
def _hero_slices(df):
    cl = df[df["clicked"] & (df["channel"] == HERO_CHANNEL) & (df["segment"] == HERO_SEGMENT)]
    may = cl[cl["click_timestamp"] >= PRIOR_CUTOFF]
    pre = cl[cl["click_timestamp"] < PRIOR_CUTOFF]
    return {"may": {"A": may[may["partner"] == HERO_A], "B": may[may["partner"] == HERO_B]},
            "pre": {"A": pre[pre["partner"] == HERO_A], "B": pre[pre["partner"] == HERO_B]}}


def check_gates(df, n_boot=C.N_BOOT, verbose=True):
    g = _hero_slices(df)
    a, b = g["may"]["A"], g["may"]["B"]
    av, bv = a["revenue"].to_numpy(float), b["revenue"].to_numpy(float)
    res, out = {}, []

    t_p = stats.ttest_ind(av, bv, equal_var=False).pvalue
    res["C1"] = bool((0.003 < t_p < 0.05) and av.mean() > bv.mean())
    out.append(f"C1 naive Welch t (clicks, May Bing): p={t_p:.4f}  means {av.mean():.2f} vs {bv.mean():.2f} (n={len(av)}/{len(bv)})")

    boot = C.cluster_bootstrap_diff_ci(av, a["visit_iid"].to_numpy(), bv, b["visit_iid"].to_numpy(), n_boot=n_boot)
    res["C2"] = bool(boot["diff_lo"] < 0 < boot["diff_hi"])
    out.append(f"C2 cluster bootstrap diff CI: [{boot['diff_lo']:+.2f}, {boot['diff_hi']:+.2f}]")

    mw_p = stats.mannwhitneyu(av, bv, alternative="two-sided").pvalue
    res["C3"] = bool(mw_p > 0.10)
    out.append(f"C3 Mann-Whitney: p={mw_p:.3f}")

    pooled = pd.concat([a, b])
    de = C.design_effect(pooled["revenue"].to_numpy(float), pooled["visit_iid"].to_numpy())
    res["C4"] = bool(de["deff"] >= 2.0)
    out.append(f"C4 design effect (slice): deff={de['deff']:.2f}  icc={de['icc']:.2f}  n={de['n']} -> n_eff={de['n_eff']:.0f}")

    post = {}
    for who, d_may, d_pre in ((HERO_A, a, g["pre"]["A"]), (HERO_B, b, g["pre"]["B"])):
        dm, dse2, _ = C.mean_and_se2(d_may["revenue"].to_numpy(float))
        dse2 *= C.design_effect(d_may["revenue"].to_numpy(float), d_may["visit_iid"].to_numpy())["deff"]
        pm, pse2, _ = C.mean_and_se2(d_pre["revenue"].to_numpy(float))
        pse2 *= C.design_effect(d_pre["revenue"].to_numpy(float), d_pre["visit_iid"].to_numpy())["deff"]
        post[who] = C.posterior_from_prior(pm, pse2, dm, dse2)
    p_b = C.p_better(post[HERO_B]["mean"], post[HERO_B]["var"], post[HERO_A]["mean"], post[HERO_A]["var"])
    res["C5"] = bool(p_b >= 0.95)
    res["C6"] = bool(post[HERO_B]["mean"] > post[HERO_A]["mean"])  # truth: Cedar > Summit
    out.append(f"C5 posterior P({HERO_B} > {HERO_A}) = {p_b:.3f}   "
               f"posteriors {post[HERO_B]['mean']:.2f} vs {post[HERO_A]['mean']:.2f}")
    out.append(f"C6 posterior ordering matches TRUE ordering (Cedar 10.5 > Summit 7.8): {res['C6']}")

    cl_all = df[df["clicked"]]
    oa = cl_all[cl_all["partner"] == HERO_A]["revenue"].to_numpy(float)
    ob = cl_all[cl_all["partner"] == HERO_B]["revenue"].to_numpy(float)
    ov = C.bootstrap_mean_diff_ci(ob, oa, n_boot=n_boot)
    res["C7"] = bool(ov["diff_lo"] > 0)
    out.append(f"C7 overall chart (all ch., full period): Cedar-Summit = {ov['diff']:+.2f} CI [{ov['diff_lo']:+.2f}, {ov['diff_hi']:+.2f}]")

    bing = df[df["clicked"] & (df["channel"] == HERO_CHANNEL)]
    tr = C.trend_check(bing, "revenue")
    res["C8"] = bool((not tr.get("regime_break", False)) and tr.get("within_band", False))
    out.append(f"C8 Bing EPC trend: regime_z={tr.get('regime_z', float('nan')):+.2f}  within_band={tr.get('within_band')}")

    va = a.groupby("visit_iid")["revenue"].mean().to_numpy()
    vb = b.groupby("visit_iid")["revenue"].mean().to_numpy()
    tv_p = stats.ttest_ind(va, vb, equal_var=False).pvalue
    res["C9"] = bool(tv_p > 0.10)
    out.append(f"C9 visit-grain Welch t: p={tv_p:.3f}  (visits {len(va)}/{len(vb)})")

    if verbose:
        s = C.describe_shape(np.concatenate([av, bv]))
        out.append(f"--  slice shape: zero-share={s['zero_share']*100:.0f}%  skew={s['skew']:.1f}  top-1% holds {s['top1_share']*100:.0f}%")
        print("\n".join(out))
    return res, out


def check_budget(df, cost):
    vis = (df.groupby("visit_iid")
             .agg(channel=("channel", "first"), platform=("platform", "first"),
                  converted=("converted", "max"), revenue=("revenue", "sum")).reset_index())
    tab = C.budget_table(vis, cost).set_index(["channel", "platform"])
    gm = float(tab.loc[("Google", "mobile"), "profit/visit"])
    top = tab["profit/visit"].idxmax()
    ok = bool(gm < 0 and top == ("Organic", "desktop"))
    line = f"C10 budget beats: Google-mobile profit/visit={gm:+.3f} (<0), top cell={top}"
    print(line)
    return ok, line


# --------------------------------------------------------------------------- #
# Cost table (derived from realized EPVs -> beats hold by construction)
# --------------------------------------------------------------------------- #
def build_cost(df):
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 777]))
    vis = (df.groupby("visit_iid")
             .agg(date=("date", "first"), channel=("channel", "first"),
                  platform=("platform", "first"), segment=("segment", "first"),
                  campaign=("campaign", "first"), revenue=("revenue", "sum")).reset_index())
    epv = vis.groupby(["channel", "platform"])["revenue"].mean()
    counts = (vis.groupby(["date", "channel", "platform", "segment", "campaign"])
                 .size().rename("visits").reset_index())
    counts["cpv"] = [epv[(r.channel, r.platform)] * COST_RATIO[(r.channel, r.platform)]
                     for r in counts.itertuples()]
    counts["cost"] = (counts["visits"] * counts["cpv"] * rng.uniform(0.95, 1.05, len(counts))).round(3)
    counts["day_of_week"] = counts["date"].dt.dayofweek.map(lambda i: DOW_NAMES[i])
    counts["is_weekend"] = counts["date"].dt.dayofweek >= 5
    counts["date"] = counts["date"].dt.strftime("%Y-%m-%d")
    return counts[["date", "day_of_week", "is_weekend", "channel", "platform", "segment", "campaign", "cost"]]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_all(df):
    out = df.copy()
    out["day_of_week"] = out["click_timestamp"].dt.dayofweek.map(lambda i: DOW_NAMES[i])
    out["is_weekend"] = out["click_timestamp"].dt.dayofweek >= 5
    out["click_timestamp"] = out["click_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["click_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    ck = out.index[out["clicked"]]
    out.loc[ck, "click_id"] = [f"clk_{i:09d}" for i in range(len(ck))]
    out["clickout_position"] = out["clickout_position"].astype("Int64")
    cols = ["visit_iid", "click_timestamp", "day_of_week", "is_weekend", "channel",
            "platform", "segment", "campaign", "click_id", "partner",
            "clickout_position", "converted", "revenue"]
    out[cols].to_csv(OUT_VISITS, index=False)

    cost = build_cost(df)
    cost.to_csv(OUT_COST, index=False)

    truth = [{"segment": seg, "partner": p, "channel": ch,
              "true_epc": cfg["true_epc"][p][ch], "true_conv_per_click": cfg["conv"][p],
              "true_rev_per_conv": round(cfg["true_epc"][p][ch] / cfg["conv"][p], 2)}
             for seg, cfg in SEG_CFG.items() for p in cfg["partners"] for ch in CHANNELS]
    pd.DataFrame(truth).to_csv(OUT_TRUTH, index=False)
    print(f"wrote {OUT_VISITS.name} ({len(out):,} rows, {df['visit_iid'].nunique():,} visits), "
          f"{OUT_COST.name} ({len(cost):,} rows), {OUT_TRUTH.name}")
    return cost


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def hunt(n_seeds, start=0):
    print(f"hunting hero seeds {start}..{start + n_seeds - 1} (gates C1-C9, fast bootstrap)")
    for seed in range(start, start + n_seeds):
        df = build_dataset(hero_seed=seed, light=True)
        g = _hero_slices(df)
        av = g["may"]["A"]["revenue"].to_numpy(float)
        bv = g["may"]["B"]["revenue"].to_numpy(float)
        if not (av.mean() > bv.mean()):                       # cheap pre-gates first
            continue
        t_p = stats.ttest_ind(av, bv, equal_var=False).pvalue
        if not (0.003 < t_p < 0.05):
            continue
        res, _ = check_gates(df, n_boot=500, verbose=False)
        status = " ".join(f"{k}:{'+' if v else '-'}" for k, v in sorted(res.items()))
        print(f"  seed {seed}: t_p={t_p:.4f}  {status}")
        if all(res.values()):
            print(f"  >>> seed {seed} passes fast gates; re-verifying with full n_boot…")
            res2, _ = check_gates(df, verbose=True)
            if all(res2.values()):
                print(f"\n*** HERO_SEED = {seed} ***  (commit this constant)")
                return seed
    print("no seed passed — widen the range or revisit parameters")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunt", type=int, default=0, help="search N hero seeds (dev)")
    ap.add_argument("--start", type=int, default=0, help="first seed for --hunt")
    ap.add_argument("--seed", type=int, default=HERO_SEED, help="hero seed to build with")
    ap.add_argument("--check", action="store_true", help="validate the WRITTEN csvs against gates C1-C10")
    args = ap.parse_args()

    if args.hunt:
        hunt(args.hunt, args.start)
        return

    if args.check:
        df = C.load_clickouts(OUT_VISITS)
        if "month" not in df.columns:
            df["month"] = df["click_timestamp"].dt.strftime("%Y-%m")
        print(C.hr("DATASET VALIDATION — gates the notebooks & skills rely on"))
        res, _ = check_gates(df)
        ok10, _ = check_budget(df, C.load_cost(OUT_COST))
        res["C10"] = ok10
        print(C.hr("VERDICT"))
        bad = [k for k, v in res.items() if not v]
        if bad:
            print(C.flag(f"FAILED gates: {bad}"))
            sys.exit(1)
        print(C.ok("all gates pass — the dataset certifies the three-act arc."))
        return

    df = build_dataset(hero_seed=args.seed)
    res, _ = check_gates(df)
    if not all(res.values()):
        print(C.flag(f"gates failing for seed {args.seed}: {[k for k, v in res.items() if not v]}"))
        sys.exit(1)
    write_all(df)


if __name__ == "__main__":
    main()
