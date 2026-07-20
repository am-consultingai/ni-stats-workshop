# Workshop TODO — open items (deferred)

Coherence seams and cleanups found while reviewing the M0→M7 spine on the real
`online_banking` data. The statistics are sound; these are mostly **narrative
coherence** fixes. Ordered by priority.

## Narrative coherence

- [ ] **#1 (biggest) — M6 silently switches the unit of analysis.** The spine is about
      two **partners** (Summit vs Cedar), but M6 / `/decide`'s budget reframe is by
      **channel × platform** (Google mobile, Bing desktop…) — the two partners never appear
      in that table. The logic is fine ("stop obsessing over the partner EPC gap; the real
      lever is channel×platform profit") but the story jumps from "which partner?" to
      "which channel?" with no bridge. **Fix:** add a bridging passage in M6 (and the
      `/decide` synthesis) that closes the partner question before pivoting to the budget metric.

- [ ] **#2 — M4 only teaches the *reversal* flavor of Bayes.** It shows history *killing* a
      false positive (Summit/Cedar → P=43%, don't reorder) but never history *rescuing* a
      thin-but-real signal. The "reinforce" flow is latent in the skill, absent from the notebook.
      **Fix:** add a short (~2-cell) contrast case to `04_bayesian_inference` using
      **Metro SMB Banking vs NorthStar Bank in Bing**: frequentist CI [−0.63, +6.26] includes 0
      (coin-flip), but Bayes anchors Metro's thin slice (n=233) on its strong history (9.06) →
      posterior 7.64 vs 3.48, **P(Metro>NorthStar)=100%, "a defensible case to act."** Same
      coin-flip start, opposite resolution. (Skill already prints the REINFORCE verdict.)

- [ ] **#3 — M3 teaches on a different example than the spine.** The notebook demonstrates the
      test mechanics on **Google vs Bing *conversion*** (a clean, significant result), then the
      *skill* is applied to **Summit vs Cedar *EPC*** (the coin-flip). Deliberate, but the
      hand-off is implicit. **Fix:** one bridging sentence connecting the taught example to the
      spine slice.

- [ ] **#4 — `/relationship` (confound / Simpson's) is taught in M3 but not wired into `/decide`.**
      The orchestrator runs profile→significance→bayesian→trend→budget; the confound check sits
      on the side. **Fix:** either fold a confound step into `/decide`, or state explicitly in
      the notebook + `/decide` that it's a manual guard.

- [ ] **M0 A/B label flip.** Cell 0's abstract framing says *"partner B ($9) beats partner A ($8)"*
      (B higher), but the concrete code assigns **Summit = A** as the higher one ($10.48). Align
      the abstract labels with the code. Two-word fix.

## Cleanup

- [ ] **Obsolete authoring scripts.** `src/generate_data.py` and `src/build_notebooks.py` created
      the now-deleted synthetic data + old notebooks. Nothing runs or imports them during the
      workshop. Decide: archive or delete.

- [ ] **Commit the current work.** The rebasing (synthetic data removed; M1/M3/M6 + 3 reference
      notebooks rebased onto real data; `ni_style` → presentation-only; `setup.sh` +
      `requirements.txt` added) is staged/unstaged but **not committed**. On a branch (currently on `main`).
