# exp_10 Report: External Full-Schema Replication (Frame.md's 37 Variables + MRI-PCA(2)) under ARD-KDM

## 1. Summary

Adopting the external repo's complete 37-variable tabular schema (`Frame.md`, minus 2 exact
duplicates found in EDA) + MRI-PCA(2) — 48 encoded columns, roughly double `exp_9`'s
already-flagged-as-risky 23-column frame — under `exp_9`'s ARD-KDM backbone and `Frame.md`'s own
preprocessing convention (MinMax + one-hot + missing-flags) initially produced the **sharpest
CV/held-out disagreement this project has seen**: cross-validation showed a clean, uniform
regression across every single subtask relative to `exp_9`'s 23-column ARD reference, while the
mandatory held-out check showed what looked like the **best decision result this project has ever
produced** (0.708 macro-F1). This was the fourth consecutive KDM architecture experiment
(`exp_7`–`exp_10`) to show CV/held-out disagreement, and the most extreme case yet — so, per the
same-session follow-up request, a leave-one-out CV check and 10 repeated held-out splits were run
against both frames (§3b). **The disagreement is now resolved: the 0.708 held-out result was a
lucky single split** (repeated-split mean 0.509 ± 0.134, LOO 0.530 — both close to CV's original
0.548, nowhere near 0.708). AUROC and Brier score, added in a second follow-up (§3c), make this
unanimous: `exp_9`'s 23-column frame wins on **every one of 4 verification methods × 2 metrics, no
exceptions** — including on the exact seed=0 split where the full-schema frame's macro-F1 briefly
looked best-ever, its AUROC/Brier were *already* worse. The full-schema frame is worse than `exp_9`'s
reference on every subtask, confirmed by every verification method and metric now available, not
just CV.

One clear, unambiguous negative finding survived both signals: **`confidence_kdm_dispersion_isotonic`'s
regression, which `exp_9`'s ARD mechanism successfully closed at 23 columns, re-opens at 48** —
ordinal distance goes `0.804` (`exp_9`, 23-col ARD) → `1.410` (`exp_10`, 48-col ARD), a larger
regression than the original scalar-bandwidth problem that motivated ARD in the first place
(`exp_8`'s `0.776→1.085`). ARD's per-dimension bandwidth rescued the dilution problem once, at a
scale it was designed and tested for; it did not rescue it a second time at roughly double the
frame width. Reveal-sequence also regressed on both metrics, and — for the first time since
`exp_6` — **underperformed the naive baseline** on set-precision (0.718 vs. 0.783).

| Subtask | This experiment's result | Baseline | Incumbent | Beats baseline? | Beats incumbent? |
|---|---|---|---|---|---|
| Decision (CV) | 0.548 macro-F1 | 0.381 | 0.650 (Extra Trees) | ✅ | ❌ |
| Decision (held-out, seed=0) | 0.708 macro-F1 *(see §3b: not representative — 10-seed mean is 0.509)* | 0.381 | 0.650 (Extra Trees) | ✅ | ❌ once corrected for seed selection |
| Decision (LOO, 91-fold pooled) | 0.530 macro-F1 | 0.381 | 0.650 (Extra Trees) | ✅ | ❌ |
| Confidence | 0.797 ord. dist. best (`kernel_distance`-blend n/a; `dispersion_isotonic`=1.410 worst) | 0.527 | 0.468 (`confidence_svm`) | ❌ | ❌ |
| Variable weights | 0.537 ord. error (`occlusion`) | 0.413 | 0.382/0.392 (`weights_svm`) | ❌ | ❌ |
| Reveal | 0.718 set-precision (`occlusion`) | 0.783 | 0.853 (`reveal_flags`) | ❌ (first time any KDM reveal condition has lost to baseline) | ❌ |

## 2. What Was Run

9 CV conditions (decision + 5 confidence signals + 3 weights mechanisms) + the mandatory held-out
check + reveal + an importance-comparison diagnostic, all on the 48-column full-schema frame, fixed
ARD hyperparameters (`n_epochs=300, lr=1e-2, sigma_mult=1.0`, matching `exp_9`'s defaults exactly —
no search, per `DESIGN.md` §1's guardrail). New preprocessing (`fit_transform_fullschema`, in
`features_fullschema.py`) replaces `build_preprocessor`+`StandardScaler`: `Frame.md`'s own
convention (MinMax scaling, one-hot categoricals built as plain fixed-category comparisons rather
than `sklearn.OneHotEncoder`, and explicit missing-flags instead of blanket median imputation —
only 5 of the 48 columns get genuine per-fold-fit median imputation). All four smoke tests from
`IMPLEMENTATION.md` passed before the full runs: frame shape/NaN pattern, transform shape/range,
factor-group column resolution, and one ARD fit+signals+occlusion call (`probs_check_ok=True`).

The DESIGN.md arithmetic was corrected twice during implementation — the frame is **48** columns
(37 raw source fields), not the 46 or 47 stated at various earlier points; verified
programmatically, not by hand, after two manual counting errors. No decisions changed.

## 3. Decision: CV says clear regression, held-out says best result ever — the sharpest disagreement this project has produced

| Comparison | CV macro-F1 | Held-out macro-F1 (n=19) |
|---|---|---|
| `exp_9` ARD, 23-col (reference) | 0.608 | 0.680 |
| `exp_10` ARD, 48-col full-schema | **0.548** (Δ**-0.060**) | **0.708** (Δ**+0.028**) |
| `exp_6` scalar, 19-col (original reference) | 0.593 | 0.593 |

Two completely different stories from the same model family, same held-out discipline this project
has used since `exp_3`:

1. **CV shows a clean, uniform regression.** Not just decision — every one of the 9 CV conditions
   run this round (confidence's 5 signals, weights' 3 mechanisms) came in worse than `exp_9`'s
   23-column ARD numbers (§4/§5 below). A single noisy number would be easy to discount; a
   consistent regression across every condition in the run is a much stronger signal that
   something about the wider frame is genuinely hurting fold-to-fold generalization under CV.
2. **Held-out shows the best decision result this project has ever produced.** 0.708 macro-F1 on
   the fixed 19-case split, beating `exp_9`'s already-strong 0.680, `exp_6`'s original 0.593, and —
   for the first time — the Extra Trees incumbent (0.650). `classification_report` shows a
   reasonably balanced result (no-class F1=0.615, yes-class F1=0.800), not a degenerate
   all-one-class prediction.

Per `DESIGN.md` §8's decision rules, **none of the three anticipated branches cleanly fits this
outcome** — the design anticipated either a clean win, a marginal result, or "CV fine, held-out
bad" (the classic overfitting signature `exp_7` showed). This is the reverse: CV bad, held-out
good — flagged in the original version of this section as needing a follow-up replication before
trusting 0.708. §3b resolves it.

### 3b. Follow-up verification: the 0.708 held-out result was a lucky single split

Run immediately after this report's first draft, per the user's request: leave-one-out CV (91
folds, pooled prediction, deterministic) and 10 repeated 80/20 held-out splits (seeds 0-9, seed 0
reproducing the already-reported numbers exactly as a same-script sanity check), on both frames —
`experiments/exp_10/scripts/verify_decision_loo_repeated_holdout.py`.

| Frame | CV (original) | Held-out, seed=0 (original) | Held-out, 10-seed mean±std | LOO (91-fold, pooled) |
|---|---|---|---|---|
| `exp_9` ARD, 23-col | 0.608 | 0.680 | 0.670 ± 0.064 | 0.639 |
| `exp_10` ARD, 48-col full-schema | 0.548 | **0.708** | **0.509 ± 0.134** | **0.530** |

This resolves the disagreement cleanly, and **not in the full-schema frame's favor**:

- **`exp_9`'s 23-column result is confirmed genuinely robust.** All four estimates cluster tightly
  (0.608–0.680, std across repeated splits only 0.064) — the original seed=0 held-out number
  (0.680) was on the optimistic side of a fairly narrow distribution, not an outlier. This is
  independent, additional confirmation that `exp_9`'s ARD improvement over `exp_6` is real.
- **`exp_10`'s 0.708 held-out result was a lucky single split, not a representative estimate.**
  The other 9 seeds range from 0.244 to 0.661 (mean 0.509) — a much wider spread (std 0.134, more
  than double the 23-column frame's) — and LOO's deterministic, pooled 91-fold estimate (0.530)
  lands close to that same mean, not anywhere near 0.708. **Both new, independent methods agree
  with the original CV estimate (0.548), not with the single held-out split that briefly looked
  like the project's best-ever result.**

The practical lesson this confirms for the project going forward: **a single 19-case held-out
split is genuinely unreliable at this N** — a std of 0.134 across just 10 draws means any one
split can land ±0.13 or more away from the true mean, easily enough to manufacture an apparent
"best result ever" from a frame that is, on the balance of evidence, worse than the existing
reference. `exp_9`'s original held-out number happened to be checked this way too and held up;
`exp_10`'s did not. Neither could have been told apart from a single split alone — this is the
concrete case study future experiments should point to when deciding whether a single held-out
check is enough.

### 3c. AUROC + Brier score: unanimous confirmation, and a clue about *why* macro-F1's single split misled

Added per a follow-up request: AUROC and Brier score, backfilled into every decision result already
produced (`experiments/exp_9/scripts/backfill_decision_auroc_brier.py`,
`experiments/exp_10/scripts/backfill_decision_auroc_brier.py`, and folded into
`verify_decision_loo_repeated_holdout.py`'s LOO/repeated-holdout runs). CV/repeated-holdout values
are per-repeat-pooled then averaged, matching macro-F1's own aggregation convention.

| Metric | Method | `exp_9` ARD, 23-col | `exp_10` ARD, 48-col full-schema |
|---|---|---|---|
| AUROC (↑ better) | CV | 0.676 ± 0.018 | 0.570 ± 0.027 |
| AUROC | Held-out, seed=0 | **0.857** | 0.833 |
| AUROC | Held-out, 10-seed mean±std | 0.739 ± 0.062 | 0.559 ± 0.155 |
| AUROC | LOO (91-fold, pooled) | 0.694 | 0.574 |
| Brier score (↓ better) | CV | 0.269 ± 0.013 | 0.325 ± 0.019 |
| Brier score | Held-out, seed=0 | **0.170** | 0.202 |
| Brier score | Held-out, 10-seed mean±std | 0.240 ± 0.044 | 0.340 ± 0.095 |
| Brier score | LOO (91-fold, pooled) | 0.264 | 0.327 |

**Every single row favors `exp_9`'s 23-column frame — no exceptions, across 4 independent
verification methods and 2 metrics.** This is more unanimous than the macro-F1 picture ever was,
even before §3b's resolution.

The most telling row is the first held-out one: **on the exact same seed=0 split where the
full-schema frame's macro-F1 (0.708) briefly looked like the project's best-ever result, its AUROC
(0.833) and Brier score (0.202) were *already* worse than `exp_9`'s 23-column frame on that same
split (0.857 / 0.170).** Macro-F1 is computed from hard `argmax` predictions, sensitive to exactly
where the 0.5 decision boundary happens to fall; AUROC and Brier score both grade the full predicted
probability, not just which side of a threshold it lands on. The full-schema frame's one lucky split
happened to place its argmax boundary favorably for that particular 19-case draw, while its
underlying probability estimates were — correctly, as it turns out — already ranked as worse than
`exp_9`'s. **AUROC/Brier would have caught this without needing §3b's LOO/repeated-holdout
follow-up at all** — worth remembering for future single-split held-out checks: look at AUROC/Brier
alongside the headline classification metric, not after the fact.

## 4. Confidence: `dispersion_isotonic`'s regression re-opens, worse than the original problem

| Signal | `exp_9` ARD 23-col ord.dist / macro-F1 | `exp_10` ARD 48-col ord.dist / macro-F1 |
|---|---|---|
| `entropy_zeroshot` | 1.415 / 0.119 | 1.433 / 0.109 |
| `entropy_isotonic` | 0.945 / 0.177 | 1.227 / 0.176 |
| `dispersion_isotonic` | **0.804** / 0.132 | **1.410** / 0.124 |
| `participation_isotonic` | 0.787 / 0.131 | 1.349 / 0.142 |
| `blend` | 0.779 / 0.224 | 1.443 / 0.114 |

Every signal's ordinal distance got worse; macro-F1 stayed roughly flat or dropped depending on
signal (another instance of the two metrics disagreeing on magnitude, though not on direction
here). The one number that matters most for this experiment's central question:
**`dispersion_isotonic` — the exact signal `exp_9`'s ARD mechanism was built to rescue — regressed
from `0.804` (23-col ARD, closely matching `exp_6`'s original 19-col `0.776`) to `1.410`** at
48 columns. That's a **+0.606** regression, larger than the **+0.309** regression a scalar
bandwidth produced going from 19→23 columns in `exp_8` (the finding that motivated building ARD in
the first place). Read together with `exp_9`'s successful rescue at 23 columns, the honest
conclusion is: **ARD's per-dimension bandwidth has a real but bounded capacity** — it closed the
dilution gap once, cleanly, at the scale it was built and tested for, and that capacity appears to
run out well before 48 dimensions at N=91. This is a genuinely new, concrete finding this
experiment adds to the project's ARD story, not previously known from `exp_9` alone.

No confidence condition beats the incumbent or baseline, same as every KDM confidence attempt since
`exp_6`.

## 5. Weights: uniformly worse than `exp_9`, no baseline win

| Condition | `exp_9` 23-col ord.err / macro-F1 | `exp_10` 48-col ord.err / macro-F1 |
|---|---|---|
| `occlusion` | 0.459 / 0.259 | **0.537** / 0.251 |
| `kernel_distance` | 0.491 / 0.208 | 0.581 / 0.199 |
| `blend` | 0.637 / 0.248 | 0.848 / 0.212 |

`occlusion` — the only mechanism that ever narrowly beat baseline, back in `exp_6`/`exp_8` under
the scalar backbone (0.405/0.412 vs. 0.413) — stays clearly above baseline here too (0.537),
continuing the pattern `exp_9` already showed: ARD (at any frame width tried so far) has not
recovered that narrow scalar-backbone win.

**Per-factor** (occlusion): `dre` (0.256) and `age` (0.391) remain the strongest factors, same
ranking as every prior experiment. `psa` is notably weak here (ord.err=0.777, worst of the 9
factors) — consistent with §7's importance-comparison finding that ARD assigns `psa` very low
relevance on this frame specifically, likely because the retained PSA-trend family (`psa_tr_min`,
`_mean`, `_delta`, `_slope`, `_count`, `_first_val`) gives the model several alternative,
less-diluted routes to the same underlying signal, reducing how much any single occlusion of
`cli_psa` alone perturbs the model.

## 6. Reveal: first-ever loss to baseline

| Metric | `exp_9` 23-col ARD | `exp_10` 48-col ARD | Naive baseline |
|---|---|---|---|
| Set-precision | 0.823 | **0.718** | 0.783 |
| Macro-F1 | 0.599 | 0.572 | — |

This is the first time any KDM reveal condition has scored *below* the naive "always predict the
mode pattern" baseline (0.783) since reveal-sequence was first attempted for this backbone in
`exp_8`. Per-section:

| Section | `exp_9` 23-col F1 | `exp_10` 48-col F1 |
|---|---|---|
| `previous_notes` | 0.772 | **0.548** |
| `psa_trend` | 0.769 | 0.706 |
| `radiology_report` | 0.521 | **0.566** (improved) |
| `laboratory_results` | 0.334 | **0.468** (improved) |

Mixed at the section level: `previous_notes` (this frame's largest catch-all group, 28 of 48
columns — see `run_reveal_fullschema.py`'s `SECTION_FEATURE_GROUPS`) and `psa_trend` both got
worse, plausibly because occlusion of such a large, heterogeneous column group produces a noisier
entropy-delta signal for the downstream univariate classifier to learn from. `radiology_report` and
`laboratory_results` both *improved* — the added `lab_creatinine_mg_dl`/`lab_free_psa_ng_ml`/
`lab_free_total_ratio` columns genuinely helped `laboratory_results` specifically, a plausible,
targeted win inside an overall regression.

## 7. Importance comparison: same 2/5 agreement pattern as `exp_9`, `psa` still ranks last

| Frame | Top-5 ARD relevance | Agreement with `exp_5`'s solved set |
|---|---|---|
| `exp_9`, 19-col | age, vol, psad, pirads, cspca | 2/5 |
| `exp_9`, 23-col | age, vol, pirads, cspca, psad | 2/5 |
| `exp_10`, 48-col | vol, comorbidity, cspca, age, pirads | **2/5** |

A third frame, a third 2/5 result — this now looks like a stable property of ARD's relevance signal
relative to `exp_5`'s SVM-based solved/unsolved split, not noise specific to any one frame size.
`age` and `pirads` are the two consistent points of agreement across all three frames tried so far.
`psa` ranks **last of 9** factors here (`1/σⱼ=1.014`, the weakest of any factor on any frame tried
to date) — the clearest instance yet of the pattern first seen in `exp_9`: ARD consistently
under-ranks `psa`'s relevance despite it being one of `exp_5`'s most confidently solved factors.
With the full PSA-trend family now in the frame, `cli_psa` itself has several close substitutes
(`psav`, `psap`, `psad`, `psa_tr_delta`, `psa_tr_mean` all cluster in the same weak-relevance band,
`σⱼ` 0.88–0.99) — plausibly ARD is correctly recognizing that no single PSA-family column is
individually load-bearing once redundant alternatives exist, which is a defensible, even correct,
reading of the *frame's* redundancy, but is a different question from whether "PSA-ness" as a
concept matters to the decision (which `exp_5`'s per-factor SVM measured on a frame with no such
redundant alternatives).

## 8. Interpretation

The central finding is no longer an open reversal — §3b resolved it. Held alongside `exp_9`'s ARD
story, a coherent picture emerges: **ARD-KDM has a real, bounded ability to absorb added dimensions
without the shared-bandwidth dilution failure mode** — it worked cleanly at 23 columns (`exp_9`,
now doubly confirmed by §3b's LOO/repeated-holdout check), and by 48 columns that mechanism clearly
runs out. §4's `dispersion_isotonic` regression, confidence/weights/reveal's uniform CV regression,
*and* decision's own true performance (~0.51–0.55 across CV, LOO, and repeated-holdout, once the
single lucky split is set aside) all now point the same direction: **the 48-column full-schema
frame is worse than `exp_9`'s 23-column frame, not better, on every subtask and every verification
method except one single, since-explained held-out draw.** The held-out check did exactly the job
it exists for — surfacing a result that needed a second look before being trusted — and the
follow-up verification the user requested closed the loop within the same report.

## 9. Recommendation

- **Do not adopt the 48-column full-schema frame.** This is no longer a "CV vs. held-out, pick a
  side" call — §3b's independent LOO and repeated-holdout checks, and §3c's AUROC/Brier score,
  all agree with CV, not with the single 0.708 split. The full-schema frame underperforms `exp_9`'s
  23-column reference on every subtask and by every measure now available.
- **Report AUROC and Brier score alongside macro-F1 for decision going forward** — §3c showed they
  would have flagged `exp_10`'s misleading held-out split immediately, without needing the
  LOO/repeated-holdout follow-up at all, since they grade the full predicted probability rather
  than an argmax threshold. Cheap to add (both come from probabilities already computed during
  every CV/held-out fit) and clearly worth it.
- **`exp_9`'s 23-column ARD frame remains this project's best-validated configuration**, and is now
  even more strongly confirmed: its own LOO (0.639) and repeated-holdout (0.670 ± 0.064) numbers
  cluster tightly with its original CV (0.608) and held-out (0.680) numbers — four independent
  estimates in a 0.072 range, the tightest agreement any KDM decision result has shown in this
  project.
- **ARD's dilution-rescue capacity has a ceiling somewhere between 23 and 48 columns at N=91** — a
  concrete, useful finding for any future frame-expansion attempt in this project, and a natural
  narrower follow-up (e.g. a frame in the 30–35 column range) if further expansion is wanted.
- **Standing methodology lesson, worth carrying into every future experiment**: a single 19-case
  held-out split has demonstrated, concretely, a standard deviation of 0.134 across repeated draws
  — enough to manufacture an apparent "best result ever" from a frame that is actually worse than
  the existing reference. Any future single-split held-out result this striking should get the same
  LOO/repeated-holdout treatment before being reported as a finding, not just noted as a caveat.
- **Do not replace `confidence_svm` or `weights_svm` with any KDM mechanism** — unchanged from
  every prior KDM experiment's recommendation.
- `Frame.md`'s preprocessing convention (MinMax + one-hot + explicit flags) was not itself isolated
  as a variable in this experiment (§1's deliberate scope decision) — a natural `exp_11` ablation
  would hold the frame fixed and vary only preprocessing convention, to separate "does the wider
  frame help" from "does this preprocessing convention help," which this experiment could not
  disentangle.
