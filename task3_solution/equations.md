# Model derivation: `cox_p03_prostatectomy_pca5_p90`

Every equation below maps directly to code in `code/task3_config.py` and `code/task3_embeddings.py`.
Rendered with GitHub's native LaTeX support.

## 1. Inputs

Two raw sources feed the model per case:

- **13 tabular/pathology-text fields** — 5 from `structured-prompt.json` (parsed), 8 regex-extracted
  from the surgical pathology report in `…clinical-data.json`. See the README's input table.
- **A bag of prostatectomy-slide embeddings** — 1 to 3 vectors per case, each in
  $\mathbb{R}^{960}$, from `…neural-representations.json`, produced by a frozen foundation model
  over the whole-slide image.

## 2. From a bag of slides to one vector

**Masked mean pooling.** For case $i$ with $n_i \in \{1,2,3\}$ valid slide vectors
$e_{i,1},\dots,e_{i,n_i} \in \mathbb{R}^{960}$:

$$\bar{e}_i = \frac{1}{n_i}\sum_{k=1}^{n_i} e_{i,k} \tag{1}$$

*Why:* the bag has no fixed order or count. Masked mean pooling is the simplest permutation-invariant
aggregator, with zero trainable parameters — appropriate at $n=75$, where a learned (e.g.
attention-weighted) pool would have nowhere near enough cases to fit safely.

**Dimensionality reduction (PCA).** Let $X \in \mathbb{R}^{74 \times 960}$ be the pooled vectors of
the *current LOOCV training fold only*, mean-centered as $\tilde X = X - \mu$. Its SVD
$\tilde X = U\Sigma V^{\mathsf T}$ gives the top-5 principal directions $V_5 \in \mathbb{R}^{960\times5}$.
Every case — train or held-out — is then projected through that same, already-fitted $V_5$:

$$z_i = V_5^{\mathsf T}(\bar{e}_i - \mu) \in \mathbb{R}^5 \tag{2}$$

*Why 5, and why refit every fold:* 960 raw dimensions against 74 training cases is a severe
curse-of-dimensionality risk. $k=5$ is not a default — 2, 5, and 10 were each swept under full
leave-one-out validation (0.845 / **0.858** / 0.847 C-index) and 5 won. The SVD is re-computed
inside every one of the 75 LOOCV training folds, never on the case being predicted; doing it once
on all 75 cases up front is exactly the leakage class that collapsed an earlier experiment's result
from 0.724 to 0.363 once corrected (see `experiments/exp_3_pathology_text_extraction/`).

## 3. The covariate vector

$$x_i = [\,x_{i,1},\dots,x_{i,13},\,z_{i,1},\dots,z_{i,5}\,] \in \mathbb{R}^{18} \tag{3}$$

One field considered and deliberately excluded: `active_treatment_flag` was 1.0 for exactly one of
75 cases. As a fixed numeric column it singularizes the Cox fit in the one LOOCV fold that holds
that case out — near-zero signal at a real convergence cost.

## 4. The model

A Cox proportional-hazards model — semi-parametric: it estimates how covariates scale risk without
assuming a shape for how risk evolves over time.

$$h(t \mid x) = h_0(t)\cdot e^{\beta \cdot x} \tag{4}$$

| Term | Meaning |
|---|---|
| $h(t\mid x)$ | Instantaneous recurrence risk at time $t$, given covariates $x$ |
| $h_0(t)$ | Baseline hazard — shared by every case, left entirely unspecified |
| $e^{\beta \cdot x}$ | The case's risk *relative* to baseline |
| $\beta$ | The 18 fitted coefficients |

*Why Cox, over the alternatives actually tried:* three parametric AFT families (Weibull, LogNormal,
LogLogistic — each assumes a specific closed-form survival-time shape) were fit on the identical
$x_i$ and lost: best AFT scored 0.755 against Cox's 0.858. Random Survival Forest / Gradient-Boosted
Cox were evaluated for install and are unavailable (no C++ compiler toolchain in the source
environment).

## 5. Fitting β

$h_0(t)$ cancels out of a ratio of hazards — this is what lets Cox's *partial* likelihood estimate
$\beta$ without ever specifying it. With $R(t)$ the risk set still under observation at time $t$,
and Efron's correction distributing tied event times (7 of this cohort's 19 events share a month
with another):

$$L(\beta) = \prod_{i \,\in\, \text{events}} \frac{e^{\beta \cdot x_i}}{\sum_{j \,\in\, R(t_i)} e^{\beta \cdot x_j}} \tag{5}$$

18 covariates against 74 training cases (of which only 19 are ever events) is thin enough that an
unpenalized fit would chase noise. A ridge (L2) penalty shrinks it instead:

$$\beta = \arg\max_\beta \Big[\, \log L(\beta) \;-\; \lambda \cdot \tfrac{1}{2}\lVert\beta\rVert^2 \,\Big] \tag{6}$$

Solved by Newton–Raphson (`lifelines`' own optimizer, unmodified).

**Why $\lambda = 0.3$, precisely** — not a library default. Swept 0.01 → 2.0 in a full leave-one-out
grid; the curve is a clean, single-peaked function of $\lambda$, not a noisy scatter:

| λ | 0.01 | 0.1 | **0.3** | 0.5 | 0.7 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| C-index* | 0.704 | 0.749 | **0.766** | 0.758 | 0.754 | 0.751 | 0.736 |

*swept on the pre-embedding feature set; $\lambda=0.3$ carried forward unchanged once embeddings
were added, and still tops that feature set at 0.858.

## 6. From a fitted hazard to a date

$\beta$ alone gives a *relative* risk, not a date. The non-parametric Breslow estimator first
recovers the cumulative baseline hazard the partial likelihood deliberately never specified:

$$H_0(t) = \sum_{t_j \le t} \frac{d_j}{\sum_{l \,\in\, R(t_j)} e^{\beta \cdot x_l}} \tag{7}$$

giving every case its own predicted survival curve,

$$S(t \mid x) = \exp\big(-H_0(t)\cdot e^{\beta \cdot x}\big) \tag{8}$$

A first, natural choice for the output is that curve's expectation (mean survival time):

$$E[T \mid x] = \int_0^{\infty} S(t \mid x)\, dt \tag{9}$$

**Why not the median, and why (9) alone isn't what ships:** the median survival time requires
$S(t\mid x)$ to cross 50% *within the observed follow-up window* — it never does for 62 of 75 cases
here, silently collapsing them all to the same fallback value. Equation (9)'s expectation is always
finite and was the model's first shipped choice (LOOCV C-index 0.8575) — but a systematic
calibration check across all 19 known-truth (event=1) cases found it severely biased: **+40.1
month** mean overestimate, up to 11× on cases that recurred within a year. With 75% of this cohort
censored, the mean gets dragged far right by a long, poorly-supported tail beyond the last real
observation — a known property of the mean vs. the median under heavy censoring.

The fix, and what actually ships, is a shorter, better-defined quantile of the same curve — the
time by which a fraction $p$ of the population would already have recurred:

$$\text{months\_to\_recurrence} = \min\{\, t : S(t \mid x) \le p \,\}, \quad p = 0.90 \tag{10}$$

`lifelines.CoxPHFitter.predict_percentile(p=0.90)` — the earliest time at which the survival curve
has dropped to 0.90 (10% of the population would have recurred by then). Swept $p \in
\{0.5,\dots,0.95\}$; $p=0.90$ wins on **both** axes: LOOCV C-index $0.8575 \to$ **0.8765**, and the
event=1 mean bias collapses from $+40.1$ to **$-9.7$ months** (MAE $42.2 \to 10.0$). Bootstrap on the
C-index gain: 95% CI $[-0.002, +0.046]$ — borderline, unlike equation (2)'s embedding-source result,
but never shown worse. Results remain capped at 330 months (3× the longest observed follow-up) as a
last-resort guard against numerical blow-up on an extremely low-risk case. Full sweep and
methodology: `experiments/exp_6_calibration_check/`.

## 7. Output

`Task3Output`: `case_id`, `months_to_recurrence` (equation 10, clipped to [0, 330]), `event`
(equation 11, below), `reasoning` (template text citing $x_6$–$x_{13}$'s findings). **Correction**:
an earlier version of this document said no `event` field was required — that was wrong, traced to
treating the reference baseline's `Task3Output` pydantic model as the full contract when it only
ever validated an internal LLM form-fill step. The org's task-3 webpage and the reference agent's
real output-writing code (`run.py`) both confirm `event` is required. Only `months_to_recurrence`
is scored (`evaluate.py`'s `concordance_index`, against ground-truth time and event) — `event` is
required for a complete submission but doesn't change that metric. See
`experiments/exp_7_output_contract_fix/`.

**A real example** (case T3-001, true recurrence at 1.8 months): equation (9) predicted 34.0 months;
equation (10) predicts 1.6 months.

### Equation (11) — deriving `event` from the same survival curve

No second model. A fixed horizon $h$ and threshold $\tau$ turn the already-fitted $S(t \mid x)$
(equation 6) into a binary call:

$$
\hat{e}(x) = \mathbb{1}\left[ S(h \mid x) < \tau \right]
$$

Both swept under LOOCV against real ground-truth labels (not assumed): $h=60$ months (the standard
clinical 5-year BCR benchmark) and $\tau=0.7$ won, balanced accuracy 0.779 against a trivial
"always predict 0" baseline's useless 0.500. Full sweep grid and the classification
report/confusion matrix: `experiments/exp_7_output_contract_fix/`.
