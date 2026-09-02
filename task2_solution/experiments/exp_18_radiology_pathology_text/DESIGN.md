# exp_18 Design: Radiology + Pathology Free-Text Staging Features

## Background

`structured.csv` already carries the core biopsy/MRI numbers (`bx_isup`, `bx_gl_prim`, `bx_gl_sec`,
`pirads`, `psad`, `cspca`, `ct`) directly, but `radiology_report`/`pathology_report`'s free text has
real, unextracted staging detail — surfaced by direct inspection at the user's prompt, with a strong
precedent (task3's exp_3 anchored its biggest early win on exactly this feature category:
extracapsular extension / seminal vesicle invasion, extracted from free-text pathology).

## Extraction — corpus-scan-first, same discipline as task3's exp_3

Unlike task3's tightly templated surgical-pathology sentences, this corpus is real free prose —
several regex iterations were needed, each caught by direct corpus scanning before being trusted
(documented in `scripts/extract_features_task2.py`'s docstrings):

- A naive first pass on **EPE** matched "clinical suspicion for extracapsular extension" inside the
  scan's *indication* line (why the MRI was ordered) as if it were a finding — fixed by restricting
  the search to the FINDINGS/IMPRESSION sections only.
- **Cribriform**'s negation is purely prefix-style ("No cribriform..."), confirmed by scan.
- **Perineural invasion**'s negation/affirmation is mixed prefix *and* suffix
  ("No perineural invasion" vs. "Perineural invasion is absent" vs. "...is identified") — a pattern
  task3's pathology text never needed, caught only by scanning every distinct phrasing.

Final features, coverage from the confirmed scan:

| Feature | Source | Coverage | Positive rate |
|---|---|---|---|
| `epe_present` | radiology | 134/153 (87.6%) | 5/134 |
| `svi_present` | radiology | 130/153 (85.0%) | 2/130 |
| `ai_isup_discordance` | pathology | 147/153 (96.1%) | mean +1.73, AI systematically grades higher |
| `cribriform_present` | pathology | 147/153 (96.1%) | 51/147 |
| `perineural_invasion_present` | pathology | 95/153 (62.1%) | 25/95 |

`lymphadenopathy` was scanned and dropped before extraction: 0 positive cases across the entire
153-case corpus — a zero-variance feature, confirmed useless before it was ever coded.

## Hypothesis

`epe_present`/`svi_present` (radiology) and `ai_isup_discordance`/`cribriform_present`/
`perineural_invasion_present` (pathology), tested as two separate groups (not combined, mirroring
exp_16/17's isolation discipline), improve `ranking_score` for at least one sub-task — stated with
real optimism given the direct task3 precedent for the radiology group specifically, though EPE/SVI's
extreme rarity (5 and 2 positive cases respectively) is a genuine risk flag for how much signal a
near-zero-variance feature can add at n=72.

## Method

Each group tested against each sub-task's own currently-promoted feature set (not just the plain
10-feature base — `decision` already has `psa_kinetics` promoted), via `task2_variant_tester.py`.
Any point-estimate winner bootstrap-confirmed before promotion, same bar as every other experiment.

## What would change the promoted config

Only a feature-group addition that beats the current promoted baseline for that sub-task on
`ranking_score`, bootstrap-confirmed.
