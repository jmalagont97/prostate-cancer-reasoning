#!/usr/bin/env python3
"""Update hidden logbooks with exp_15 entries."""

from pathlib import Path

logbook = '''
## [2026-09-06] - Cascaded 2.1 -> 2.2 Treatment-Informed Confidence Completed (`exp_15`)
- **Current Project State:** Cascade experiment injecting exp_11 fuzzy-SVR treatment output into 2.2 confidence model under nested 5-fold OOF. Initial run had critical preprocessing leakage; v2 rewritten leak-free.
- **Specific Advancements:** 4 conditions (C0 none, C1 probs+4, C2 argmax OH+4, C3 both+8) x 2 targets (3-class, binary), Soft-SVR tabular grid, LOOCV leak-free per-fold. Audited and fixed 2 critical bugs: (1) StandardScaler/TF-IDF global pre-fit leaked into LOOCV -> v2 fits inside each fold; (2) eval() replaced by numeric C/eps.
- **Key Results:** Cascade NO meaningful improvement: MCCV-100 Macro-F1 deltas vs C0 were Delta~-0.002 (C1), -0.001 (C2), +0.002 (C3) on both targets; LOOCV accuracy uniform +0.014 (78.6->80.0%, one extra correct clear). H2 (probs>argmax) rejected: C1 <= C2. Conclusion: treatment (2.1) and confidence (2.2) signals largely orthogonal under tabular framework. C3 pickle persisted, loadable.
- **Next Tasks:** Attack 2.2 via multimodal fusion (MRI/text) instead of 2.1 injection; oracle-cascade test to separate signal vs model capacity; feature engineering for the confidence contract.
'''

discussion = '''
### [2026-09-06] - Cascaded Treatment-Informed Confidence (`exp_15`)
- **Principal Investigator Proposal (User):** Use subtask 2.1 output as input for subtask 2.2 (treatment-informed confidence) with complete confusion matrices per model.
- **Co-Investigator Analysis (AI):** Designed nested OOF cascade (leak-free). Flagged critical preprocessing-leakage bug in v1 and insecure eval(); rewrote v2. Tested 3 injection variants x 2 targets with LOOCV.
- **Consensus:** Executed exp_15 v2 (555s). Result: cascade provides no statistically meaningful improvement (deltas 30x below split sigma; LOOCV +1 case). H2 rejected (probabilities don't beat argmax). Concluded treatment and confidence are largely orthogonal signals. Persisted C3 pickle. Next: multimodal attack on 2.2 or oracle cascade.
'''

p1 = Path('.logbook.md')
with open(p1, 'a', encoding='utf-8') as f:
    f.write(logbook)
print('logbook.md updated (exp_15).')

p2 = Path('.discussion.md')
with open(p2, 'a', encoding='utf-8') as f2:
    f2.write(discussion)
print('discussion.md updated (exp_15).')
