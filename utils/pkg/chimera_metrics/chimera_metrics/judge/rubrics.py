"""Clinical rationale evaluation rubrics for LLM-as-a-Judge."""

DECISION_RATIONALE_RUBRIC = (
    "Score the agent's free-text rationale (Actual Output) against the "
    "pathologist's rationale (Expected Output) and the case's expected clinical "
    "decision. Score HIGH if the rationale: (1) supports the same "
    "decision; (2) cites the same important/decisive clinical "
    "variables; (3) does not contradict the clinical_data; (4) does not "
    "invent unavailable information; (5) expresses uncertainty consistent "
    "with the stated confidence. Score LOW if it contradicts the decision, "
    "misses major decisive factors, hallucinates clinical facts, or gives "
    "generic case-agnostic reasoning."
)

RECURRENCE_RATIONALE_RUBRIC = (
    "Score the agent's free-text rationale (Actual Output) for a prostate "
    "biochemical-recurrence (time-to-event) prediction. Use the clinical "
    "inputs and reference outcome given in Input and the agent's own "
    "predicted event/time in Actual Output. Score HIGH if the rationale: "
    "(1) is consistent with its own predicted recurrence event and timing; "
    "(2) cites concrete post-operative prognostic features actually present "
    "in the clinical inputs (e.g. Gleason/ISUP grade, pathological stage, "
    "surgical margin status, seminal-vesicle invasion, extraprostatic "
    "extension, lymph-node status, PSA / PSA density); (3) does not "
    "contradict the clinical inputs; (4) does not invent unavailable "
    "information. Score LOW if it contradicts the inputs, ignores major "
    "prognostic factors, hallucinates facts, or gives generic case-agnostic "
    "reasoning. There is no reference free-text rationale, so judge clinical "
    "soundness and internal consistency, not verbatim agreement."
)
