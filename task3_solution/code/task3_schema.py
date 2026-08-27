"""Own Task3Output, ported from external/chimera-agent-baseline/src/chimera_agent_baseline/output/
schema.py (confirmed by reading source, not assumed): case_id, task=3, months_to_recurrence (>=0),
reasoning (>=40 chars). No confidence/variable_weights/reveal_sequence -- task3 has none of task1's
per-case tool/eligibility machinery. `extra="forbid"` mirrors the source exactly: an "event" field
would be REJECTED by this schema even though evaluate.py's diagnostic (non-ranking) case_score
optionally reads one if present -- confirmed in evaluate.py's process_interf2, but not part of
Task3Output's actual contract, so deliberately not added here.
"""

from pydantic import BaseModel, ConfigDict, Field


class Task3Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: int = 3
    months_to_recurrence: float = Field(ge=0.0)
    reasoning: str = Field(min_length=40)
