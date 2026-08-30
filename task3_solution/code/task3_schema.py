"""Own Task3Output. Originally ported from external/chimera-agent-baseline/src/chimera_agent_
baseline/output/schema.py's pydantic model, which omits `event` entirely -- that turned out to be
an INCOMPLETE reference: it's used only as an internal constraint on the reference agent's own LLM
form-fill step, not the actual submission contract. Two independent sources both confirm `event`
IS required: (1) the org's own task-3 webpage ("Predicted BCR status within the observation window
(1 = BCR; 0 = no BCR)" listed as a required output alongside the time and reasoning), and (2) the
reference agent's real output-writing code (run.py::_write_task_outputs), which writes
`{"event": ..., "months_to_recurrence": ...}` into the actual submitted file -- the pydantic model
in schema.py is never used to validate that final write. Corrected here; see experiments/exp_7/.
"""

from pydantic import BaseModel, ConfigDict, Field


class Task3Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: int = 3
    months_to_recurrence: float = Field(ge=0.0)
    event: int = Field(ge=0, le=1)
    reasoning: str = Field(min_length=40)
