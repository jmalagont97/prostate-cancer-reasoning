from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictInt, field_validator


class Task3Outcome(BaseModel):
    """Official Task 3 outcome file: only evaluator-facing fields."""

    model_config = ConfigDict(extra="forbid")
    months_to_recurrence: float = Field(ge=0)
    event: StrictInt = Field(ge=0, le=1)

    @field_validator("event")
    @classmethod
    def validate_event(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("event must be 0 or 1")
        return value


class Task3Reasoning(RootModel[str]):
    """Official Task 3 reasoning file is a JSON string."""

    pass
