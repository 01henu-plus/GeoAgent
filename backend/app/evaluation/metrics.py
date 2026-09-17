from pydantic import BaseModel, Field


class EvaluationSummary(BaseModel):
    total: int = Field(default=0, ge=0)
    succeeded: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    recovery_cases_passed: int = Field(default=0, ge=0)
    recovery_cases: int = Field(default=0, ge=0)
    tool_selection_cases: int = Field(default=0, ge=0)
    tool_selection_passed: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    execution_time_ms: float = Field(default=0.0, ge=0.0)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0
