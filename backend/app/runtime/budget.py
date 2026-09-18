"""Run 预算。"""

from datetime import UTC, datetime

from app.core.models import Run, RunBudget


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, budget: RunBudget) -> None:
        self.budget = budget

    def check_turn(self, run: Run) -> None:
        if run.turn_count >= self.budget.max_agent_turns:
            raise BudgetExceeded("Agent turn budget exceeded")

    def check_tool(self, run: Run) -> None:
        if run.tool_call_count >= self.budget.max_tool_calls:
            raise BudgetExceeded("Tool call budget exceeded")

    def check_subagents(self, count: int) -> None:
        if count > self.budget.max_subagents:
            raise BudgetExceeded("SubAgent budget exceeded")

    def check_replan(self, run: Run) -> None:
        if run.replan_count >= self.budget.max_replans:
            raise BudgetExceeded("REPLAN_BUDGET_EXCEEDED")

    def check_execution_time(self, run: Run) -> None:
        if run.started_at is None:
            return
        elapsed = (datetime.now(UTC) - run.started_at).total_seconds()
        if elapsed >= self.budget.max_execution_seconds:
            raise BudgetExceeded("Run execution time budget exceeded")
