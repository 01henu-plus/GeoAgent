"""上一轮运行/结果等引用的最小解析器。"""

from app.core.models import Run
from app.state import StateStore

_LATEST_REFERENCES = {"last", "latest", "刚才", "刚才的结果", "上一轮", "上一次", "上一张图"}


class ReferenceResolver:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def resolve_runs(
        self,
        identifiers: list[str],
        *,
        conversation_id: str | None = None,
        exclude_run_id: str | None = None,
    ) -> list[Run]:
        result: list[Run] = []
        for identifier in identifiers:
            if identifier.casefold() in _LATEST_REFERENCES:
                runs = self.store.list_runs()
                if exclude_run_id:
                    runs = [item for item in runs if item.id != exclude_run_id]
                if conversation_id:
                    runs = [item for item in runs if item.conversation_id == conversation_id]
                if runs:
                    result.append(runs[0])
                continue
            run = self.store.get_run(identifier)
            if run and run.id != exclude_run_id:
                result.append(run)
        return result
