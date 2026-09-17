"""Task 级 WorkingMemory 的加载、合并和工具结果更新。"""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.models import (
    RequestFrame,
    RequestResources,
    ToolResult,
    ToolStatus,
    WorkingMemory,
    WorkingMemoryItem,
)

from .store import StateStore


class WorkingMemoryUpdater:
    """集中处理工作状态，避免把字段更新分散到各个 GIS Tool。"""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def load_or_create(self, task_id: str, conversation_id: str | None = None) -> WorkingMemory:
        memory = self.store.get_working_memory(task_id)
        if memory is not None:
            return memory
        memory = WorkingMemory(task_id=task_id, conversation_id=conversation_id)
        self.store.save_working_memory(memory)
        return memory

    def apply_request(
        self,
        memory: WorkingMemory,
        frame: RequestFrame,
        resources: RequestResources | None = None,
    ) -> WorkingMemory:
        dataset_ids = list(memory.active_dataset_ids)
        artifact_ids = list(memory.active_artifact_ids)
        for dataset in resources.datasets if resources else []:
            _append_unique(dataset_ids, dataset.id)
        for reference in frame.references:
            if not reference.target_id:
                continue
            if reference.type == "dataset":
                _append_unique(dataset_ids, reference.target_id)
            elif reference.type == "artifact":
                _append_unique(artifact_ids, reference.target_id)

        constraints = list(memory.constraints)
        for constraint in frame.constraints:
            _append_text_unique(constraints, constraint)
        return memory.model_copy(
            update={
                "active_dataset_ids": dataset_ids,
                "active_artifact_ids": artifact_ids,
                "constraints": constraints,
                "updated_at": _now(),
            }
        )

    def update_from_tool_result(self, task_id: str | None, result: ToolResult, *, run_id: str | None) -> WorkingMemory | None:
        if not task_id:
            return None
        memory = self.store.get_working_memory(task_id)
        if memory is None:
            return None

        dataset_ids = list(memory.active_dataset_ids)
        artifact_ids = list(memory.active_artifact_ids)
        for dataset_id in result.datasets:
            _append_unique(dataset_ids, dataset_id)
        for artifact_id in result.artifacts:
            _append_unique(artifact_ids, artifact_id)

        intermediate = list(memory.intermediate_results)
        if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} or result.error is not None:
            if not any(item.reference_id == result.call_id for item in intermediate):
                intermediate.append(
                    WorkingMemoryItem(
                        kind="tool_result",
                        reference_id=result.call_id,
                        summary=_result_summary(result),
                        source_run_id=run_id,
                    )
                )

        unresolved = list(memory.unresolved_questions)
        if result.status in {ToolStatus.BLOCKED, ToolStatus.FAILED} and result.error is not None:
            if result.error.category.value in {"INPUT", "DATA", "CRS"}:
                _append_text_unique(unresolved, f"需要补充工具输入：{result.error.code}")

        updated = memory.model_copy(
            update={
                "active_dataset_ids": dataset_ids,
                "active_artifact_ids": artifact_ids,
                "intermediate_results": intermediate,
                "unresolved_questions": unresolved,
                "updated_at": _now(),
            }
        )
        self.store.save_working_memory(updated)
        return updated


def _result_summary(result: ToolResult) -> str:
    if result.datasets:
        return f"工具完成，涉及数据集：{', '.join(result.datasets)}。"
    if result.artifacts:
        return f"工具完成，生成产物：{', '.join(result.artifacts)}。"
    if result.error is not None:
        return f"工具未完成：{result.error.code}。"
    if isinstance(result.output, dict):
        keys = list(result.output)[:6]
        return f"工具完成，返回结构化结果字段：{', '.join(str(key) for key in keys)}。"
    return "工具执行完成。"


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _append_text_unique(values: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized.casefold() not in {item.casefold() for item in values}:
        values.append(normalized)


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = ["WorkingMemoryUpdater"]
