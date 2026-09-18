"""ContextAssembler：把持久化状态组织成有优先级的临时 Model View。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from app.core.models import (
    AgentRequest,
    ConversationMemory,
    Dataset,
    IntentResult,
    MemoryItem,
    Plan,
    RequestFrame,
    UserProfile,
    WorkingMemory,
)


class ContextPriority(IntEnum):
    REQUIRED = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(frozen=True, slots=True)
class ContextSection:
    name: str
    priority: ContextPriority
    content: Any
    required: bool = False
    compressible: bool = True
    estimated_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextAssembler:
    """只构建临时上下文，不修改任何底层 State / Memory。"""

    def __init__(self, *, recent_message_count: int = 8) -> None:
        self.recent_message_count = max(2, recent_message_count)

    def assemble_main(
        self,
        request: AgentRequest,
        datasets: list[Dataset],
        plan: Plan | None,
        memories: list[MemoryItem],
        *,
        conversation: list[dict[str, Any]] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        working_memory: WorkingMemory | dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        referenced_runs: list[dict[str, Any]] | None = None,
        findings: list[Any] | None = None,
        errors: list[str] | None = None,
        intent_hint: IntentResult | None = None,
        request_frame: RequestFrame | None = None,
        user_profile: UserProfile | dict[str, Any] | None = None,
        conversation_memory: ConversationMemory | dict[str, Any] | None = None,
        request_resources: dict[str, Any] | None = None,
        task_goal: str | None = None,
        run_state: Any = None,
        current_observation: Any = None,
        plan_progress: dict[str, Any] | None = None,
        latest_failure: dict[str, Any] | None = None,
    ) -> list[ContextSection]:
        goal = task_goal or (request_frame.goal if request_frame else request.user_input)
        working_core, working_details = _working_memory_views(working_memory)
        sections = [
            self._section(
                "user_request",
                ContextPriority.REQUIRED,
                request.user_input,
                required=True,
                metadata={"hard_min_chars": 256},
            ),
            self._section(
                "request_frame",
                ContextPriority.REQUIRED,
                _request_frame_view(request_frame, goal),
                required=True,
            ),
            self._section(
                "task_goal",
                ContextPriority.REQUIRED,
                goal,
                required=True,
                metadata={"hard_min_chars": 128},
            ),
            self._section(
                "request_resources",
                ContextPriority.REQUIRED,
                request_resources
                or {
                    "dataset_ids": list(dict.fromkeys([*request.dataset_ids, *(item.id for item in datasets)])),
                    "attachment_ids": list(dict.fromkeys(request.attachment_ids)),
                    "referenced_run_ids": list(dict.fromkeys(request.referenced_run_ids)),
                },
                required=True,
            ),
            self._section(
                "working_memory",
                ContextPriority.REQUIRED,
                working_core,
                required=True,
            ),
        ]
        if current_observation is not None:
            sections.append(
                self._section(
                    "current_observation",
                    ContextPriority.REQUIRED,
                    _observation_view(current_observation),
                    required=True,
                )
            )

        if plan is not None:
            sections.append(
                self._section(
                    "plan_state",
                    ContextPriority.HIGH,
                    _plan_state_view(plan, plan_progress),
                    metadata={"drop_rank": 65, "kind": "plan_state"},
                )
            )
        if latest_failure:
            sections.append(
                self._section(
                    "latest_failure",
                    ContextPriority.HIGH,
                    _compact_value(latest_failure),
                    metadata={"drop_rank": 25},
                )
            )

        optional = [
            self._section(
                "conversation_memory",
                ContextPriority.HIGH,
                _conversation_memory_view(conversation_memory),
                metadata={"drop_rank": 20, "kind": "conversation_memory"},
            ),
            self._section(
                "project_memory",
                ContextPriority.HIGH,
                [_memory_view(item) for item in memories],
                metadata={"drop_rank": 50, "trim_side": "right", "min_items": 1},
            ),
            self._section(
                "referenced_runs",
                ContextPriority.HIGH,
                [_run_view(item) for item in referenced_runs or []],
                metadata={"drop_rank": 60, "trim_side": "right", "min_items": 1},
            ),
            self._section(
                "datasets",
                ContextPriority.HIGH,
                [_dataset_view(item) for item in datasets],
                metadata={"drop_rank": 40, "trim_side": "right", "min_items": 1, "kind": "datasets"},
            ),
            self._section(
                "run_state",
                ContextPriority.HIGH,
                _run_view(run_state),
                metadata={"drop_rank": 10},
            ),
            self._section(
                "errors",
                ContextPriority.HIGH,
                [_clip_text(item, 500) for item in errors or [] if str(item).strip()],
                metadata={"drop_rank": 30, "trim_side": "left", "min_items": 1},
            ),
            self._section(
                "working_memory_details",
                ContextPriority.HIGH,
                working_details,
                metadata={"drop_rank": 70, "kind": "working_memory_details"},
            ),
            self._section(
                "recent_messages",
                ContextPriority.MEDIUM,
                _recent_messages(conversation or [], self.recent_message_count),
                metadata={"drop_rank": 30, "trim_side": "left", "min_items": 2},
            ),
            self._section(
                "user_profile",
                ContextPriority.MEDIUM,
                _profile_view(user_profile),
                metadata={"drop_rank": 70},
            ),
            self._section(
                "deterministic_hint",
                ContextPriority.MEDIUM,
                {
                    "intent": _dump(intent_hint),
                    "plan": _dump(plan),
                    "instruction": "仅供参考，不代表已经确认的用户意图，也不是必须执行的步骤。",
                }
                if intent_hint is not None or plan is not None
                else {},
                metadata={"drop_rank": 80},
            ),
            self._section(
                "budget",
                ContextPriority.MEDIUM,
                budget or {},
                metadata={"drop_rank": 90},
            ),
            self._section(
                "request_context",
                ContextPriority.MEDIUM,
                _compact_value(request.context),
                metadata={"drop_rank": 40},
            ),
            self._section(
                "findings",
                ContextPriority.LOW,
                [_compact_value(item) for item in findings or []],
                metadata={"drop_rank": 90, "trim_side": "left", "min_items": 0},
            ),
            self._section(
                "tool_capabilities",
                ContextPriority.LOW,
                _tool_capabilities(tool_definitions or []),
                metadata={"drop_rank": 100, "trim_side": "right", "min_items": 0},
            ),
        ]
        sections.extend(section for section in optional if not _is_empty(section.content))
        return sections

    def assemble_sub(
        self,
        request: AgentRequest,
        subtask: dict[str, Any],
        datasets: list[Dataset],
        *,
        parent_findings: list[Any] | None = None,
        working_memory: WorkingMemory | dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> list[ContextSection]:
        working_core, working_details = _working_memory_views(working_memory)
        sections = [
            self._section("user_request", ContextPriority.REQUIRED, request.user_input, required=True),
            self._section("subtask", ContextPriority.REQUIRED, _compact_value(subtask), required=True),
            self._section("working_memory", ContextPriority.REQUIRED, working_core, required=True),
            self._section(
                "datasets",
                ContextPriority.HIGH,
                [_dataset_view(item) for item in datasets],
                metadata={"drop_rank": 30, "trim_side": "right", "min_items": 1, "kind": "datasets"},
            ),
            self._section(
                "working_memory_details",
                ContextPriority.HIGH,
                working_details,
                metadata={"drop_rank": 70, "kind": "working_memory_details"},
            ),
            self._section(
                "parent_findings",
                ContextPriority.MEDIUM,
                [_compact_value(item) for item in parent_findings or []],
                metadata={"drop_rank": 60, "trim_side": "left", "min_items": 0},
            ),
            self._section("budget", ContextPriority.MEDIUM, budget or {}, metadata={"drop_rank": 80}),
            self._section(
                "allowed_tools",
                ContextPriority.MEDIUM,
                allowed_tools
                or [
                    "dataset.inspect",
                    "vector.validate",
                    "analysis.distance",
                    "raster.inspect",
                    "raster.slope",
                ],
                metadata={"drop_rank": 50, "trim_side": "right", "min_items": 1},
            ),
        ]
        return [section for section in sections if section.required or not _is_empty(section.content)]

    @staticmethod
    def _section(
        name: str,
        priority: ContextPriority,
        content: Any,
        *,
        required: bool = False,
        compressible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ContextSection:
        return ContextSection(
            name=name,
            priority=priority,
            content=content,
            required=required,
            compressible=compressible,
            estimated_tokens=estimate_tokens(content),
            metadata=metadata or {},
        )


def estimate_tokens(value: Any) -> int:
    """无 provider tokenizer 时的保守 Unicode-aware token 估算器。"""

    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    pieces = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u3400-\u9fff]", text)
    total = 0
    for piece in pieces:
        if len(piece) == 1 and "\u3400" <= piece <= "\u9fff":
            total += 1
        elif re.fullmatch(r"[A-Za-z0-9_]+", piece):
            total += max(1, math.ceil(len(piece) / 4))
        else:
            total += 1
    return max(1, total)


def _request_frame_view(frame: RequestFrame | None, goal: str) -> dict[str, Any]:
    if frame is None:
        return {"goal": goal}
    return {
        "mode": frame.mode.value,
        "goal": frame.goal,
        "references": [
            {
                "mention": item.mention,
                "type": item.type,
                "target_id": item.target_id,
                "label": item.label,
                "confidence": item.confidence,
            }
            for item in frame.references
        ],
        "constraints": list(frame.constraints),
        "capabilities": list(frame.capabilities),
        "target_task_id": frame.target_task_id,
        "target_run_id": frame.target_run_id,
        "needs_planning": frame.needs_planning,
        "needs_tool": frame.needs_tool,
        "unresolved_references": list(frame.unresolved_references),
        "resolution_status": frame.resolution_status.value,
        "blocking_issues": list(frame.blocking_issues),
    }


def _plan_state_view(plan: Plan, progress: dict[str, Any] | None) -> dict[str, Any]:
    progress = progress or {}
    completed = set(progress.get("completed_steps") or [])
    steps = []
    for step in plan.steps:
        steps.append(
            {
                "id": step.id,
                "title": step.title,
                "action": step.action,
                "tool_name": step.tool_name,
                "depends_on": list(step.depends_on),
                "status": "SUCCEEDED" if step.id in completed else step.status.value,
            }
        )
    return {
        "id": plan.id,
        "goal": plan.goal,
        "intent": plan.intent.value,
        "revision": plan.revision,
        "clarification": plan.clarification,
        "steps": steps,
        "completed_steps": sorted(completed),
        "step_outputs": _compact_value(progress.get("step_outputs") or {}),
    }


def _working_memory_views(value: WorkingMemory | dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _dump(value)
    if not raw:
        return {}, {}
    core = {
        "task_id": raw.get("task_id"),
        "conversation_id": raw.get("conversation_id"),
        "active_dataset_ids": list(raw.get("active_dataset_ids") or []),
        "active_artifact_ids": list(raw.get("active_artifact_ids") or []),
        "constraints": [_clip_text(item, 400) for item in raw.get("constraints") or []],
        "unresolved_questions": [_clip_text(item, 500) for item in raw.get("unresolved_questions") or []],
    }
    details = {
        "assumptions": [_clip_text(item, 400) for item in raw.get("assumptions") or []],
        "intermediate_results": [
            {
                "kind": item.get("kind"),
                "reference_id": item.get("reference_id"),
                "summary": _clip_text(item.get("summary", ""), 500),
                "source_run_id": item.get("source_run_id"),
            }
            for item in raw.get("intermediate_results") or []
            if isinstance(item, dict)
        ][-8:],
    }
    return core, details


def _dataset_view(dataset: Dataset) -> dict[str, Any]:
    schema = dataset.schema
    schema_view: dict[str, Any] | None = None
    if schema is not None:
        fields = list(schema.fields.items())[:32]
        schema_view = {
            "fields": dict(fields),
            "geometry_type": schema.geometry_type,
            "feature_count": schema.feature_count,
            "width": schema.width,
            "height": schema.height,
            "bands": schema.bands,
            "resolution": schema.resolution,
            "nodata": schema.nodata,
            "invalid_geometry_count": schema.invalid_geometry_count,
        }
        schema_view = {key: value for key, value in schema_view.items() if value not in (None, {}, [])}
    metadata_keys = (
        "driver",
        "layer",
        "encoding",
        "geometry_column",
        "dtype",
        "band_count",
        "resolution",
        "nodata",
        "units",
        "time_field",
        "id_field",
    )
    metadata = {key: _compact_value(dataset.metadata[key]) for key in metadata_keys if key in dataset.metadata}
    result = {
        "id": dataset.id,
        "name": dataset.name,
        "kind": dataset.kind.value,
        "format": dataset.format,
        "crs": _dump(dataset.crs),
        "extent": _dump(dataset.extent),
        "schema": schema_view,
        "lineage": {
            "source_dataset_ids": list(dataset.source_dataset_ids),
            "created_by_run_id": dataset.created_by_run_id,
        },
        "metadata": metadata,
    }
    return {key: value for key, value in result.items() if value not in (None, {}, [])}


def _memory_view(item: MemoryItem) -> dict[str, Any]:
    metadata = {
        key: item.metadata.get(key)
        for key in ("category", "confidence", "importance", "durability", "source_task_id", "source_run_id")
        if item.metadata.get(key) is not None
    }
    return {
        "id": item.id,
        "key": _clip_text(item.key, 200),
        "value": _clip_text(item.value, 1200),
        "metadata": metadata,
    }


def _conversation_memory_view(value: ConversationMemory | dict[str, Any] | None) -> dict[str, Any]:
    raw = _dump(value)
    if not raw:
        return {}

    def entries(key: str, limit: int = 8) -> list[dict[str, Any]]:
        result = []
        for item in (raw.get(key) or [])[-limit:]:
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "content": _clip_text(item.get("content", ""), 500),
                    "source_message_id": item.get("source_message_id"),
                    "source_task_id": item.get("source_task_id"),
                    "source_run_id": item.get("source_run_id"),
                    "reference_type": item.get("reference_type"),
                    "reference_id": item.get("reference_id"),
                }
            )
        return result

    return {
        "summary": _clip_text(raw.get("summary", ""), 1800),
        "key_facts": entries("key_facts"),
        "decisions": entries("decisions"),
        "important_references": entries("important_references"),
        "unresolved_topics": entries("unresolved_topics", 20),
    }


def _profile_view(value: UserProfile | dict[str, Any] | None) -> dict[str, Any]:
    raw = _dump(value)
    if not raw:
        return {}
    return {
        key: raw.get(key)
        for key in ("language", "response_style", "measurement_system", "preferred_output_format")
        if raw.get(key) is not None
    }


def _run_view(value: Any) -> dict[str, Any]:
    raw = _dump(value)
    if not raw:
        return {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    result = metadata.get("result") if isinstance(metadata.get("result"), dict) else {}
    relations = {
        key: metadata.get(key)
        for key in ("continued_from", "retry_of", "resumed_from")
        if metadata.get(key) is not None
    }
    view = {
        "id": raw.get("id"),
        "task_id": raw.get("task_id"),
        "parent_run_id": raw.get("parent_run_id"),
        "agent_id": raw.get("agent_id"),
        "status": raw.get("status"),
        "error": _clip_text(raw.get("error", ""), 500) if raw.get("error") else None,
        "turn_count": raw.get("turn_count"),
        "tool_call_count": raw.get("tool_call_count"),
        "relations": relations,
        "result": {
            key: _compact_value(result.get(key))
            for key in ("status", "summary", "datasets", "artifacts", "warnings", "error")
            if result.get(key) not in (None, [], {}, "")
        },
    }
    return {key: item for key, item in view.items() if item not in (None, {}, [], "")}


def _observation_view(value: Any) -> Any:
    raw = _dump(value)
    if not raw:
        return _compact_value(value)
    batch = raw.get("tool_observations")
    if isinstance(batch, list):
        return {
            "tool_observations": [
                _observation_item_view(item)
                for item in batch[:8]
                if isinstance(item, dict)
            ]
        }
    return _observation_item_view(raw)


def _observation_item_view(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _compact_value(raw.get(key))
        for key in (
            "type",
            "code",
            "completed",
            "total",
            "fingerprint",
            "results",
            "message",
            "call_id",
            "status",
            "output",
            "error",
            "warnings",
            "datasets",
            "artifacts",
            "retryable",
            "duration_ms",
            "accepted",
            "verified",
            "verification_problems",
            "recovery_action",
            "directive",
            "attempts",
            "rationale",
        )
        if raw.get(key) not in (None, [], {}, "")
    }


def _recent_messages(messages: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    result = []
    for item in messages:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        result.append({"role": role, "content": _clip_text(content, 1600)})
    return result[-limit:]


def _tool_capabilities(definitions: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    seen: set[str] = set()
    for item in definitions:
        name = str(item.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(
            {
                "name": name,
                "summary": _clip_text(item.get("description", ""), 180),
            }
        )
    return result


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return _clip_text(str(value), 300)
    if isinstance(value, str):
        return _clip_text(value, 1000)
    if isinstance(value, dict):
        items = list(value.items())[:16]
        return {str(key): _compact_value(item, depth=depth + 1) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_compact_value(item, depth=depth + 1) for item in list(value)[:16]]
    return value


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {}


def _clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


__all__ = ["ContextAssembler", "ContextPriority", "ContextSection", "estimate_tokens"]
