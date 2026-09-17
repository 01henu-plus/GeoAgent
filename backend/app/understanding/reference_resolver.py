"""基于状态事实解析用户对任务、运行、数据和产物的上下文引用。"""

from __future__ import annotations

import re

from app.core.models import (
    Artifact,
    Dataset,
    RequestResources,
    ResolvedReference,
    Run,
    StateSnapshot,
    Task,
)
from app.understanding.models import ReferenceResolution

_EXPLICIT_ID_RE = re.compile(r"\b(?:artifact|art|run|task|dataset|ds)_[a-z0-9]+\b", re.IGNORECASE)
_MENTIONS = (
    "刚才下载的数据",
    "刚才生成的文件",
    "之前生成的文件",
    "刚才生成的数据",
    "刚才那个",
    "刚才的数据",
    "上一个结果",
    "前面的结果",
    "上一轮",
    "上一次",
    "当前任务",
    "这个任务",
    "刚才的运行",
    "上一次运行",
    "这个",
    "那个",
    "它",
)


class ReferenceResolver:
    """优先使用显式 ID 和最近真实对象，不依赖大模型消解第一版引用。"""

    def resolve(self, message: str, state: StateSnapshot, resources: RequestResources | None = None) -> ReferenceResolution:
        references: list[ResolvedReference] = []
        unresolved: list[str] = []
        known = self._known_objects(state, resources)

        for token in _EXPLICIT_ID_RE.findall(message):
            target = known.get(token.casefold())
            if target is None:
                unresolved.append(token)
                continue
            references.append(self._reference(token, target, confidence=1.0))

        lowered = message.casefold()
        occupied: list[tuple[int, int]] = []
        mentions: list[tuple[int, int, str]] = []
        for mention in _MENTIONS:
            start = 0
            token = mention.casefold()
            while True:
                index = lowered.find(token, start)
                if index < 0:
                    break
                mentions.append((index, index + len(token), mention))
                start = index + 1

        # 先按出现位置，再按长度排序；被较长指代占用的区间不会再次匹配短指代。
        for start, end, mention in sorted(mentions, key=lambda item: (item[0], -(item[1] - item[0]))):
            if any(start < right and end > left for left, right in occupied):
                continue
            occupied.append((start, end))
            if any(item.mention == mention for item in references):
                continue
            target = self._resolve_mention(mention, lowered, state, resources)
            if target is None:
                unresolved.append(mention)
                continue
            references.append(self._reference(mention, target, confidence=0.9))

        return ReferenceResolution(references=_dedupe(references), unresolved_references=list(dict.fromkeys(unresolved)))

    @staticmethod
    def _known_objects(state: StateSnapshot, resources: RequestResources | None = None) -> dict[str, object]:
        objects: dict[str, object] = {}
        if resources:
            for item in resources.datasets:
                objects[item.id.casefold()] = item
            for item in resources.runs:
                objects[item.id.casefold()] = item
        if state.active_task_id:
            objects[state.active_task_id.casefold()] = Task(id=state.active_task_id, goal=state.task_goal or "", status=state.task_status or "PENDING")
        for item in state.recent_runs:
            objects[item.id.casefold()] = item
        for item in state.recent_artifacts:
            objects[item.id.casefold()] = item
        for item in state.recent_datasets:
            objects[item.id.casefold()] = item
        return objects

    @staticmethod
    def _resolve_mention(mention: str, lowered: str, state: StateSnapshot, resources: RequestResources | None = None) -> object | None:
        request_datasets = resources.datasets if resources else []
        request_runs = resources.runs if resources else []
        if "任务" in mention:
            if state.active_task_id:
                return _task_reference(state)
            return None
        if "运行" in mention or mention in {"上一轮", "上一次"}:
            return request_runs[0] if request_runs else (state.recent_runs[0] if state.recent_runs else None)

        prefer_artifact = any(term in lowered for term in ("文件", "结果", "数据", "影像", "栅格", "图层"))
        if request_datasets:
            return request_datasets[0]
        if request_runs and not prefer_artifact:
            return request_runs[0]
        if prefer_artifact and state.recent_artifacts:
            return state.recent_artifacts[0]
        if prefer_artifact:
            return state.recent_datasets[0] if state.recent_datasets else None
        if state.recent_datasets:
            return state.recent_datasets[0]
        if state.recent_artifacts:
            return state.recent_artifacts[0]
        return state.recent_runs[0] if state.recent_runs else None

    @staticmethod
    def _reference(mention: str, target: object, *, confidence: float) -> ResolvedReference:
        if isinstance(target, Artifact):
            return ResolvedReference(mention=mention, type="artifact", target_id=target.id, label=target.name, confidence=confidence)
        if isinstance(target, Dataset):
            return ResolvedReference(mention=mention, type="dataset", target_id=target.id, label=target.name, confidence=confidence)
        if isinstance(target, Run):
            return ResolvedReference(mention=mention, type="run", target_id=target.id, label=target.id, confidence=confidence)
        if isinstance(target, Task):
            return ResolvedReference(mention=mention, type="task", target_id=target.id, label=target.goal, confidence=confidence)
        raise TypeError(f"不支持的引用对象：{type(target)!r}")


def _task_reference(state: StateSnapshot) -> Task:
    return Task(id=state.active_task_id or "", goal=state.task_goal or "", status=state.task_status or "PENDING")


def _dedupe(references: list[ResolvedReference]) -> list[ResolvedReference]:
    seen: set[tuple[str, str | None]] = set()
    result: list[ResolvedReference] = []
    for item in references:
        key = (item.type, item.target_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
