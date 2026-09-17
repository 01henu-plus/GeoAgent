"""把请求中的数据集名称、附件和上一轮结果解析为 Dataset。"""

from __future__ import annotations

import re

from app.core.models import AgentRequest, Dataset

_LATEST_REFERENCES = ("刚才", "上一轮", "上一次", "刚生成", "上个结果", "上一张图")
_ALL_REFERENCES = ("全部", "所有", "这些数据", "所有数据集")
_ROLE_TERMS = {
    "road": ("道路", "路网", "公路", "road", "roads"),
    "population": ("人口", "居民", "population", "pop"),
    "terrain": ("dem", "高程", "地形", "terrain", "elevation"),
    "boundary": ("边界", "行政区", "掩膜", "boundary", "mask"),
}


class DatasetResolver:
    """解析显式选择优先、自然语言其次、历史结果兜底的数据引用。"""

    def resolve(self, request: AgentRequest, registry, *, store=None) -> list[Dataset]:
        identifiers = [*request.dataset_ids, *request.attachment_ids]
        explicit = _unique(item for identifier in identifiers if (item := registry.resolve(identifier)))

        text = request.user_input.casefold()
        if store is not None and any(term in text for term in _LATEST_REFERENCES):
            referenced = self._datasets_from_latest_run(request.conversation_id, registry, store)
            if referenced:
                return _unique([*explicit, *referenced])

        datasets = registry.list()
        ordinal = _ordinal(request.user_input)
        if ordinal is not None:
            return _unique([*explicit, *datasets[ordinal - 1 : ordinal]])
        specific = [dataset for dataset in datasets if _specific_dataset_reference(text, dataset)]
        if specific:
            return _unique([*explicit, *specific])
        role_selected: list[Dataset] = []
        for role, terms in _ROLE_TERMS.items():
            if not any(term in text for term in terms):
                continue
            matches = [dataset for dataset in datasets if _dataset_matches_role(dataset, role)]
            if matches:
                roots = [dataset for dataset in matches if _is_root_dataset(dataset)]
                role_selected.append((roots or matches)[0])
        if role_selected:
            return _unique([*explicit, *role_selected])
        mentioned = [dataset for dataset in datasets if _mentions_dataset(text, dataset)]
        if mentioned:
            return _unique([*explicit, *mentioned])
        if explicit:
            return explicit
        if any(term in text for term in _ALL_REFERENCES):
            return _unique([*explicit, *datasets])
        # 单数据集时可以自然地理解“检查这个文件”；多数据集时不擅自选第一个。
        return datasets if len(datasets) == 1 else []

    @staticmethod
    def _datasets_from_latest_run(conversation_id: str | None, registry, store) -> list[Dataset]:
        runs = store.list_runs()
        candidates = [item for item in runs if conversation_id and item.conversation_id == conversation_id]
        for run in candidates:
            payload = run.metadata.get("result") if isinstance(run.metadata, dict) else None
            if not isinstance(payload, dict):
                continue
            ids = payload.get("datasets")
            if not isinstance(ids, list):
                continue
            lineage_outputs = {
                item["output_dataset_id"]
                for item in store.list_lineage()
                if item.get("run_id") == run.id
            }
            ordered_ids = [
                *[identifier for identifier in ids if identifier in lineage_outputs],
                *[identifier for identifier in ids if identifier not in lineage_outputs],
            ]
            resolved = _unique(item for identifier in ordered_ids if isinstance(identifier, str) and (item := registry.resolve(identifier)))
            if resolved:
                return [item for item in resolved if item.id in lineage_outputs] or resolved
        return []


def _unique(items) -> list[Dataset]:
    result: list[Dataset] = []
    seen: set[str] = set()
    for item in items:
        if item.id not in seen:
            result.append(item)
            seen.add(item.id)
    return result


def _ordinal(text: str) -> int | None:
    match = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*个|\b(first|second|third|1st|2nd|3rd)\b", text.casefold())
    if not match:
        return None
    value = match.group(1) or match.group(2)
    names = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "first": 1, "second": 2, "third": 3, "1st": 1, "2nd": 2, "3rd": 3}
    return names.get(value, int(value) if value.isdigit() else None)


def _mentions_dataset(text: str, dataset: Dataset) -> bool:
    names = {
        dataset.name.casefold(),
        dataset.path.casefold(),
        dataset.path.replace("/", "\\").rsplit("\\", 1)[-1].casefold(),
    }
    if any(value and value in text for value in names):
        return True
    candidate = f"{dataset.name} {dataset.path}".casefold()
    for terms in _ROLE_TERMS.values():
        if any(term in text for term in terms) and any(term in candidate for term in terms):
            return True
    return False


def _specific_dataset_reference(text: str, dataset: Dataset) -> bool:
    names = {
        dataset.name.casefold(),
        dataset.path.casefold(),
        dataset.path.replace("/", "\\").rsplit("\\", 1)[-1].casefold(),
    }
    return any(value and value in text for value in names)


def _dataset_matches_role(dataset: Dataset, role: str) -> bool:
    candidate = f"{dataset.name} {dataset.path}".casefold()
    return any(term in candidate for term in _ROLE_TERMS[role])


def _is_root_dataset(dataset: Dataset) -> bool:
    return not dataset.source_dataset_ids and not dataset.created_by_run_id


__all__ = ["DatasetResolver"]
