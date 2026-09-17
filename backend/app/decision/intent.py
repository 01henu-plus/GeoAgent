"""GeoAgent 的请求理解层。

这里不执行 GIS 运算。它是没有配置大模型时的有限离线兜底，不是开放域自然语言
理解器，也不参与已配置模型时的主决策循环。配置模型后，Main Agent 直接把会话、
数据上下文和工具定义交给模型；只有模型未配置或没有返回有效答复时，才使用这里
的规则结果保证常见 GIS 请求仍能走完 ``理解 -> 规划 -> 工具 -> 验证`` 链路。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.core.models import AgentRequest, Dataset, IntentResult, IntentType

_DISTANCE_RE = re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>米|公尺|m|公里|千米|km)", re.IGNORECASE)
_CRS_RE = re.compile(r"\bEPSG\s*[:：]?\s*(\d{4,6})\b", re.IGNORECASE)
_FIELD_RE = re.compile(r"(?:字段|列|属性|field|按)\s*[：:=]?\s*[`\"“”']?([\w\u3400-\u9fff-]+)", re.IGNORECASE)

_OPERATION_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("buffer", ("缓冲区", "缓冲", "buffer")),
    ("clip", ("裁剪", "截取", "clip")),
    ("intersection", ("相交", "交集", "叠加", "intersection", "overlay")),
    ("spatial_join", ("空间连接", "空间关联", "spatial join", "sjoin")),
    ("dissolve", ("融合", "溶解", "dissolve")),
    ("zonal_statistics", ("分区统计", "区域统计", "zonal statistics", "zonal")),
    ("slope", ("坡度", "坡向", "地形分析", "slope")),
    ("reproject", ("重投影", "转投影", "坐标转换", "投影转换", "reproject", "transform crs")),
    ("repair", ("修复几何", "修复无效", "几何修复", "repair")),
    ("validate", ("验证几何", "几何检查", "有效性检查", "validate")),
    ("distance", ("距离分析", "距离分布", "最近距离", "测距", "distance")),
)

_INSPECTION_TERMS = ("检查", "查看", "数据质量", "字段", "属性", "元数据", "信息", "inspect", "schema")
_KNOWLEDGE_TERMS = ("什么是", "如何", "怎么理解", "原理", "为什么要", "区别", "epsg", "坐标系", "crs")
_DIAGNOSIS_TERMS = ("运行记录", "运行状态", "为什么失败", "刚才失败", "上一轮失败", "错误信息", "trace", "日志")
_RESULT_TERMS = ("结果", "产物", "地图", "图层", "刚才", "上一轮", "上一次", "刚生成", "输出")
_ROLE_TERMS: dict[str, tuple[str, ...]] = {
    "road": ("道路", "路网", "公路", "road", "roads", "street", "可达"),
    "population": ("人口", "居民", "人口栅格", "population", "pop", "人群"),
    "terrain": ("dem", "高程", "地形", "坡度", "terrain", "elevation"),
    "boundary": ("边界", "行政区", "掩膜", "范围", "boundary", "mask", "polygon"),
    "raster": ("栅格", "影像", "遥感", "raster", "tif", "tiff"),
    "vector": ("矢量", "图层", "vector", "shp", "geojson", "gpkg"),
}


class IntentResolver:
    """离线兜底：从请求文本和已解析数据集提取有限的任务提示。"""

    def resolve(self, request: AgentRequest, datasets: list[Dataset] | None = None) -> IntentResult:
        raw = request.user_input.strip()
        text = raw.casefold()
        entities: dict[str, object] = {
            "operations": [],
            "dataset_roles": [],
            "mentioned_dataset_ids": [],
            "is_greeting": _is_greeting(text),
            "is_question": _is_question(text),
            "result_reference_requested": any(term in text for term in _RESULT_TERMS),
        }

        operations = _operations_in_text(text)
        # “坡度/地形”是明确的分析操作；单独出现 DEM 仍只表示数据主题。
        if "slope" in operations:
            entities["terrain_requested"] = True
        else:
            entities["terrain_requested"] = any(term in text for term in _ROLE_TERMS["terrain"])
        entities["operations"] = operations
        entities["operation_sequence"] = list(operations)
        entities["operation"] = operations[0] if operations else None
        entities["buffer_requested"] = "buffer" in operations
        entities["distance_analysis_requested"] = "distance" in operations
        entities["render_requested"] = any(term in text for term in ("地图", "制图", "可视化", "渲染", "map", "render"))
        entities["population_requested"] = any(term in text for term in _ROLE_TERMS["population"])
        entities["road_requested"] = any(term in text for term in _ROLE_TERMS["road"])
        entities["boundary_requested"] = any(term in text for term in _ROLE_TERMS["boundary"])

        distance = _DISTANCE_RE.search(text)
        if distance:
            amount = float(distance.group("amount"))
            unit = distance.group("unit").casefold()
            entities["distance"] = amount * (1000 if unit in {"公里", "千米", "km"} else 1)
            entities["distance_unit"] = "meter"

        crs = _CRS_RE.search(raw)
        if crs:
            entities["target_crs"] = f"EPSG:{crs.group(1)}"
        elif any(term in text for term in ("wgs84", "wgs 84")):
            entities["target_crs"] = "EPSG:4326"
        field = _FIELD_RE.search(raw)
        if field:
            entities["field"] = field.group(1).strip("`\"“”'")
        if any(term in text for term in ("包含", "within")):
            entities["predicate"] = "within"
        elif any(term in text for term in ("最近", "nearest")):
            entities["predicate"] = "nearest"
        else:
            entities["predicate"] = "intersects"

        role_names = [role for role, terms in _ROLE_TERMS.items() if any(term in text for term in terms)]
        entities["dataset_roles"] = role_names
        if datasets:
            mentioned = [dataset.id for dataset in datasets if _mentions_dataset(text, dataset)]
            entities["mentioned_dataset_ids"] = mentioned

        intent, confidence = self._classify(text, operations, entities)
        entities["requires_dataset"] = intent in {IntentType.DATA_INSPECTION, IntentType.SPATIAL_ANALYSIS, IntentType.DATA_TRANSFORMATION}
        rationale = _rationale(intent, operations, entities)
        return IntentResult(intent=intent, confidence=confidence, entities=entities, rationale=rationale)

    @staticmethod
    def _classify(text: str, operations: list[str], entities: dict[str, object]) -> tuple[IntentType, float]:
        if any(term in text for term in _DIAGNOSIS_TERMS):
            return IntentType.RUN_DIAGNOSIS, 0.96
        thematic_count = sum(bool(entities.get(key)) for key in ("road_requested", "population_requested", "terrain_requested"))
        if not operations and thematic_count >= 2 and any(term in text for term in ("综合", "评价", "分别", "多方面", "多个方面", "分析")):
            return IntentType.SPATIAL_ANALYSIS, 0.93
        if not operations and entities.get("result_reference_requested") and any(term in text for term in ("解释", "怎么看", "分析一下", "详情", "是否完成", "怎么样")):
            return IntentType.RESULT_INTERPRETATION, 0.9
        if not operations and entities.get("render_requested"):
            return IntentType.DATA_INSPECTION, 0.86
        if operations:
            if operations == ["validate"]:
                return IntentType.DATA_INSPECTION, 0.92
            if operations and set(operations).issubset({"reproject", "repair", "dissolve"}):
                return IntentType.DATA_TRANSFORMATION, 0.92
            return IntentType.SPATIAL_ANALYSIS, 0.94
        if any(term in text for term in ("脚本", "python", "代码", "编程")):
            return IntentType.CODE_TASK, 0.88
        if any(term in text for term in _KNOWLEDGE_TERMS):
            return IntentType.KNOWLEDGE_QUERY, 0.82
        if any(term in text for term in _INSPECTION_TERMS):
            return IntentType.DATA_INSPECTION, 0.84
        return IntentType.UNKNOWN, 0.9 if entities.get("is_greeting") else 0.35


def _is_greeting(text: str) -> bool:
    compact = re.sub(r"[\s，。！!？?~～]+", "", text)
    return compact in {"你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "在吗"}


def _is_question(text: str) -> bool:
    return "?" in text or "？" in text or any(term in text for term in ("什么", "怎么", "如何", "为什么", "能否", "可以吗"))


def _operations_in_text(text: str) -> list[str]:
    """按用户表达顺序返回操作，避免固定关键词顺序改变执行语义。"""

    found: list[tuple[int, str]] = []
    for name, terms in _OPERATION_TERMS:
        positions = [text.find(term.casefold()) for term in terms if text.find(term.casefold()) >= 0]
        if positions:
            found.append((min(positions), name))
    return [name for _, name in sorted(found)]


def _mentions_dataset(text: str, dataset: Dataset) -> bool:
    names = {dataset.name.casefold(), dataset.path.casefold(), dataset.path.split("\\")[-1].casefold(), dataset.path.split("/")[-1].casefold()}
    if any(value and value in text for value in names):
        return True
    kind = dataset.kind.value.casefold()
    role_map: dict[str, tuple[str, ...]] = {
        "vector": _ROLE_TERMS["vector"],
        "raster": _ROLE_TERMS["raster"],
    }
    if kind in role_map and any(term in text for term in role_map[kind]):
        return True
    candidate = f"{dataset.name} {dataset.path}".casefold()
    return any(any(term in candidate for term in terms) and any(term in text for term in terms) for terms in _ROLE_TERMS.values())


def _rationale(intent: IntentType, operations: Iterable[str], entities: dict[str, object]) -> str:
    operation_text = "、".join(operations) if operations else "无明确空间操作"
    dataset_text = "、".join(str(item) for item in entities.get("dataset_roles", [])) or "未指定主题"
    return f"识别为 {intent.value}：操作={operation_text}；数据主题={dataset_text}。"


__all__ = ["IntentResolver"]
