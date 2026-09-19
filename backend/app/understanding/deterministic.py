"""离线请求理解所需的确定性文本解析。

这里输出的是 understanding 层内部使用的 hints，而不是跨层的 Intent 模型。
它们帮助 RequestInterpreter 构造 RequestFrame，不进入 Runtime、Checkpoint 或
其它业务状态。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from app.core.models import Dataset

_DISTANCE_RE = re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>米|公尺|m|公里|千米|km)", re.IGNORECASE)
_CRS_RE = re.compile(r"\bEPSG\s*[:：]?\s*(\d{4,6})\b", re.IGNORECASE)
_FIELD_RE = re.compile(r"(?:字段|列|属性|field|按(?:字段|列|属性)?)\s*[：:=]?\s*[`\"“”']?([\w\u3400-\u9fff-]+)", re.IGNORECASE)

OPERATION_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
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

INSPECTION_TERMS = ("检查", "查看", "数据质量", "字段", "属性", "元数据", "信息", "inspect", "schema")
KNOWLEDGE_TERMS = ("什么是", "如何", "怎么理解", "原理", "为什么要", "区别", "epsg", "坐标系", "crs")
DIAGNOSIS_TERMS = ("运行记录", "运行状态", "为什么失败", "刚才失败", "上一轮失败", "错误信息", "trace", "日志")
RESULT_TERMS = ("结果", "产物", "地图", "图层", "刚才", "上一轮", "上一次", "刚生成", "输出")
ROLE_TERMS: dict[str, tuple[str, ...]] = {
    "road": ("道路", "路网", "公路", "road", "roads", "street", "可达"),
    "population": ("人口", "居民", "人口栅格", "population", "pop", "人群"),
    "terrain": ("dem", "高程", "地形", "坡度", "terrain", "elevation"),
    "boundary": ("边界", "行政区", "掩膜", "范围", "boundary", "mask", "polygon"),
    "raster": ("栅格", "影像", "遥感", "raster", "tif", "tiff"),
    "vector": ("矢量", "图层", "vector", "shp", "geojson", "gpkg"),
}


@dataclass(slots=True)
class DeterministicRequestHints:
    operations: list[str] = dataclass_field(default_factory=list)
    distance: float | None = None
    target_crs: str | None = None
    field: str | None = None
    predicate: str | None = None
    dataset_roles: list[str] = dataclass_field(default_factory=list)
    mentioned_dataset_ids: list[str] = dataclass_field(default_factory=list)
    road_requested: bool = False
    population_requested: bool = False
    terrain_requested: bool = False
    boundary_requested: bool = False
    render_requested: bool = False
    result_reference_requested: bool = False
    is_greeting: bool = False
    is_question: bool = False
    is_knowledge_query: bool = False
    is_diagnosis: bool = False
    requires_dataset: bool = False

    @property
    def operation(self) -> str | None:
        return self.operations[0] if self.operations else None


def extract_request_hints(message: str, datasets: list[Dataset] | None = None) -> DeterministicRequestHints:
    raw = message.strip()
    text = raw.casefold()
    operations = operations_in_text(text)
    distance_match = _DISTANCE_RE.search(text)
    distance = None
    if distance_match:
        amount = float(distance_match.group("amount"))
        unit = distance_match.group("unit").casefold()
        distance = amount * (1000 if unit in {"公里", "千米", "km"} else 1)
    crs_match = _CRS_RE.search(raw)
    target_crs = f"EPSG:{crs_match.group(1)}" if crs_match else None
    if target_crs is None and any(term in text for term in ("wgs84", "wgs 84")):
        target_crs = "EPSG:4326"
    field_match = _FIELD_RE.search(raw)
    field_name = field_match.group(1).strip("`\"“”'") if field_match else None
    roles = [role for role, terms in ROLE_TERMS.items() if any(term in text for term in terms)]
    terrain_requested = "slope" in operations or any(term in text for term in ROLE_TERMS["terrain"])
    is_question = _is_question(text)
    is_knowledge = any(term in text for term in KNOWLEDGE_TERMS)
    is_diagnosis = any(term in text for term in DIAGNOSIS_TERMS)
    requires_dataset = bool(operations) or any(term in text for term in INSPECTION_TERMS)
    mentioned = [dataset.id for dataset in datasets or [] if mentions_dataset(text, dataset)]
    predicate = None
    if any(term in text for term in ("包含", "within")):
        predicate = "within"
    elif any(term in text for term in ("最近", "nearest")):
        predicate = "nearest"
    elif any(term in text for term in ("相交", "intersects")):
        predicate = "intersects"
    return DeterministicRequestHints(
        operations=operations,
        distance=distance,
        target_crs=target_crs,
        field=field_name,
        predicate=predicate,
        dataset_roles=roles,
        mentioned_dataset_ids=mentioned,
        road_requested=any(term in text for term in ROLE_TERMS["road"]),
        population_requested=any(term in text for term in ROLE_TERMS["population"]),
        terrain_requested=terrain_requested,
        boundary_requested=any(term in text for term in ROLE_TERMS["boundary"]),
        render_requested=any(term in text for term in ("地图", "制图", "可视化", "渲染", "map", "render")),
        result_reference_requested=any(term in text for term in RESULT_TERMS),
        is_greeting=_is_greeting(text),
        is_question=is_question,
        is_knowledge_query=is_knowledge,
        is_diagnosis=is_diagnosis,
        requires_dataset=requires_dataset,
    )


def operations_in_text(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for name, terms in OPERATION_TERMS:
        positions = [text.find(term.casefold()) for term in terms if text.find(term.casefold()) >= 0]
        if positions:
            found.append((min(positions), name))
    return [name for _, name in sorted(found)]


def mentions_dataset(text: str, dataset: Dataset) -> bool:
    names = {
        dataset.name.casefold(),
        dataset.path.casefold(),
        dataset.path.split("\\")[-1].casefold(),
        dataset.path.split("/")[-1].casefold(),
    }
    if any(value and value in text for value in names):
        return True
    role_terms = ROLE_TERMS.get(dataset.kind.value.casefold(), ())
    if any(term in text for term in role_terms):
        return True
    candidate = f"{dataset.name} {dataset.path}".casefold()
    return any(
        any(term in candidate for term in terms) and any(term in text for term in terms)
        for terms in ROLE_TERMS.values()
    )


def _is_greeting(text: str) -> bool:
    compact = re.sub(r"[\s，。！!？?~～]+", "", text)
    return compact in {"你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "在吗"}


def _is_question(text: str) -> bool:
    return "?" in text or "？" in text or any(term in text for term in ("什么", "怎么", "如何", "为什么", "能否", "可以吗"))


__all__ = ["DeterministicRequestHints", "extract_request_hints", "operations_in_text"]
