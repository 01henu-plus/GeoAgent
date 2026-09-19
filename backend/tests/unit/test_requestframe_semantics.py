import json

import pytest

from app.core.models import Dataset, DatasetKind, RequestFrame, StateSnapshot
from app.decision import Planner, TaskDecomposer
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.understanding.deterministic import extract_request_hints
from app.understanding.interpreter import RequestInterpreter
from app.understanding.models import ReferenceResolution


def _state() -> StateSnapshot:
    return StateSnapshot(conversation_id="conv_semantics")


def test_deterministic_request_fields_cover_operations_parameters_roles_and_rendering():
    buffer_hints = extract_request_hints("给 roads 做 500 米缓冲")
    assert buffer_hints.operations == ["buffer"]
    assert buffer_hints.distance == 500
    assert "road" in buffer_hints.dataset_roles

    terrain_hints = extract_request_hints("把 DEM 重投影到 EPSG:3857 再计算坡度")
    assert terrain_hints.operations == ["reproject", "slope"]
    assert terrain_hints.target_crs == "EPSG:3857"
    assert "terrain" in terrain_hints.dataset_roles

    zonal_hints = extract_request_hints("用 population 和 boundary 做分区统计")
    assert zonal_hints.operations == ["zonal_statistics"]
    assert {"population", "boundary"} <= set(zonal_hints.dataset_roles)

    field_hints = extract_request_hints("按字段 landuse 融合图斑")
    assert field_hints.field == "landuse"
    assert extract_request_hints("查找最近的道路").predicate == "nearest"
    assert extract_request_hints("把结果渲染到地图").render_requested is True
    assert extract_request_hints("执行空间连接").predicate is None


@pytest.mark.asyncio
async def test_deterministic_interpreter_populates_request_frame_fields():
    frame = await RequestInterpreter().interpret(
        message="给 roads 做 500 米缓冲",
        state=_state(),
        resolution=ReferenceResolution(),
    )

    assert frame.operations == ["buffer"]
    assert frame.parameters["distance"] == 500
    assert "road" in frame.dataset_roles


class StructuredFrameAdapter(ModelAdapter):
    supports_structured_output = True

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=json.dumps(
                {
                    "mode": "new_task",
                    "goal": "执行用户指定的空间操作",
                    "operations": ["buffer"],
                    "parameters": {"distance": 250},
                    "dataset_roles": ["road"],
                    "render_requested": True,
                    "references": [],
                    "capabilities": ["vector_analysis"],
                    "needs_planning": True,
                    "needs_tool": True,
                },
                ensure_ascii=False,
            )
        )


@pytest.mark.asyncio
async def test_model_structured_request_frame_fields_are_preserved():
    frame = await RequestInterpreter().interpret(
        message="执行用户指定的空间操作",
        state=_state(),
        resolution=ReferenceResolution(),
        model_adapter=StructuredFrameAdapter(),
    )

    assert frame.operations == ["buffer"]
    assert frame.parameters == {"distance": 250}
    assert frame.dataset_roles == ["road"]
    assert frame.render_requested is True


def test_planner_consumes_structured_fields_without_goal_parsing():
    roads = Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    frame = RequestFrame(
        mode="new_task",
        goal="执行用户指定的空间操作",
        operations=["buffer"],
        parameters={"distance": 250},
        dataset_roles=["road"],
        capabilities=["vector_analysis"],
        needs_planning=True,
        needs_tool=True,
    )
    plan = Planner().build(frame, [roads])
    assert plan.steps[1].tool_name == "vector.buffer"
    assert plan.steps[1].arguments["distance"] == 250

    keyword_only = RequestFrame(
        mode="new_task",
        goal="给 roads 做缓冲",
        capabilities=["vector_analysis"],
        needs_planning=True,
        needs_tool=True,
    )
    assert Planner().build(keyword_only, [roads]).steps == []


def test_task_decomposer_only_uses_structured_dataset_roles():
    datasets = [
        Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson"),
        Dataset(name="population", kind=DatasetKind.VECTOR, path="population.geojson", format="geojson"),
    ]
    structured = RequestFrame(mode="new_task", goal="无角色文本", dataset_roles=["road", "population"])
    assert len(TaskDecomposer().decompose(structured, datasets)) == 2

    unstructured = RequestFrame(mode="new_task", goal="道路、人口和 DEM", dataset_roles=[])
    assert TaskDecomposer().decompose(unstructured, datasets) == []
