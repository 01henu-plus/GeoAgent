import pytest

from app.core.models import AgentRequest, Dataset, DatasetKind, Plan, PlanStep, RequestFrame
from app.decision import ParallelismAnalyzer, Planner, TaskDecomposer
from app.understanding.deterministic import extract_request_hints


def test_intent_extracts_meter_distance_and_topics():
    request = AgentRequest(user_input="综合道路、人口和 DEM，按 500 米评价区域")
    datasets = [
        Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson"),
        Dataset(name="population", kind=DatasetKind.VECTOR, path="population.geojson", format="geojson"),
        Dataset(name="dem", kind=DatasetKind.RASTER, path="dem.tif", format="tif"),
    ]
    hints = extract_request_hints(request.user_input, datasets)
    assert hints.distance == 500
    assert hints.terrain_requested is True
    assert hints.operations == []


def test_distance_analysis_is_not_classified_as_buffer():
    request = AgentRequest(user_input="计算 roads 和 population 的 500 米距离分布")

    hints = extract_request_hints(request.user_input)

    assert hints.operations == ["distance"]


def test_parallelism_groups_dependency_levels():
    from app.core.models import SubTask

    a = SubTask(id="a", goal="a", description="a")
    b = SubTask(id="b", goal="b", description="b")
    c = SubTask(id="c", goal="c", description="c", dependencies=["a", "b"])
    batches = ParallelismAnalyzer().batches([a, b, c])
    assert {item.id for item in batches[0]} == {"a", "b"}
    assert [item.id for item in batches[1]] == ["c"]


def test_decomposer_creates_thematic_tasks():
    request = AgentRequest(user_input="综合道路、人口和 DEM")
    datasets = [
        Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson"),
        Dataset(name="population", kind=DatasetKind.VECTOR, path="population.geojson", format="geojson"),
        Dataset(name="dem", kind=DatasetKind.RASTER, path="dem.tif", format="tif"),
    ]
    frame = RequestFrame(mode="new_task", goal=request.user_input, dataset_roles=["road", "population", "terrain"])
    tasks = TaskDecomposer().decompose(frame, datasets)
    assert {task.goal.split()[1] for task in tasks} == {"road", "population", "terrain"}
    assert all(task.dataset_ids for task in tasks)


def test_planner_composes_reprojection_before_buffer():
    request = AgentRequest(user_input="先将 roads 重投影到 EPSG:3857，再生成 500 米缓冲区")
    roads = Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    hints = extract_request_hints(request.user_input, [roads])

    frame = RequestFrame(
        mode="new_task",
        goal=request.user_input,
        operations=["reproject", "buffer"],
        parameters={"target_crs": "EPSG:3857", "distance": 500},
        dataset_roles=["road", "vector"],
        capabilities=["vector_analysis", "crs_transform"],
        needs_planning=True,
        needs_tool=True,
    )
    plan = Planner().build(frame, [roads])

    assert hints.operations == ["reproject", "buffer"]
    assert [step.tool_name for step in plan.steps] == [
        "dataset.inspect",
        "crs.reproject",
        "vector.buffer",
        "vector.validate",
        "map.render",
    ]
    assert plan.steps[2].arguments["dataset_id"] == "${reproject.dataset_id}"
    assert plan.metadata["operation"] == "buffer"


def test_planner_asks_for_missing_required_parameter():
    request = AgentRequest(user_input="给 roads 生成缓冲区")
    roads = Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    frame = RequestFrame(
        mode="new_task",
        goal=request.user_input,
        operations=["buffer"],
        dataset_roles=["road"],
        capabilities=["vector_analysis"],
        needs_planning=True,
        needs_tool=True,
    )
    plan = Planner().build(frame, [roads])

    assert plan.clarification is not None
    assert not any(step.tool_name for step in plan.steps)


def test_delegation_is_a_decision_not_fake_plan_steps():
    datasets = [
        Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson"),
        Dataset(name="population", kind=DatasetKind.VECTOR, path="population.geojson", format="geojson"),
        Dataset(name="dem", kind=DatasetKind.RASTER, path="dem.tif", format="tif"),
    ]
    frame = RequestFrame(
        mode="new_task",
        goal="综合道路、人口和 DEM 分析当前区域",
        dataset_roles=["road", "population", "terrain"],
        capabilities=["vector_analysis", "raster_analysis"],
        needs_planning=True,
        needs_tool=True,
    )

    plan = Planner().build(frame, datasets)

    assert plan.steps == []
    assert plan.clarification is None


def test_unknown_request_does_not_create_context_prepare_step():
    frame = RequestFrame(mode="new_task", goal="帮我看看这个问题", capabilities=[], needs_planning=True)

    plan = Planner().build(frame, [])

    assert plan.steps == []


def test_plan_rejects_non_executable_step():
    with pytest.raises(ValueError, match="不可执行步骤"):
        Plan(
            goal="非法计划",
            intent="UNKNOWN",
            steps=[PlanStep(id="invalid", title="控制步骤", action="delegate")],
        )
