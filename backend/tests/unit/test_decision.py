from app.core.models import AgentRequest, Dataset, DatasetKind
from app.decision import IntentResolver, ParallelismAnalyzer, Planner, TaskDecomposer


def test_intent_extracts_meter_distance_and_topics():
    request = AgentRequest(user_input="综合道路、人口和 DEM，按 500 米评价区域")
    datasets = [
        Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson"),
        Dataset(name="population", kind=DatasetKind.VECTOR, path="population.geojson", format="geojson"),
        Dataset(name="dem", kind=DatasetKind.RASTER, path="dem.tif", format="tif"),
    ]
    result = IntentResolver().resolve(request, datasets)
    assert result.entities["distance"] == 500
    assert result.entities["terrain_requested"] is True
    assert result.intent.value == "SPATIAL_ANALYSIS"


def test_distance_analysis_is_not_classified_as_buffer():
    request = AgentRequest(user_input="计算 roads 和 population 的 500 米距离分布")

    result = IntentResolver().resolve(request)

    assert result.entities["distance_analysis_requested"] is True
    assert result.entities["buffer_requested"] is False


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
    tasks = TaskDecomposer().decompose(request, datasets)
    assert {task.goal.split()[1] for task in tasks} == {"road", "population", "terrain"}
    assert all(task.dataset_ids for task in tasks)


def test_planner_composes_reprojection_before_buffer():
    request = AgentRequest(user_input="先将 roads 重投影到 EPSG:3857，再生成 500 米缓冲区")
    roads = Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    intent = IntentResolver().resolve(request, [roads])

    plan = Planner().build(request.user_input, intent, [roads])

    assert intent.entities["operations"] == ["reproject", "buffer"]
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
    intent = IntentResolver().resolve(request, [roads])

    plan = Planner().build(request.user_input, intent, [roads])

    assert plan.clarification is not None
    assert not any(step.tool_name for step in plan.steps)
