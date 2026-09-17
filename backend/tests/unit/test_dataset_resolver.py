from app.core.models import AgentRequest, Dataset, DatasetKind
from app.entry.dataset_resolver import DatasetResolver


class FakeRegistry:
    def __init__(self, datasets):
        self.datasets = datasets

    def resolve(self, identifier):
        return next((item for item in self.datasets if item.id == identifier), None)

    def list(self):
        return list(self.datasets)


def _registry():
    return FakeRegistry(
        [
            Dataset(id="roads", name="道路", kind=DatasetKind.VECTOR, path="input/roads.geojson", format="geojson"),
            Dataset(id="population", name="人口", kind=DatasetKind.VECTOR, path="input/population.geojson", format="geojson"),
            Dataset(id="dem", name="DEM", kind=DatasetKind.RASTER, path="input/dem.tif", format="tif"),
        ]
    )


def test_dataset_name_and_path_are_resolved():
    registry = _registry()

    assert [item.id for item in DatasetResolver().resolve(AgentRequest(user_input="检查 道路"), registry)] == ["roads"]
    assert [item.id for item in DatasetResolver().resolve(AgentRequest(user_input="检查 input/dem.tif"), registry)] == ["dem"]


def test_role_terms_resolve_road_population_and_dem():
    registry = _registry()

    assert [item.id for item in DatasetResolver().resolve(AgentRequest(user_input="分析道路"), registry)] == ["roads"]
    assert [item.id for item in DatasetResolver().resolve(AgentRequest(user_input="分析人口"), registry)] == ["population"]
    assert [item.id for item in DatasetResolver().resolve(AgentRequest(user_input="分析 DEM"), registry)] == ["dem"]


def test_multiple_named_datasets_are_all_resolved():
    registry = _registry()

    resolved = DatasetResolver().resolve(AgentRequest(user_input="比较道路和人口"), registry)

    assert [item.id for item in resolved] == ["roads", "population"]


def test_explicit_dataset_ids_take_priority():
    registry = _registry()

    resolved = DatasetResolver().resolve(AgentRequest(user_input="检查这个数据", dataset_ids=["dem"]), registry)

    assert [item.id for item in resolved] == ["dem"]
