"""地形分析入口。"""

from app.core.models import Dataset
from app.gis.raster.service import RasterService


class TerrainAnalysis:
    def __init__(self, rasters: RasterService | None = None) -> None:
        self.rasters = rasters or RasterService()

    def slope(self, dataset: Dataset, output_path: str, **kwargs):
        return self.rasters.slope(dataset, output_path, **kwargs)

