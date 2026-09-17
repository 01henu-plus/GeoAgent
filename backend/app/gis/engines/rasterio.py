"""Rasterio 引擎状态。"""

from .base import GISEngine


class RasterioEngine(GISEngine):
    name = "rasterio"

    def available(self) -> bool:
        try:
            import rasterio  # noqa: F401

            return True
        except ImportError:
            return False

