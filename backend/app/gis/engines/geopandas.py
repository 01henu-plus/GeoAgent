"""GeoPandas 引擎状态。"""

from .base import GISEngine


class GeoPandasEngine(GISEngine):
    name = "geopandas"

    def available(self) -> bool:
        try:
            import geopandas  # noqa: F401

            return True
        except ImportError:
            return False

