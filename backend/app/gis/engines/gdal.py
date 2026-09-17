"""GDAL CLI 探测；实际命令由受控 Shell Executor 调度。"""

from __future__ import annotations

import shutil

from .base import GISEngine


class GDALEngine(GISEngine):
    name = "gdal"

    def available(self) -> bool:
        return shutil.which("gdalinfo") is not None

