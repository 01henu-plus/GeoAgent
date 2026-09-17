"""叠加分析的领域入口。"""

from app.gis.vector.service import VectorService


class OverlayAnalysis:
    def __init__(self, vectors: VectorService | None = None) -> None:
        self.vectors = vectors or VectorService()

    def intersection(self, *args, **kwargs):
        return self.vectors.intersection(*args, **kwargs)

