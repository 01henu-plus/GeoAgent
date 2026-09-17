from .gis_knowledge import KNOWLEDGE


class KnowledgeRepository:
    def search(self, query: str) -> list[dict[str, str]]:
        text = query.casefold()
        matches = [(key, value) for key, value in KNOWLEDGE.items() if key in text or any(word in text for word in _keywords(key))]
        return [{"topic": key, "content": value} for key, value in matches]


def _keywords(topic: str) -> tuple[str, ...]:
    return {"crs": ("坐标", "投影", "epsg"), "buffer": ("缓冲", "距离"), "nodata": ("栅格", "nodata"), "lineage": ("谱系", "复现")}.get(topic, ())
