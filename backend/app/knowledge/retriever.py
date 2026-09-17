from .repository import KnowledgeRepository


class KnowledgeRetriever:
    def __init__(self, repository: KnowledgeRepository | None = None) -> None:
        self.repository = repository or KnowledgeRepository()

    def retrieve(self, query: str) -> list[dict[str, str]]:
        return self.repository.search(query)

