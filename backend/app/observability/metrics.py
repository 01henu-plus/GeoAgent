"""运行期轻量计数器。"""

from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def increment(self, name: str, amount: int = 1) -> None:
        self._counts[name] += amount

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)

