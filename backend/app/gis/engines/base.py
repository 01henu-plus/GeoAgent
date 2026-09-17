"""GIS 引擎的最小协议。"""

from abc import ABC, abstractmethod


class GISEngine(ABC):
    name: str

    @abstractmethod
    def available(self) -> bool:
        """返回当前环境能否使用此引擎。"""

