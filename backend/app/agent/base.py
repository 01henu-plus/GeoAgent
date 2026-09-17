"""Agent 执行单元协议。"""

from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    async def run(self, *args, **kwargs):
        """执行一个局部 Agent Loop。"""

