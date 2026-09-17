"""有并发上限和超时的 SubAgent 调度器。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class AgentScheduler:
    def __init__(self, max_parallel: int = 3, timeout_seconds: int = 180) -> None:
        self.max_parallel = max(1, max_parallel)
        self.timeout_seconds = timeout_seconds

    async def run_parallel(self, jobs: list[Callable[[], Awaitable[T]]]) -> list[T | Exception]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_one(job: Callable[[], Awaitable[T]]) -> T | Exception:
            async with semaphore:
                try:
                    return await asyncio.wait_for(job(), timeout=self.timeout_seconds)
                except Exception as exc:  # partial failure 是调度器的正式输出
                    return exc

        return list(await asyncio.gather(*(run_one(job) for job in jobs)))

