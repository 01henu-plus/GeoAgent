"""固定 GIS Agent case 的离线 Runner。"""

from __future__ import annotations

import time

from app.core.models import AgentResultStatus
from app.demo import seed_demo

from .metrics import EvaluationSummary


class EvaluationRunner:
    def __init__(self, application) -> None:
        self.application = application

    async def run(self) -> EvaluationSummary:
        ids = seed_demo(self.application)
        cases = [
            ("检查 roads", [ids["roads"]], False, "dataset.inspect"),
            ("检查 roads 并生成 500 米缓冲区", [ids["roads"]], True, "vector.buffer"),
            ("综合道路、人口和 DEM，从三个方面评价当前区域", list(ids.values()), False, None),
            ("什么是 EPSG:4326，坐标单位是什么？", [], False, None),
            ("检查第二个数据", [], False, "dataset.inspect"),
        ]
        summary = EvaluationSummary(total=len(cases))
        started = time.perf_counter()
        for prompt, dataset_ids, recovery, expected_tool in cases:
            if recovery:
                summary.recovery_cases += 1
            result = await self.application.ask(prompt, dataset_ids=dataset_ids)
            run = self.application.store.get_run(result.trace_id)
            if run:
                summary.tool_calls += run.tool_call_count
            if expected_tool:
                summary.tool_selection_cases += 1
                events = self.application.store.list_events(result.trace_id)
                if any(event.payload.get("tool") == expected_tool for event in events):
                    summary.tool_selection_passed += 1
            if result.status is AgentResultStatus.SUCCESS:
                summary.succeeded += 1
                if recovery and any(event.event_type == "RepairSelected" for event in self.application.store.list_events(result.trace_id)):
                    summary.recovery_cases_passed += 1
            elif result.status is AgentResultStatus.PARTIAL:
                summary.partial += 1
            else:
                summary.failed += 1
        summary.execution_time_ms = round((time.perf_counter() - started) * 1000, 2)
        return summary
