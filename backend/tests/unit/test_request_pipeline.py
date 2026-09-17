import json

import pytest

from app.core.models import (
    Artifact,
    ArtifactKind,
    InteractionMode,
    RequestFrame,
    ResolvedReference,
    Run,
    RunStatus,
    StateSnapshot,
    Task,
    TaskStatus,
)
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.state import StateStore
from app.understanding.interpreter import RequestInterpreter
from app.understanding.models import ReferenceResolution
from app.understanding.pipeline import RequestUnderstandingPipeline
from app.understanding.validator import RequestFrameValidator


class FakeInterpreterAdapter(ModelAdapter):
    supports_structured_output = True

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=json.dumps(
                {
                    "mode": "continue_task",
                    "goal": "使用刚才的数据计算 NDVI",
                    "references": [
                        {"mention": "刚才的数据", "type": "artifact", "target_id": "art_real"}
                    ],
                    "capabilities": ["raster_analysis", "artifact_read", "artifact_write"],
                    "target_task_id": "task_real",
                    "target_run_id": None,
                    "needs_planning": True,
                    "needs_tool": True,
                    "unresolved_references": [],
                    "confidence": 0.95,
                },
                ensure_ascii=False,
            )
        )


def _state() -> StateSnapshot:
    run = Run(id="run_real", task_id="task_real", agent_id="main", conversation_id="conv_test", status=RunStatus.RUNNING)
    artifact = Artifact(id="art_real", name="sentinel.tif", kind=ArtifactKind.DATASET, run_id=run.id)
    return StateSnapshot(
        conversation_id="conv_test",
        active_task_id="task_real",
        active_run_id=run.id,
        task_goal="分析栅格数据",
        task_status=TaskStatus.RUNNING,
        recent_runs=[run],
        recent_artifacts=[artifact],
        known_task_ids=["task_real"],
        known_run_ids=[run.id],
        known_artifact_ids=[artifact.id],
    )


@pytest.mark.asyncio
async def test_interpreter_uses_structured_model_frame():
    frame = await RequestInterpreter().interpret(
        message="继续用刚才的数据计算 NDVI",
        state=_state(),
        resolution=ReferenceResolution(),
        model_adapter=FakeInterpreterAdapter(),
    )

    assert frame.mode is InteractionMode.CONTINUE_TASK
    assert frame.capabilities == ["raster_analysis", "artifact_read", "artifact_write"]
    assert frame.target_task_id == "task_real"


def test_validator_rejects_hallucinated_targets_and_references():
    frame = RequestFrame(
        mode=InteractionMode.CONTINUE_TASK,
        goal="使用不存在的结果",
        references=[ResolvedReference(mention="不存在的结果", type="artifact", target_id="art_fake")],
        target_task_id="task_fake",
        target_run_id="run_fake",
        needs_tool=True,
    )

    validated = RequestFrameValidator().validate(frame, _state())

    assert validated.references == []
    assert validated.target_task_id == "task_real"
    assert validated.target_run_id is None
    assert validated.needs_tool is False
    assert "不存在的结果" in validated.unresolved_references
    assert "task:task_fake" in validated.unresolved_references


@pytest.mark.asyncio
async def test_pipeline_uses_state_for_continue_without_llm(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    store.upsert_conversation("conv_test", "测试", "2026-01-01T00:00:00+00:00")
    store.save_task(Task(id="task_real", goal="分析栅格数据", status=TaskStatus.RUNNING, conversation_id="conv_test"))
    store.save_run(Run(id="run_real", task_id="task_real", agent_id="main", conversation_id="conv_test", status=RunStatus.RUNNING))

    frame = await RequestUnderstandingPipeline(store).understand("conv_test", "继续")

    assert frame.mode is InteractionMode.CONTINUE_TASK
    assert frame.target_task_id == "task_real"
    assert frame.unresolved_references == []


@pytest.mark.asyncio
async def test_pipeline_marks_continue_without_task_as_unresolved(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()

    frame = await RequestUnderstandingPipeline(store).understand("conv_empty", "继续")

    assert frame.mode is InteractionMode.CONTINUE_TASK
    assert frame.target_task_id is None
    assert "当前任务" in frame.unresolved_references
    assert frame.confidence <= 0.35
