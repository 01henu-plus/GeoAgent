from app.core.models import (
    Artifact,
    ArtifactKind,
    InteractionMode,
    Run,
    RunStatus,
    StateSnapshot,
    TaskStatus,
)
from app.understanding.models import ReferenceResolution
from app.understanding.reference_resolver import ReferenceResolver
from app.understanding.rule_gate import RuleGate


def _state(*, active: bool = True, failed: bool = False, artifacts: list[Artifact] | None = None) -> StateSnapshot:
    run = Run(
        id="run_previous",
        task_id="task_current",
        agent_id="main",
        conversation_id="conv_test",
        status=RunStatus.FAILED if failed else RunStatus.RUNNING,
        error="工具执行失败" if failed else None,
    )
    return StateSnapshot(
        conversation_id="conv_test",
        active_task_id="task_current" if active else None,
        active_run_id="run_previous" if active else None,
        task_goal="分析当前栅格数据" if active else None,
        task_status=TaskStatus.RUNNING if active else None,
        last_run_status=run.status if active or failed else None,
        last_error=run.error if failed else None,
        recent_runs=[run] if active or failed else [],
        recent_artifacts=artifacts or [],
        known_task_ids=["task_current"] if active else [],
        known_run_ids=["run_previous"] if active or failed else [],
    )


def test_rule_gate_recognizes_continue_with_active_task():
    frame = RuleGate().match("继续", _state(), ReferenceResolution())

    assert frame is not None
    assert frame.mode is InteractionMode.CONTINUE_TASK
    assert frame.target_task_id == "task_current"


def test_rule_gate_recognizes_retry_only_after_failure():
    frame = RuleGate().match("再试一次", _state(failed=True), ReferenceResolution())

    assert frame is not None
    assert frame.mode is InteractionMode.RETRY_TASK
    assert frame.target_run_id == "run_previous"


def test_rule_gate_recognizes_cancel_modify_query_and_chat():
    gate = RuleGate()
    state = _state()

    assert gate.match("取消", state, ReferenceResolution()).mode is InteractionMode.CANCEL_TASK
    assert gate.match("不对，把范围改成上海", state, ReferenceResolution()).mode is InteractionMode.MODIFY_TASK
    assert gate.match("刚才生成了哪些文件？", state, ReferenceResolution()).mode is InteractionMode.QUERY
    assert gate.match("你好", state, ReferenceResolution()).mode is InteractionMode.CHAT


def test_reference_resolver_prefers_latest_artifact_for_previous_result():
    artifacts = [
        Artifact(id="art_new", name="最新结果.tif", kind=ArtifactKind.DATASET, run_id="run_previous"),
        Artifact(id="art_old", name="旧结果.tif", kind=ArtifactKind.DATASET, run_id="run_previous"),
    ]
    resolution = ReferenceResolver().resolve("用上一个结果继续分析", _state(artifacts=artifacts))

    assert resolution.references[0].target_id == "art_new"
    assert resolution.references[0].type == "artifact"


def test_reference_resolver_does_not_invent_missing_artifact():
    resolution = ReferenceResolver().resolve("用刚才的数据继续", _state(artifacts=[]))

    assert resolution.references == []
    assert "刚才的数据" in resolution.unresolved_references


def test_rule_gate_does_not_accept_continue_without_task():
    assert RuleGate().match("继续", _state(active=False), ReferenceResolution()) is None

