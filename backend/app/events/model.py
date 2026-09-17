"""GeoAgent 可观测事件类型。"""

from enum import StrEnum


class EventType(StrEnum):
    RUN_CREATED = "RunCreated"
    INTENT_RESOLVED = "IntentResolved"
    PLAN_CREATED = "PlanCreated"
    DECISION_MADE = "DecisionMade"
    SUBTASK_CREATED = "SubTaskCreated"
    SUBAGENT_SPAWNED = "SubAgentSpawned"
    TOOL_STARTED = "ToolStarted"
    TOOL_COMPLETED = "ToolCompleted"
    TOOL_FAILED = "ToolFailed"
    RETRY_STARTED = "RetryStarted"
    REPAIR_SELECTED = "RepairSelected"
    REPLAN_STARTED = "ReplanStarted"
    DATASET_CREATED = "DatasetCreated"
    ARTIFACT_CREATED = "ArtifactCreated"
    VERIFICATION_STARTED = "VerificationStarted"
    VERIFICATION_FAILED = "VerificationFailed"
    SUBAGENT_COMPLETED = "SubAgentCompleted"
    CHECKPOINT_SAVED = "CheckpointSaved"
    RESUME_STARTED = "ResumeStarted"
    RUN_COMPLETED = "RunCompleted"
    RUN_FAILED = "RunFailed"
    RUN_CANCELLED = "RunCancelled"
