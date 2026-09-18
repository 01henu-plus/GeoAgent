"""Artifact 的发布和查询。"""

from __future__ import annotations

from pathlib import Path

from app.core.models import Artifact, ArtifactKind
from app.execution.sandbox import WorkspaceManager
from app.state import StateStore


class ArtifactService:
    def __init__(self, store: StateStore, workspace: WorkspaceManager) -> None:
        self.store = store
        self.workspace = workspace

    def publish(self, path: str | Path, *, run_id: str, kind: ArtifactKind = ArtifactKind.OTHER, dataset_id: str | None = None, description: str = "", owner_user_id: str | None = None) -> Artifact:
        target = self.workspace.resolve(path, allow_missing=False)
        owner = owner_user_id or self.store.user_id_for_run(run_id)
        artifact = Artifact(name=target.name, kind=kind, path=str(target), dataset_id=dataset_id, run_id=run_id, owner_user_id=owner, description=description)
        self.store.save_artifact(artifact)
        return artifact

    def list(self, run_id: str | None = None) -> list[Artifact]:
        return self.store.list_artifacts(run_id)
