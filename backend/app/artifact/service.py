"""Artifact 的发布和查询。"""

from __future__ import annotations

from pathlib import Path

from app.core.models import Artifact, ArtifactKind
from app.execution.sandbox import WorkspaceManager
from app.state import StateStore


class ArtifactService:
    def __init__(self, store: StateStore, workspace: WorkspaceManager, *, system_owned: bool = False) -> None:
        self.store = store
        self.workspace = workspace
        self.system_owned = system_owned

    def publish(self, path: str | Path, *, run_id: str, kind: ArtifactKind = ArtifactKind.OTHER, dataset_id: str | None = None, description: str = "", owner_user_id: str | None = None, system_owned: bool = False) -> Artifact:
        target = self.workspace.resolve(path, allow_missing=False)
        owner = owner_user_id or self.store.user_id_for_run(run_id)
        if owner is None and not (system_owned or self.system_owned):
            raise PermissionError("创建产物必须绑定用户；系统产物请明确指定 system_owned=True")
        artifact = Artifact(name=target.name, kind=kind, path=str(target), dataset_id=dataset_id, run_id=run_id, owner_user_id=owner, description=description)
        self.store.save_artifact(artifact)
        return artifact

    def list(self, run_id: str | None = None) -> list[Artifact]:
        return self.store.list_artifacts(run_id)
