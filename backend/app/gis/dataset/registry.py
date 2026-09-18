"""Dataset Registry：所有输入和派生数据的可追踪目录。"""

from __future__ import annotations

from pathlib import Path

from app.core.models import Dataset, DatasetKind, new_id
from app.gis.dataset.inspector import DatasetInspector
from app.state import StateStore


class DatasetRegistry:
    def __init__(self, store: StateStore, inspector: DatasetInspector | None = None, *, owner_user_id: str | None = None) -> None:
        self.store = store
        self.inspector = inspector or DatasetInspector()
        self.owner_user_id = owner_user_id

    def for_user(self, user_id: str | None) -> DatasetRegistry:
        return DatasetRegistry(self.store, self.inspector, owner_user_id=user_id)

    def register(self, dataset: Dataset) -> Dataset:
        self.store.save_dataset(dataset)
        return dataset

    def register_path(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        run_id: str | None = None,
        source_dataset_ids: list[str] | None = None,
        operation: str | None = None,
        parameters: dict | None = None,
        tool_call_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> Dataset:
        target = Path(path).expanduser().resolve()
        owner = owner_user_id or self.owner_user_id
        if owner is None and run_id:
            owner = self.store.user_id_for_run(run_id)
        if run_id is None and operation is None:
            existing = next(
                (item for item in self.store.list_datasets_for_user(owner) if Path(item.path).expanduser().resolve() == target)
                if owner is not None
                else (item for item in self.store.list_datasets() if Path(item.path).expanduser().resolve() == target),
                None,
            )
            if existing:
                return existing
        dataset = self.inspector.inspect(target, name=name)
        dataset = dataset.model_copy(
            update={
                "created_by_run_id": run_id,
                "source_dataset_ids": source_dataset_ids or [],
                "owner_user_id": owner,
            }
        )
        self.register(dataset)
        if operation:
            self.store.save_lineage(
                lineage_id=new_id("lineage"),
                run_id=run_id,
                operation=operation,
                input_dataset_ids=source_dataset_ids or [],
                output_dataset_id=dataset.id,
                tool_call_id=tool_call_id,
                parameters=parameters or {},
                created_at=dataset.created_at.isoformat(),
            )
        return dataset

    def get(self, dataset_id: str, *, user_id: str | None = None) -> Dataset | None:
        owner = user_id if user_id is not None else self.owner_user_id
        return self.store.get_dataset_for_user(dataset_id, owner) if owner is not None else self.store.get_dataset(dataset_id)

    def resolve(self, identifier: str, *, user_id: str | None = None) -> Dataset | None:
        identifier = identifier.strip()
        if not identifier:
            return None
        owner = user_id if user_id is not None else self.owner_user_id
        exact = self.get(identifier, user_id=owner)
        if exact:
            return exact
        path = Path(identifier).expanduser()
        candidates = self.store.list_datasets_for_user(owner) if owner is not None else self.store.list_datasets()
        for dataset in candidates:
            if dataset.name.casefold() == identifier.casefold() or Path(dataset.path).name.casefold() == identifier.casefold():
                return dataset
            if path.exists() and Path(dataset.path).resolve() == path.resolve():
                return dataset
        return None

    def list(self, kind: DatasetKind | None = None, *, user_id: str | None = None) -> list[Dataset]:
        owner = user_id if user_id is not None else self.owner_user_id
        return self.store.list_datasets_for_user(owner, kind.value if kind else None) if owner is not None else self.store.list_datasets(kind.value if kind else None)
