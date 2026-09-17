"""轻量 SQLite 状态仓库。

GeoAgent 第一版不引入 ORM：状态表很少、字段主要是结构化 JSON，直接使用
sqlite3 便于在 CLI、FastAPI 和并行 SubAgent 中共享同一份事实记录。每次操作
使用独立连接，SQLite 的 WAL 模式负责读写并发；GIS 计算本身不在数据库事务里。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from app.core.models import (
    AgentResult,
    Artifact,
    Checkpoint,
    Conversation,
    Dataset,
    MemoryItem,
    Message,
    Run,
    SubTask,
    Task,
    ToolCall,
    ToolResult,
    TraceEvent,
    utc_now,
)

T = TypeVar("T")

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subtasks (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    result_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_lineage (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    operation TEXT NOT NULL,
    input_dataset_ids_json TEXT NOT NULL,
    output_dataset_id TEXT NOT NULL,
    tool_call_id TEXT,
    parameters_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(scope, memory_key)
);
CREATE TABLE IF NOT EXISTS trace_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at);
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_lineage_output ON dataset_lineage(output_dataset_id);
"""


class StateStore:
    """保存 GeoAgent 的可恢复事实和派生索引。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            db.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database_path, timeout=30, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA busy_timeout = 30000")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _json(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def _model(model_type: type[T], value: str) -> T:
        return model_type.model_validate_json(value)  # type: ignore[attr-defined]

    def upsert_conversation(self, conversation_id: str, title: str, timestamp: str) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at""",
                (conversation_id, title, timestamp, timestamp),
            )
            db.commit()

    def create_conversation(self, title: str = "新对话") -> Conversation:
        conversation = Conversation(title=title)
        self.upsert_conversation(conversation.id, conversation.title, conversation.created_at.isoformat())
        return conversation

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (max(1, limit),)).fetchall()
        return [Conversation.model_validate(dict(row)) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return Conversation.model_validate(dict(row)) if row else None

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            db.commit()
        return cursor.rowcount > 0

    def save_message(self, message: Message) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO messages(id,conversation_id,role,content,run_id,created_at) VALUES(?,?,?,?,?,?)",
                (message.id, message.conversation_id, message.role, message.content, message.run_id, message.created_at.isoformat()),
            )
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (message.created_at.isoformat(), message.conversation_id))
            db.commit()

    def list_messages(self, conversation_id: str, limit: int = 100) -> list[Message]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at LIMIT ?",
                (conversation_id, max(1, limit)),
            ).fetchall()
        return [Message.model_validate(dict(row)) for row in rows]

    def save_task(self, task: Task) -> None:
        payload = task.model_dump_json()
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO tasks(id,payload_json,updated_at) VALUES(?,?,?)",
                (task.id, payload, task.updated_at.isoformat()),
            )
            db.commit()

    def get_task(self, task_id: str) -> Task | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._model(Task, row[0]) if row else None

    def list_tasks(self, conversation_id: str | None = None, limit: int = 50) -> list[Task]:
        query = "SELECT payload_json FROM tasks"
        args: tuple[Any, ...] = ()
        if conversation_id:
            query += " WHERE json_extract(payload_json, '$.conversation_id')=?"
            args = (conversation_id,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args += (max(1, limit),)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(Task, row[0]) for row in rows]

    def save_subtask(self, task_id: str, subtask: SubTask) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO subtasks(id,task_id,payload_json) VALUES(?,?,?)",
                (subtask.id, task_id, subtask.model_dump_json()),
            )
            db.commit()

    def save_run(self, run: Run) -> None:
        timestamp = utc_now().isoformat()
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO runs(id,payload_json,updated_at) VALUES(?,?,?)",
                (run.id, run.model_dump_json(), timestamp),
            )
            db.commit()

    def get_run(self, run_id: str) -> Run | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._model(Run, row[0]) if row else None

    def list_runs(self, limit: int = 50) -> list[Run]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM runs ORDER BY updated_at DESC LIMIT ?", (max(1, limit),)).fetchall()
        return [self._model(Run, row[0]) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        return bool(self.delete_runs([run_id]))

    def delete_runs(self, run_ids: list[str]) -> list[str]:
        deleted: list[str] = []
        with self._connect() as db:
            for run_id in dict.fromkeys(run_ids):
                row = db.execute("SELECT payload_json FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    continue
                run = self._model(Run, row[0])
                db.execute("DELETE FROM trace_events WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM checkpoints WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM tool_calls WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM dataset_lineage WHERE run_id=?", (run_id,))
                if run.task_id:
                    db.execute("DELETE FROM subtasks WHERE task_id=?", (run.task_id,))
                    db.execute("DELETE FROM tasks WHERE id=?", (run.task_id,))
                db.execute("DELETE FROM runs WHERE id=?", (run_id,))
                deleted.append(run_id)
            db.commit()
        return deleted

    def save_tool_call(self, call: ToolCall, result: ToolResult | None = None) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO tool_calls(id,run_id,name,arguments_json,result_json,created_at)
                VALUES(?,?,?,?,?,?)""",
                (call.id, call.run_id, call.name, self._json(call.arguments), result.model_dump_json() if result else None, utc_now().isoformat()),
            )
            db.commit()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO checkpoints(id,run_id,phase,payload_json,created_at) VALUES(?,?,?,?,?)",
                (checkpoint.id, checkpoint.run_id, checkpoint.phase, checkpoint.model_dump_json(), checkpoint.created_at.isoformat()),
            )
            db.commit()

    def latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload_json FROM checkpoints WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)
            ).fetchone()
        return self._model(Checkpoint, row[0]) if row else None

    def save_dataset(self, dataset: Dataset) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO datasets(id,payload_json,created_at) VALUES(?,?,?)",
                (dataset.id, dataset.model_dump_json(), dataset.created_at.isoformat()),
            )
            db.commit()

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._model(Dataset, row[0]) if row else None

    def list_datasets(self, kind: str | None = None) -> list[Dataset]:
        query = "SELECT payload_json FROM datasets"
        args: tuple[Any, ...] = ()
        if kind:
            query += " WHERE json_extract(payload_json, '$.kind')=?"
            args = (kind,)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(Dataset, row[0]) for row in rows]

    def save_lineage(
        self,
        *,
        lineage_id: str,
        run_id: str | None,
        operation: str,
        input_dataset_ids: list[str],
        output_dataset_id: str,
        tool_call_id: str | None,
        parameters: dict[str, Any],
        created_at: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO dataset_lineage
                (id,run_id,operation,input_dataset_ids_json,output_dataset_id,tool_call_id,parameters_json,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (lineage_id, run_id, operation, self._json(input_dataset_ids), output_dataset_id, tool_call_id, self._json(parameters), created_at),
            )
            db.commit()

    def list_lineage(self, output_dataset_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM dataset_lineage"
        args: tuple[Any, ...] = ()
        if output_dataset_id:
            query += " WHERE output_dataset_id=?"
            args = (output_dataset_id,)
        query += " ORDER BY created_at"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "operation": row["operation"],
                "input_dataset_ids": json.loads(row["input_dataset_ids_json"]),
                "output_dataset_id": row["output_dataset_id"],
                "tool_call_id": row["tool_call_id"],
                "parameters": json.loads(row["parameters_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_artifact(self, artifact: Artifact) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO artifacts(id,payload_json,created_at) VALUES(?,?,?)",
                (artifact.id, artifact.model_dump_json(), artifact.created_at.isoformat()),
            )
            db.commit()

    def list_artifacts(self, run_id: str | None = None) -> list[Artifact]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM artifacts ORDER BY created_at DESC").fetchall()
        items = [self._model(Artifact, row[0]) for row in rows]
        return [item for item in items if run_id is None or item.run_id == run_id]

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return self._model(Artifact, row[0]) if row else None

    def save_memory(self, memory: MemoryItem) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO memories(id,scope,memory_key,payload_json,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(scope,memory_key) DO UPDATE SET id=excluded.id,payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (memory.id, memory.scope, memory.key, memory.model_dump_json(), memory.updated_at.isoformat()),
            )
            db.commit()

    def list_memories(self, scope: str = "project") -> list[MemoryItem]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM memories WHERE scope=? ORDER BY updated_at DESC", (scope,)).fetchall()
        return [self._model(MemoryItem, row[0]) for row in rows]

    def record_event(self, event: TraceEvent) -> TraceEvent:
        with self._connect() as db:
            row = db.execute("SELECT COALESCE(MAX(sequence), -1) FROM trace_events WHERE run_id=?", (event.run_id,)).fetchone()
            sequence = max(event.sequence, int(row[0]) + 1)
            event = event.model_copy(update={"sequence": sequence})
            db.execute(
                "INSERT OR IGNORE INTO trace_events(id,run_id,sequence,payload_json) VALUES(?,?,?,?)",
                (event.id, event.run_id, event.sequence, event.model_dump_json()),
            )
            db.commit()
        return event

    def list_events(self, run_id: str) -> list[TraceEvent]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM trace_events WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
        return [self._model(TraceEvent, row[0]) for row in rows]

    def save_agent_result(self, result: AgentResult) -> None:
        """把最终 AgentResult 放入对应 Run 的 metadata，便于恢复后读取。"""

        run = self.get_run(result.trace_id)
        if run:
            self.save_run(run.model_copy(update={"metadata": {**run.metadata, "result": result.model_dump(mode="json")}}))


__all__ = ["StateStore"]
