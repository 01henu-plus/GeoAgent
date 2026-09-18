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
    User,
    UserSession,
    WorkingMemory,
    utc_now,
)

T = TypeVar("T")

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    user_id TEXT,
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
CREATE TABLE IF NOT EXISTS working_memories (
    task_id TEXT PRIMARY KEY,
    conversation_id TEXT,
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
    owner_user_id TEXT,
    scope TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_user_id, scope, memory_key)
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT UNIQUE,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT
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
CREATE INDEX IF NOT EXISTS idx_working_memories_conversation ON working_memories(conversation_id, updated_at);
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
            self._migrate_schema(db)
            db.commit()

    @staticmethod
    def _migrate_schema(db: sqlite3.Connection) -> None:
        """为已有本地 SQLite 增加身份字段，不删除旧业务数据。"""

        conversation_columns = {row[1] for row in db.execute("PRAGMA table_info(conversations)").fetchall()}
        if "user_id" not in conversation_columns:
            db.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
        memory_columns = {row[1] for row in db.execute("PRAGMA table_info(memories)").fetchall()}
        if "owner_user_id" not in memory_columns:
            db.execute("ALTER TABLE memories RENAME TO memories_legacy")
            db.execute(
                """CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT,
                scope TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_user_id, scope, memory_key)
                )"""
            )
            db.execute(
                """INSERT INTO memories(id,owner_user_id,scope,memory_key,payload_json,updated_at)
                SELECT id,NULL,scope,memory_key,payload_json,updated_at FROM memories_legacy"""
            )
            db.execute("DROP TABLE memories_legacy")
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_user_id, scope, updated_at)")

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

    def save_user(self, user: User) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO users(id,username,email,payload_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?)""",
                (user.id, user.username, user.email, user.model_dump_json(), user.created_at.isoformat(), user.updated_at.isoformat()),
            )
            db.commit()

    def count_users(self) -> int:
        with self._connect() as db:
            row = db.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0])

    def get_user(self, user_id: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM users WHERE id=?", (user_id,)).fetchone()
        return self._model(User, row[0]) if row else None

    def get_user_by_username(self, username: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM users WHERE username=?", (username.casefold(),)).fetchone()
        return self._model(User, row[0]) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM users WHERE email=?", (email.casefold(),)).fetchone()
        return self._model(User, row[0]) if row else None

    def save_session(self, session: UserSession) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO sessions(id,user_id,token_hash,payload_json,expires_at,created_at,last_seen_at)
                VALUES(?,?,?,?,?,?,?)""",
                (session.id, session.user_id, session.token_hash, session.model_dump_json(), session.expires_at.isoformat(), session.created_at.isoformat(), session.last_seen_at.isoformat() if session.last_seen_at else None),
            )
            db.commit()

    def get_session(self, token_hash: str) -> UserSession | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM sessions WHERE token_hash=?", (token_hash,)).fetchone()
        return self._model(UserSession, row[0]) if row else None

    def touch_session(self, session_id: str, timestamp) -> None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row:
                session = self._model(UserSession, row[0]).model_copy(update={"last_seen_at": timestamp})
                db.execute("UPDATE sessions SET payload_json=?, last_seen_at=? WHERE id=?", (session.model_dump_json(), timestamp.isoformat(), session_id))
                db.commit()

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            db.commit()
        return cursor.rowcount > 0

    def delete_session_by_token(self, token_hash: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            db.commit()
        return cursor.rowcount > 0

    def upsert_conversation(self, conversation_id: str, title: str, timestamp: str, user_id: str | None = None) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO conversations(id,title,user_id,created_at,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                user_id=COALESCE(conversations.user_id, excluded.user_id), updated_at=excluded.updated_at""",
                (conversation_id, title, user_id, timestamp, timestamp),
            )
            db.commit()

    def create_conversation(self, title: str = "新对话", *, user_id: str | None = None) -> Conversation:
        conversation = Conversation(title=title, user_id=user_id)
        self.upsert_conversation(conversation.id, conversation.title, conversation.created_at.isoformat(), user_id)
        return conversation

    def list_conversations(self, limit: int = 50, *, user_id: str | None = None) -> list[Conversation]:
        query = "SELECT * FROM conversations"
        args: tuple[Any, ...] = ()
        if user_id is not None:
            query += " WHERE user_id=?"
            args = (user_id,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args += (max(1, limit),)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [Conversation.model_validate(dict(row)) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return Conversation.model_validate(dict(row)) if row else None

    def get_conversation_for_user(self, conversation_id: str, user_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        return Conversation.model_validate(dict(row)) if row else None

    def delete_conversation(self, conversation_id: str, *, user_id: str | None = None) -> bool:
        with self._connect() as db:
            if user_id is None:
                cursor = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            else:
                cursor = db.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id))
            if cursor.rowcount:
                db.execute("DELETE FROM working_memories WHERE conversation_id=?", (conversation_id,))
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

    def task_belongs_to_user(self, task_id: str, user_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                """SELECT 1 FROM tasks t JOIN conversations c
                ON json_extract(t.payload_json, '$.conversation_id')=c.id
                WHERE t.id=? AND c.user_id=?""",
                (task_id, user_id),
            ).fetchone()
        return row is not None

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

    def save_working_memory(self, memory: WorkingMemory) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO working_memories(task_id,conversation_id,payload_json,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET conversation_id=excluded.conversation_id,
                payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
                (memory.task_id, memory.conversation_id, memory.model_dump_json(), memory.updated_at.isoformat()),
            )
            db.commit()

    def get_working_memory(self, task_id: str) -> WorkingMemory | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM working_memories WHERE task_id=?", (task_id,)).fetchone()
        return self._model(WorkingMemory, row[0]) if row else None

    def delete_working_memory(self, task_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM working_memories WHERE task_id=?", (task_id,))
            db.commit()
        return cursor.rowcount > 0

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

    def run_belongs_to_user(self, run_id: str, user_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                """SELECT 1 FROM runs r JOIN conversations c
                ON json_extract(r.payload_json, '$.conversation_id')=c.id
                WHERE r.id=? AND c.user_id=?""",
                (run_id, user_id),
            ).fetchone()
        return row is not None

    def user_id_for_run(self, run_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT c.user_id FROM runs r JOIN conversations c
                ON json_extract(r.payload_json, '$.conversation_id')=c.id
                WHERE r.id=?""",
                (run_id,),
            ).fetchone()
        return row[0] if row else None

    def list_runs(self, limit: int = 50, *, user_id: str | None = None) -> list[Run]:
        query = "SELECT r.payload_json FROM runs r"
        args: tuple[Any, ...] = ()
        if user_id is not None:
            query += " JOIN conversations c ON json_extract(r.payload_json, '$.conversation_id')=c.id WHERE c.user_id=?"
            args = (user_id,)
        query += " ORDER BY r.updated_at DESC LIMIT ?"
        args += (max(1, limit),)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(Run, row[0]) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        return bool(self.delete_runs([run_id]))

    def delete_runs(self, run_ids: list[str]) -> list[str]:
        deleted: list[str] = []
        task_ids: set[str] = set()
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
                    task_ids.add(run.task_id)
                db.execute("DELETE FROM runs WHERE id=?", (run_id,))
                deleted.append(run_id)
            for task_id in task_ids:
                remaining = db.execute("SELECT 1 FROM runs WHERE json_extract(payload_json, '$.task_id')=? LIMIT 1", (task_id,)).fetchone()
                if remaining is None:
                    db.execute("DELETE FROM subtasks WHERE task_id=?", (task_id,))
                    db.execute("DELETE FROM working_memories WHERE task_id=?", (task_id,))
                    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
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

    def get_dataset_for_user(self, dataset_id: str, user_id: str) -> Dataset | None:
        dataset = self.get_dataset(dataset_id)
        return dataset if dataset is not None and dataset.owner_user_id in {None, user_id} else None

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

    def list_datasets_for_user(self, user_id: str, kind: str | None = None) -> list[Dataset]:
        return [item for item in self.list_datasets(kind) if item.owner_user_id in {None, user_id}]

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

    def list_artifacts_for_user(self, user_id: str, run_id: str | None = None) -> list[Artifact]:
        return [item for item in self.list_artifacts(run_id) if item.owner_user_id in {None, user_id}]

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return self._model(Artifact, row[0]) if row else None

    def get_artifact_for_user(self, artifact_id: str, user_id: str) -> Artifact | None:
        artifact = self.get_artifact(artifact_id)
        return artifact if artifact is not None and artifact.owner_user_id in {None, user_id} else None

    def save_memory(self, memory: MemoryItem) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO memories(id,owner_user_id,scope,memory_key,payload_json,updated_at) VALUES(?,?,?,?,?,?)
                ON CONFLICT(owner_user_id,scope,memory_key) DO UPDATE SET id=excluded.id,
                payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (memory.id, memory.owner_user_id, memory.scope, memory.key, memory.model_dump_json(), memory.updated_at.isoformat()),
            )
            db.commit()

    def list_memories(self, scope: str = "project", *, owner_user_id: str | None = None) -> list[MemoryItem]:
        query = "SELECT payload_json FROM memories WHERE scope=?"
        args: tuple[Any, ...] = (scope,)
        if owner_user_id is not None:
            query += " AND owner_user_id=?"
            args += (owner_user_id,)
        query += " ORDER BY updated_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
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
