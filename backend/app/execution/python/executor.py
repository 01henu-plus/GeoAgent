"""在 GeoAgent workspace 中运行 Python 脚本。"""

from __future__ import annotations

import re
import sys
import uuid
from threading import Event

from app.execution.process import run_process
from app.execution.sandbox import WorkspaceManager

from .result import PythonExecutionResult


class PythonExecutor:
    def __init__(self, workspace: WorkspaceManager, *, timeout_seconds: int = 120) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str, *, cancel_event: Event | None = None) -> PythonExecutionResult:
        if not code.strip():
            raise ValueError("Python code 不能为空。")
        if re.search(r"(?:\.\.[\\/]|[A-Za-z]:[\\/]|\\\\)", code):
            raise PermissionError("Python 代码中的路径必须位于当前用户 workspace 内。")
        before = self.workspace.snapshot()
        script = self.workspace.temp_dir / f"run_{uuid.uuid4().hex[:10]}.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = run_process([sys.executable, str(script)], cwd=self.workspace.root, timeout_seconds=self.timeout_seconds, cancel_event=cancel_event)
        finally:
            script.unlink(missing_ok=True)
        created = [str(path) for path in self.workspace.discover_new_files(before) if path != script]
        return PythonExecutionResult(returncode=completed.returncode, stdout=completed.stdout[-20000:], stderr=completed.stderr[-20000:], created_files=created)
