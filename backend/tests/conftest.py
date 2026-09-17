from collections.abc import Iterator

import pytest

from app.application import Application
from app.config import Settings


@pytest.fixture
def application(tmp_path) -> Iterator[Application]:
    settings = Settings(root=tmp_path, database=tmp_path / "state.sqlite3", workspace=tmp_path / "workspace")
    app = Application(settings)
    app.start()
    try:
        yield app
    finally:
        # 无模型客户端时 close 也可以同步结束；测试不启动网络服务。
        import asyncio

        asyncio.run(app.close())

