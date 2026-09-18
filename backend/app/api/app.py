"""FastAPI 接口：Chat、Dataset、Run、Trace 和 Artifact。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.application import Application
from app.core.models import AgentRequest, AgentResult, User, UserView, new_id


class AskBody(BaseModel):
    message: str
    conversation_id: str | None = None
    model_profile: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)
    referenced_run_ids: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class ConversationBody(BaseModel):
    title: str = "新对话"


class DatasetBody(BaseModel):
    path: str
    name: str | None = None


class MemoryBody(BaseModel):
    key: str
    value: str
    scope: str = "project"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunDeleteBody(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


class RegisterBody(BaseModel):
    username: str
    password: str
    email: str | None = None
    display_name: str | None = None


class LoginBody(BaseModel):
    identifier: str | None = None
    username: str | None = None
    email: str | None = None
    password: str


class UserUpdateBody(BaseModel):
    display_name: str | None = None
    email: str | None = None


def get_current_user(request: Request) -> User:
    geoagent = request.app.state.geoagent
    user = geoagent.auth.authenticate_token(request.cookies.get(geoagent.settings.auth_cookie_name))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def _set_session_cookie(response: Response, geoagent: Application, token: str) -> None:
    response.set_cookie(
        geoagent.settings.auth_cookie_name,
        token,
        httponly=True,
        secure=geoagent.settings.auth_cookie_secure,
        samesite="lax",
        max_age=geoagent.settings.auth_session_ttl_hours * 3600,
        path="/",
    )


def create_app(application: Application | None = None) -> FastAPI:
    geoagent = application or Application()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        geoagent.start()
        yield
        await geoagent.close()

    api = FastAPI(title="GeoAgent API", version="0.1.0", lifespan=lifespan)
    api.state.geoagent = geoagent

    @api.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "geoagent", "tools": len(geoagent.tool_registry.names()), "model_configured": bool(geoagent.model_adapters)}

    @api.post("/api/v1/auth/register")
    async def register(body: RegisterBody, response: Response) -> dict[str, Any]:
        try:
            user = geoagent.auth.register(body.username, body.password, email=body.email, display_name=body.display_name)
            _, token = geoagent.auth.login(body.username, body.password)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _set_session_cookie(response, geoagent, token)
        return UserView.from_user(user).model_dump(mode="json")

    @api.post("/api/v1/auth/login")
    async def login(body: LoginBody, response: Response) -> dict[str, Any]:
        try:
            identifier = body.identifier or body.username or body.email or ""
            user, token = geoagent.auth.login(identifier, body.password)
        except Exception as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        _set_session_cookie(response, geoagent, token)
        return UserView.from_user(user).model_dump(mode="json")

    @api.post("/api/v1/auth/logout")
    async def logout(request: Request, response: Response) -> dict[str, bool]:
        geoagent.auth.logout(request.cookies.get(geoagent.settings.auth_cookie_name))
        response.delete_cookie(geoagent.settings.auth_cookie_name, path="/")
        return {"logged_out": True}

    @api.get("/api/v1/users/me")
    async def current_user(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        return UserView.from_user(current_user).model_dump(mode="json")

    @api.patch("/api/v1/users/me")
    async def update_current_user(body: UserUpdateBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        try:
            updated = geoagent.auth.update_user(current_user, display_name=body.display_name or current_user.display_name, email=body.email)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UserView.from_user(updated).model_dump(mode="json")

    @api.get("/api/v1/models")
    async def model_status(_: User = Depends(get_current_user)) -> dict[str, object]:
        return geoagent.model_status()

    @api.get("/api/v1/metrics")
    async def metrics(_: User = Depends(get_current_user)) -> dict[str, int]:
        return geoagent.metrics.snapshot()

    @api.get("/api/v1/tools")
    async def tools(_: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.tool_registry.definitions()]

    @api.get("/api/v1/datasets")
    async def datasets(current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.registry.list(user_id=current_user.id)]

    @api.post("/api/v1/datasets")
    async def register_dataset(body: DatasetBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        try:
            dataset = geoagent.register_dataset(body.path, name=body.name, user_id=current_user.id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return dataset.model_dump(mode="json")

    @api.post("/api/v1/attachments")
    async def upload_attachment(file: UploadFile = File(...), current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        """接收用户文件，保存到当前用户 workspace/input 并立即登记为 Dataset。"""
        filename = file.filename or ""
        try:
            content = await file.read()
            path = geoagent.attachments.accept(filename, content, user_id=current_user.id)
            dataset = geoagent.registry.for_user(current_user.id).register_path(path, name=Path(filename).stem or path.stem)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "attachment_id": dataset.id,
            "dataset": dataset.model_dump(mode="json"),
        }

    @api.get("/api/v1/datasets/{dataset_id}/lineage")
    async def dataset_lineage(dataset_id: str, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if geoagent.registry.get(dataset_id, user_id=current_user.id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return geoagent.store.list_lineage(dataset_id)

    @api.post("/api/v1/ask")
    async def ask(body: AskBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        _ensure_body_conversation_access(geoagent, body, current_user)
        try:
            result = await geoagent.ask(_request_from_body(body, user_id=current_user.id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @api.post("/api/v1/runs")
    async def create_run(body: AskBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        _ensure_body_conversation_access(geoagent, body, current_user)
        request = _request_from_body(body, user_id=current_user.id)
        try:
            run = await geoagent.conversations.submit(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return run.model_dump(mode="json")

    @api.post("/api/v1/conversations/{conversation_id}/messages")
    async def conversation_message(conversation_id: str, body: AskBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        existing = geoagent.store.get_conversation(conversation_id)
        if existing is not None and geoagent.store.get_conversation_for_user(conversation_id, current_user.id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        try:
            result = await geoagent.ask(_request_from_body(body, conversation_id=conversation_id, user_id=current_user.id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @api.get("/api/v1/conversations")
    async def conversations(limit: int = 50, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.conversations.list(limit, user_id=current_user.id)]

    @api.post("/api/v1/conversations")
    async def create_conversation(body: ConversationBody | None = None, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        title = body.title.strip() if body and body.title.strip() else "新对话"
        return geoagent.conversations.create(title, user_id=current_user.id).model_dump(mode="json")

    @api.delete("/api/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str, current_user: User = Depends(get_current_user)) -> dict[str, bool]:
        if not geoagent.conversations.delete(conversation_id, user_id=current_user.id):
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"deleted": True}

    @api.get("/api/v1/conversations/{conversation_id}/messages")
    async def conversation_messages(conversation_id: str, limit: int = 100, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if geoagent.store.get_conversation_for_user(conversation_id, current_user.id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return [item.model_dump(mode="json") for item in geoagent.store.list_messages(conversation_id, limit)]

    @api.get("/api/v1/runs")
    async def runs(limit: int = 50, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.store.list_runs(limit, user_id=current_user.id)]

    @api.get("/api/v1/runs/{run_id}")
    async def run(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        item = geoagent.store.get_run(run_id) if geoagent.store.run_belongs_to_user(run_id, current_user.id) else None
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        return item.model_dump(mode="json")

    @api.delete("/api/v1/runs")
    async def delete_runs(body: RunDeleteBody, current_user: User = Depends(get_current_user)) -> dict[str, list[str]]:
        requested = list(dict.fromkeys(run_id.strip() for run_id in body.run_ids if run_id.strip()))
        if any(not geoagent.store.run_belongs_to_user(run_id, current_user.id) for run_id in requested):
            raise HTTPException(status_code=404, detail="run not found")
        active = [run_id for run_id in requested if geoagent.run_manager.is_active(run_id)]
        deleted = geoagent.store.delete_runs([run_id for run_id in requested if run_id not in active])
        for run_id in deleted:
            geoagent.run_manager.forget(run_id)
        return {"deleted": deleted, "skipped_active": active}

    @api.delete("/api/v1/runs/{run_id}")
    async def delete_run(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, bool]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        if geoagent.run_manager.is_active(run_id):
            raise HTTPException(status_code=409, detail="运行中的记录不能删除，请先取消运行")
        if not geoagent.store.delete_run(run_id):
            raise HTTPException(status_code=404, detail="run not found")
        geoagent.run_manager.forget(run_id)
        return {"deleted": True}

    @api.post("/api/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        if not await geoagent.run_manager.cancel(run_id):
            raise HTTPException(status_code=409, detail="run is not active")
        item = geoagent.store.get_run(run_id)
        return item.model_dump(mode="json") if item else {"id": run_id, "status": "CANCELLED"}

    @api.get("/api/v1/runs/{run_id}/events")
    async def events(run_id: str, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        return [item.model_dump(mode="json") for item in geoagent.store.list_events(run_id)]

    @api.get("/api/v1/artifacts")
    async def artifacts(run_id: str | None = None, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if run_id is not None and not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        return [item.model_dump(mode="json") for item in geoagent.store.list_artifacts_for_user(current_user.id, run_id)]

    @api.get("/api/v1/artifacts/{artifact_id}/content")
    async def artifact_content(artifact_id: str, current_user: User = Depends(get_current_user)):
        artifact = geoagent.store.get_artifact_for_user(artifact_id, current_user.id)
        if artifact is None or not artifact.path:
            raise HTTPException(status_code=404, detail="artifact not found")
        try:
            workspace = geoagent.workspace.for_user(artifact.owner_user_id)
            path = workspace.resolve(artifact.path, allow_missing=False)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="artifact file not found") from exc
        return FileResponse(path, media_type=artifact.media_type, filename=artifact.name)

    @api.get("/api/v1/memories")
    async def memories(scope: str = "project", current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.memory.list(scope, user_id=current_user.id)]

    @api.post("/api/v1/memories")
    async def save_memory(body: MemoryBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        if not body.key.strip() or not body.scope.strip():
            raise HTTPException(status_code=400, detail="memory key and scope cannot be empty")
        item = geoagent.memory.set(body.key.strip(), body.value, scope=body.scope.strip(), metadata=body.metadata, user_id=current_user.id)
        return item.model_dump(mode="json")

    @api.get("/api/v1/runs/{run_id}/checkpoint")
    async def checkpoint(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        item = geoagent.checkpoints.latest(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="checkpoint not found")
        return item.model_dump(mode="json")

    @api.post("/api/v1/runs/{run_id}/resume")
    async def resume(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        previous = geoagent.store.get_run(run_id)
        if previous is None or not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        checkpoint = geoagent.checkpoints.latest(run_id)
        if checkpoint is None:
            raise HTTPException(status_code=409, detail="run has no checkpoint")
        if checkpoint.phase == "run_completed" and isinstance(checkpoint.state.get("result"), dict):
            result = AgentResult.model_validate(checkpoint.state["result"])
            return {"resumed_from": run_id, "run_id": run_id, "checkpoint": checkpoint.id, "result": result.model_dump(mode="json")}
        saved_request = checkpoint.state.get("request")
        if not isinstance(saved_request, dict):
            goal = str(previous.metadata.get("goal", "继续上一次 GIS 任务"))
            saved_request = {"user_input": goal, "conversation_id": previous.conversation_id or new_id("conv")}
        request = AgentRequest.model_validate(saved_request).model_copy(update={"user_id": current_user.id})
        has_saved_plan = bool(checkpoint.state.get("intent") and checkpoint.state.get("plan"))
        has_model_context = isinstance(checkpoint.state.get("messages"), list) and bool(checkpoint.state["messages"])
        if not has_saved_plan and not has_model_context:
            raise HTTPException(status_code=409, detail="checkpoint 还没有可恢复的上下文")
        run = await geoagent.run_manager.submit(request, resume_from=checkpoint, metadata={"resumed_from": run_id})
        result = await geoagent.conversations.wait(run.id)
        return {"resumed_from": run_id, "run_id": run.id, "checkpoint": checkpoint.id, "result": result.model_dump(mode="json")}

    @api.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        current_user = geoagent.auth.authenticate_token(websocket.cookies.get(geoagent.settings.auth_cookie_name))
        if current_user is None:
            await websocket.close(code=1008, reason="请先登录")
            return
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                if payload.get("type") != "ask":
                    await websocket.send_json({"type": "error", "message": "只支持 type=ask"})
                    continue
                request = AgentRequest(
                    user_input=payload.get("message", ""),
                    conversation_id=payload.get("conversation_id") or new_id("conv"),
                    user_id=current_user.id,
                    model_profile=payload.get("model_profile"),
                    dataset_ids=payload.get("dataset_ids", []),
                    attachment_ids=payload.get("attachment_ids", []),
                    referenced_run_ids=payload.get("referenced_run_ids", []),
                    context=payload.get("context", {}),
                )
                event_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

                async def on_event(event) -> None:
                    if event.run_id == run.id:
                        event_queue.put_nowait(("event", event))

                async def on_model_delta(content: str) -> None:
                    event_queue.put_nowait(("delta", content))

                run = await geoagent.conversations.submit(request, on_model_delta=on_model_delta)

                geoagent.bus.subscribe(on_event)
                waiter = asyncio.create_task(geoagent.conversations.wait(run.id))
                event_waiter = asyncio.create_task(event_queue.get())
                try:
                    await websocket.send_json({"type": "run", "data": run.model_dump(mode="json")})
                    while True:
                        done, _ = await asyncio.wait((waiter, event_waiter), return_when=asyncio.FIRST_COMPLETED)
                        if event_waiter in done:
                            kind, item = event_waiter.result()
                            if kind == "event":
                                await websocket.send_json({"type": "event", "data": item.model_dump(mode="json")})
                            else:
                                await websocket.send_json({"type": "delta", "content": item})
                            event_waiter = asyncio.create_task(event_queue.get())
                            continue
                        result = waiter.result()
                        while not event_queue.empty():
                            kind, item = event_queue.get_nowait()
                            if kind == "event":
                                await websocket.send_json({"type": "event", "data": item.model_dump(mode="json")})
                            else:
                                await websocket.send_json({"type": "delta", "content": item})
                        await websocket.send_json({"type": "result", "data": result.model_dump(mode="json")})
                        break
                finally:
                    geoagent.bus.unsubscribe(on_event)
                    if not event_waiter.done():
                        event_waiter.cancel()
                    if not waiter.done():
                        await geoagent.run_manager.cancel(run.id)
        except WebSocketDisconnect:
            return

    return api


def _request_from_body(body: AskBody, *, conversation_id: str | None = None, user_id: str | None = None) -> AgentRequest:
    return AgentRequest(
        user_input=body.message,
        conversation_id=conversation_id or body.conversation_id or new_id("conv"),
        user_id=user_id,
        model_profile=body.model_profile,
        dataset_ids=body.dataset_ids,
        attachment_ids=body.attachment_ids,
        referenced_run_ids=body.referenced_run_ids,
        context=body.context,
    )


def _ensure_body_conversation_access(geoagent: Application, body: AskBody, user: User) -> None:
    if body.conversation_id and geoagent.store.get_conversation(body.conversation_id) is not None and geoagent.store.get_conversation_for_user(body.conversation_id, user.id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
