"""OpenAI-compatible Chat Completions Adapter。"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.models.adapter import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
from app.models.config import ModelConfig


class OpenAICompatibleAdapter(ModelAdapter):
    supports_structured_output = True

    def __init__(self, config: ModelConfig) -> None:
        if not config.model or not config.model.strip():
            raise ValueError("OpenAI 兼容接口需要填写模型名称。")
        self.config = config
        # 本地 Ollama、vLLM 等兼容服务通常不校验 API Key；官方或云端服务
        # 仍由用户在配置中填写真实密钥。OpenAI SDK 要求传入非空字符串，
        # 因此对无密钥的本地服务使用占位值，不会把它发送为业务凭据。
        base_url = config.base_url.strip() if config.base_url and config.base_url.strip() else None
        self.client = AsyncOpenAI(api_key=config.api_key or "local", base_url=base_url, timeout=config.timeout_seconds)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=request.messages,
            tools=request.tools or None,
            response_format=request.response_format,
            temperature=request.temperature if request.temperature is not None else self.config.temperature,
            max_tokens=request.max_tokens,
        )
        message = response.choices[0].message
        return ModelResponse(content=message.content or "", tool_calls=[call.model_dump() for call in (message.tool_calls or [])], input_tokens=response.usage.prompt_tokens if response.usage else 0, output_tokens=response.usage.completion_tokens if response.usage else 0, model=response.model)

    async def stream(self, request: ModelRequest):
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=request.messages,
            tools=request.tools or None,
            temperature=request.temperature if request.temperature is not None else self.config.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        tool_calls: dict[int, dict] = {}
        model = None
        input_tokens = 0
        output_tokens = 0
        async for chunk in stream:
            model = getattr(chunk, "model", None) or model
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                input_tokens = usage.prompt_tokens or 0
                output_tokens = usage.completion_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                yield ModelStreamChunk(content=content, model=model)
            for raw_call in delta.tool_calls or []:
                index = raw_call.index if raw_call.index is not None else len(tool_calls)
                call = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if raw_call.id:
                    call["id"] = raw_call.id
                if raw_call.type:
                    call["type"] = raw_call.type
                if raw_call.function:
                    if raw_call.function.name:
                        call["function"]["name"] += raw_call.function.name
                    if raw_call.function.arguments:
                        call["function"]["arguments"] += raw_call.function.arguments
        yield ModelStreamChunk(content="", tool_calls=[tool_calls[index] for index in sorted(tool_calls)], input_tokens=input_tokens, output_tokens=output_tokens, model=model, done=True)

    async def close(self) -> None:
        await self.client.close()
