"""模型配置。"""

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    provider: str = "openai-compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: int = Field(default=90, ge=1)
    temperature: float = Field(default=0.1, ge=0, le=2)


class ModelProfile(BaseModel):
    """启动时预加载的模型配置；API 返回时只暴露脱敏后的字段。"""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    provider: str = "openai-compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str = Field(min_length=1)
    timeout_seconds: int = Field(default=90, ge=1)
    temperature: float = Field(default=0.1, ge=0, le=2)
    default: bool = False

    def as_config(self) -> ModelConfig:
        return ModelConfig(
            provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
        )
