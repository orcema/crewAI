"""Mistral embeddings provider."""

from typing import Any

from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.mistral_embedding_function import MistralEmbeddingFunction


class MistralProvider(BaseEmbeddingsProvider[MistralEmbeddingFunction]):
    """Mistral embeddings provider."""

    embedding_callable: type[MistralEmbeddingFunction] = Field(
        default=MistralEmbeddingFunction,
        description="Mistral embedding function class",
    )
    api_key: str | None = Field(
        default=None,
        description="Mistral API key",
        validation_alias="MISTRAL_API_KEY",
    )
    model_name: str = Field(
        default="mistral-embed",
        description="Model name to use for embeddings",
        validation_alias="MISTRAL_MODEL_NAME",
    )
    model: str | None = Field(
        default=None,
        description="Model name to use for embeddings (alias for model_name)",
        validation_alias="MISTRAL_MODEL",
    )
    base_url: str = Field(
        default="https://api.mistral.ai/v1",
        description="Base URL for API requests",
        validation_alias="MISTRAL_BASE_URL",
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        validation_alias="MISTRAL_TIMEOUT",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        validation_alias="MISTRAL_MAX_RETRIES",
    )

    def model_post_init(self, __context: Any) -> None:
        """Handle model parameter mapping after initialization."""
        if self.model is not None and self.model != self.model_name:
            self.model_name = self.model

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to exclude the model field."""
        exclude = kwargs.get("exclude", set())
        if isinstance(exclude, set):
            exclude.add("model")
        else:
            exclude = set(exclude) | {"model"}
        kwargs["exclude"] = exclude
        return super().model_dump(**kwargs)