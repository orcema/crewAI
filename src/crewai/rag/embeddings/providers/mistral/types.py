"""Type definitions for Mistral embedding providers."""

from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


class MistralProviderConfig(TypedDict, total=False):
    """Configuration for Mistral provider."""

    api_key: str
    model_name: Annotated[str, "mistral-embed"]
    base_url: Annotated[str, "https://api.mistral.ai/v1"]
    timeout: Annotated[int, 30]
    max_retries: Annotated[int, 3]


class MistralProviderSpec(TypedDict, total=False):
    """Mistral provider specification."""

    provider: Required[Literal["mistral"]]
    config: MistralProviderConfig