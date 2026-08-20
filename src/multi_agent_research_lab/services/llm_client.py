"""LLM client implementation using Mistral API with typing support and fallback support."""

import logging
from dataclasses import dataclass
from typing import Any

from mistralai.client import Mistral

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client for Mistral API."""

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.mistral_api_key
        self.default_model = default_model or settings.mistral_default_model
        self.client: Any = Mistral(api_key=self.api_key) if self.api_key else None

    def complete(
        self, system_prompt: str, user_prompt: str, model: str | None = None
    ) -> LLMResponse:
        """Return a model completion from Mistral API."""
        model_name = model or self.default_model

        if not self.client:
            logger.warning("No Mistral API key configured. Returning mock LLM completion.")
            mock_content = f"Mock LLM completion for prompt: {user_prompt[:50]}..."
            return LLMResponse(
                content=mock_content, input_tokens=10, output_tokens=20, cost_usd=0.0
            )

        try:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = self.client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=0.3,
            )
            raw_content = ""
            if response and response.choices:
                msg = response.choices[0].message
                if msg and msg.content:
                    raw_content = str(msg.content)

            input_tokens = getattr(response.usage, "prompt_tokens", None) if response else None
            output_tokens = getattr(response.usage, "completion_tokens", None) if response else None

            return LLMResponse(
                content=raw_content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
            )
        except Exception as exc:
            logger.error("Mistral LLM call failed for model %s: %s", model_name, exc)
            fallback_content = (
                f"[Fallback response due to API error: {exc}]\n"
                f"Generated content based on request prompt: {user_prompt}"
            )
            return LLMResponse(
                content=fallback_content, input_tokens=0, output_tokens=0, cost_usd=0.0
            )
