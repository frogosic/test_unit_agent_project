from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from litellm import completion

from game.core.core_action import ActionRegistry

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env"


class LLM:
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate an LLM response, optionally with tool-calling enabled."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        return completion(**request_kwargs)

    def generate_text(self, prompt: list[dict[str, Any]]) -> str:
        """Generate a plain text response without using tools."""
        response = self.generate(prompt=prompt)
        return response.choices[0].message.content or ""


def build_tools(action_registry: ActionRegistry) -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool definitions from the action registry."""
    return [
        {
            "type": "function",
            "function": {
                "name": action.name,
                "description": action.description,
                "parameters": action.parameters,
            },
        }
        for action in action_registry.list_actions()
    ]
