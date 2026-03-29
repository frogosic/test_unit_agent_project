from __future__ import annotations

import json
from typing import Any


class Memory:
    def __init__(self) -> None:
        self._memories: list[dict[str, Any]] = []

    def add_memory(self, memory: dict[str, Any]) -> None:
        self._memories.append(memory)

    def add_user_message(self, content: str) -> None:
        self._memories.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(self, message: Any) -> None:
        assistant_memory: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        if raw_tool_calls:
            assistant_memory["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in raw_tool_calls
            ]

        self._memories.append(assistant_memory)

    def add_tool_result(self, tool_call_id: str, result: dict[str, Any]) -> None:
        self._memories.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result),
            }
        )

    def get_memories(self) -> list[dict[str, Any]]:
        return self._memories
