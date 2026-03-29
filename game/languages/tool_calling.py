from __future__ import annotations

import json
from typing import Any, Protocol

from game.core.core_action import Action, Goal
from game.core.memory import Memory

Prompt = list[dict[str, Any]]


class AgentLanguage(Protocol):
    def construct_prompt(
        self, actions: list[Action], goals: list[Goal], memory: Memory
    ) -> Prompt: ...

    def get_message(self, response: Any) -> Any: ...

    def get_tool_calls(self, message: Any) -> list[dict[str, Any]]: ...


class ToolCallingAgentLanguage:
    def construct_prompt(
        self, actions: list[Action], goals: list[Goal], memory: Memory
    ) -> Prompt:
        goal_text = "\n".join(
            f"- [{goal.priority}] {goal.name}: {goal.description}"
            for goal in sorted(goals, key=lambda goal: goal.priority)
        )

        terminal_actions = [action.name for action in actions if action.terminal]
        terminal_action_text = (
            ", ".join(terminal_actions) if terminal_actions else "none"
        )

        system_prompt = f"""
You are an AI agent operating under the GAME framework.

Goals:
{goal_text}

Terminal actions available:
{terminal_action_text}

Rules:
- Use actions when needed.
- Prefer list_files before reading files unless the exact file is already known.
- Read only relevant files needed to complete the task.
- Avoid unnecessary exploration.
- Do not repeat the same action with the same arguments if it does not make progress.
- Base conclusions on inspected file contents, not filenames alone.
- If a tool returns an error, do not explore alternatives blindly. Use an appropriate terminal action.
- When the task is complete, finish by calling the appropriate terminal action.
- If a structured result-return action is available, use it instead of terminate.
- Do not use terminate to return large structured data when a dedicated result action exists.
- Keep final user-facing messages short when terminate is used.
- Do not paste full file contents into terminate.
- Only call one terminal action when you are ready to finish.
""".strip()

        prompt: Prompt = [{"role": "system", "content": system_prompt}]
        prompt.extend(memory.get_memories())
        return prompt

    def get_message(self, response: Any) -> Any:
        return response.choices[0].message

    def get_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        return [
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "args": self._parse_tool_arguments(tool_call.function.arguments),
            }
            for tool_call in raw_tool_calls
        ]

    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        if raw_args is None or raw_args == "":
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tool arguments JSON: {raw_args}") from exc
        raise ValueError(f"Unsupported tool argument type: {type(raw_args).__name__}")
