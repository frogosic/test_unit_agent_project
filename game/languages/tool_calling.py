import json
from typing import Any, Protocol

from game.core.core_action import Action, Goal
from game.core.memory import Memory

Prompt = list[dict[str, Any]]


class AgentLanguage(Protocol):
    def construct_prompt(
        self,
        actions: list[Action],
        goals: list[Goal],
        memory: Memory,
    ) -> Prompt: ...

    def get_message(self, response: Any) -> Any: ...

    def get_tool_calls(self, message: Any) -> list[dict[str, Any]]: ...


class ToolCallingAgentLanguage:
    def construct_prompt(
        self,
        actions: list[Action],
        goals: list[Goal],
        memory: Memory,
    ) -> Prompt:
        """Constructs a system prompt for the agent with tool calling capabilities, including goals, available actions, and rules for using tools effectively."""
        goal_text = "\n".join(
            f"- [{goal.priority}] {goal.name}: {goal.description}"
            for goal in sorted(goals, key=lambda goal: goal.priority)
        )

        action_text = json.dumps(
            {
                action.name: {
                    "description": action.description,
                    "parameters": action.parameters,
                    "terminal": action.terminal,
                }
                for action in actions
            },
            indent=2,
        )

        system_prompt = f"""
You are an AI agent operating under the GAME framework.

Goals:
{goal_text}

Available actions:
{action_text}

Rules:
- Use actions when needed.
- Prefer list_files before reading files unless the exact file is already known.
- Read only relevant files to understand the project (e.g., main modules, config, entry points).
- Avoid reading unnecessary files.
- If a tool returns an error, do not explore alternatives. Call terminate.
- If the request cannot be completed with the available actions, call terminate with a short explanation.
- Do not repeat the same action with the same arguments if it does not make progress.
- Call terminate when the task is complete.
- The terminate message must be short and user-facing.
- Include key results in the terminate message when they are small enough.
- Do not paste full file contents or large structured outputs into terminate.
- For large outputs (e.g., README), write to a file using write_file instead of returning in terminate.
- When generating documentation, include a file structure section based on list_files.
- Use the project directory name as the title if available.
""".strip()

        prompt: Prompt = [{"role": "system", "content": system_prompt}]
        prompt.extend(memory.get_memories())
        return prompt

    def get_message(self, response: Any) -> Any:
        """Extracts the message from the LLM response."""
        return response.choices[0].message

    def get_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        """Extracts tool calls from the LLM message."""
        raw_tool_calls = getattr(message, "tool_calls", None) or []

        parsed_tool_calls: list[dict[str, Any]] = []
        for tool_call in raw_tool_calls:
            parsed_tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "args": self._parse_tool_arguments(tool_call.function.arguments),
                }
            )

        return parsed_tool_calls

    def _parse_tool_arguments(self, raw_args: Any) -> dict[str, Any]:
        """Parses the raw tool arguments into a dictionary."""
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
