from __future__ import annotations

from typing import Any
from enum import Enum, auto


from game.actions.action_context import ActionContext
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.languages.tool_calling import AgentLanguage
from game.core.llm import LLM, build_tools
from game.core.memory import Memory
from game.policies.file_security_policy import FileSecurityPolicy

import logging

logger = logging.getLogger(__name__)


class StopReason(Enum):
    TERMINAL_ACTION = auto()
    TOOL_ERROR = auto()


class BaseAgent:
    def __init__(
        self,
        name: str,
        goals: list[Goal],
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        llm: LLM,
        environment: Environment,
        file_security_policy: FileSecurityPolicy,
        max_iterations: int = 20,
    ) -> None:
        self.name = name
        self.goals = goals
        self.agent_language = agent_language
        self.action_registry = action_registry
        self.llm = llm
        self.environment = environment
        self.file_security_policy = file_security_policy
        self.max_iterations = max_iterations

    def _call_llm(self, full_prompt: list[dict[str, Any]]) -> Any:
        """Call the LLM with the given prompt and tools."""
        tools = build_tools(self.action_registry)
        return self.llm.generate(full_prompt, tools=tools)

    def _build_execution_context(
        self,
        action_context: ActionContext | None,
    ) -> ActionContext:
        """
        Build the ActionContext used during this agent run.
        """
        if action_context is None:
            return ActionContext(
                props={
                    "current_agent_name": self.name,
                    "llm": self.llm,
                    "file_security_policy": self.file_security_policy,
                }
            )

        return action_context.spawn_delegated_child(
            next_agent_name=self.name,
        )

    def _execute_tool_calls(
        self,
        memory: Memory,
        tool_calls: list[dict[str, Any]],
        action_context: ActionContext,
    ) -> StopReason | None:
        for tool_call in tool_calls:
            action = self.action_registry.get_action(tool_call["name"])

            if action is None:
                memory.add_tool_result(
                    tool_call["id"],
                    {
                        "tool_executed": False,
                        "error": f"Unknown action requested: {tool_call['name']}",
                    },
                )
                return StopReason.TOOL_ERROR

            logger.debug(
                "Executing action: %s with args: %s",
                tool_call["name"],
                tool_call["args"],
            )

            result = self.environment.execute_action(
                action=action,
                action_args=tool_call["args"],
                action_context=action_context,
            )
            logger.debug("Action result: %s", result)

            memory.add_tool_result(tool_call["id"], result)

            if not result.get("tool_executed", False):
                memory.add_memory(
                    {
                        "role": "assistant",
                        "content": f"Stopped due to tool error: {result['error']}",
                    }
                )
                return StopReason.TOOL_ERROR

            if action.terminal:
                return StopReason.TERMINAL_ACTION

        return None

    def _think(self, memory: Memory) -> list[dict[str, Any]]:
        prompt = self.agent_language.construct_prompt(
            actions=self.action_registry.list_actions(),
            goals=self.goals,
            memory=memory,
        )

        logger.debug("Agent thinking...")
        response = self._call_llm(prompt)

        message = self.agent_language.get_message(response)
        memory.add_assistant_message(message)

        tool_calls = self.agent_language.get_tool_calls(message)
        logger.debug("Tool calls: %s", tool_calls)

        return tool_calls

    def run(
        self,
        user_input: str,
        memory: Memory | None = None,
        action_context: ActionContext | None = None,
    ) -> Memory:
        memory = memory or Memory()
        execution_context = self._build_execution_context(action_context)
        memory.add_user_message(user_input)

        for _ in range(self.max_iterations):
            tool_calls = self._think(memory)

            if not tool_calls:
                break

            stop_reason = self._execute_tool_calls(
                memory=memory,
                tool_calls=tool_calls,
                action_context=execution_context,
            )
            if stop_reason is not None:
                logger.debug(
                    "Agent %s stopping — reason: %s",
                    self.name,
                    stop_reason.name,
                )
                break

        return memory
