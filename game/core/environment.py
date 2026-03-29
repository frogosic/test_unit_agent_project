from __future__ import annotations

import time
import traceback
from typing import Any

from game.actions.action_context import ActionContext
from game.core.core_action import Action


class Environment:
    """Executes actions and returns results in a standardized format."""

    def execute_action(
        self,
        action: Action,
        action_args: dict[str, Any],
        action_context: ActionContext,
    ) -> dict[str, Any]:
        """
        Execute an action with the given arguments and ActionContext.

        The ActionContext is passed through directly so actions can access:
        - current agent identity
        - agent registry
        - llm
        - file security policy
        - future runtime metadata
        """
        try:
            result = action.execute(action_context=action_context, **action_args)
            return self.format_result(result)
        except Exception as exc:
            return {
                "tool_executed": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }

    def format_result(self, result: Any) -> dict[str, Any]:
        """Format an action result into a standardized dictionary."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
