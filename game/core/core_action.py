from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable
from game.actions.action_context import ActionContext


@dataclass(frozen=True)
class Goal:
    """A Goal represents a high-level objective for the agent, with a priority to determine its importance relative to other goals."""

    priority: int
    name: str
    description: str


class Action:
    """An Action represents a callable function that the agent can execute, along with metadata for description and parameters."""

    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: dict[str, Any],
        terminal: bool = False,
    ) -> None:
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.terminal = terminal

    def execute(self, action_context: ActionContext | None = None, **args: Any) -> Any:
        signature = inspect.signature(self.function)

        if "action_context" in signature.parameters:
            return self.function(action_context=action_context, **args)

        return self.function(**args)


class ActionRegistry:
    """The ActionRegistry maintains a collection of available actions that the agent can execute, allowing for dynamic registration and retrieval of actions."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        if action.name in self._actions:
            raise ValueError(f"Action '{action.name}' is already registered.")
        self._actions[action.name] = action

    def get_action(self, name: str) -> Action | None:
        return self._actions.get(name)

    def list_actions(self) -> list[Action]:
        return list(self._actions.values())
