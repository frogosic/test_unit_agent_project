from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.core.agent_registry import AgentRegistry


@dataclass(slots=True)
class ActionContext:
    """
    Runtime context passed into actions/tools.

    This object is intentionally lightweight. It stores runtime properties
    needed during execution and exposes a few helper methods for common
    access patterns.

    Typical contents may include:
    - auth/config values
    - logger or run metadata
    - agent registry
    - current agent name
    - delegation depth
    - delegation path / visited agents
    - relevant memory for delegated child agents
    """

    props: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a property from the context.
        """
        return self.props.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set or overwrite a property in the context.
        """
        self.props[key] = value

    def require(self, key: str) -> Any:
        """
        Retrieve a required property or raise an error if missing.

        Raises:
            ValueError: If the key does not exist in the context.
        """
        if key not in self.props:
            raise ValueError(f"Missing required ActionContext property: '{key}'.")
        return self.props[key]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a shallow copy of the underlying properties.
        """
        return dict(self.props)

    def get_agent_registry(self) -> AgentRegistry | None:
        """
        Return the configured agent registry if present.
        """
        registry = self.get("agent_registry")
        if registry is None:
            return None

        if not isinstance(registry, AgentRegistry):
            raise ValueError(
                "ActionContext property 'agent_registry' is not an AgentRegistry."
            )

        return registry

    def require_agent_registry(self) -> AgentRegistry:
        """
        Return the configured agent registry or raise an error if missing.

        Raises:
            ValueError: If no registry is configured.
        """
        registry = self.get_agent_registry()
        if registry is None:
            raise ValueError("No agent registry found in ActionContext.")
        return registry

    def get_current_agent_name(self) -> str | None:
        """
        Return the current agent name if present.
        """
        agent_name = self.get("current_agent_name")
        if agent_name is None:
            return None

        if not isinstance(agent_name, str):
            raise ValueError(
                "ActionContext property 'current_agent_name' must be a string."
            )

        normalized_name = agent_name.strip()
        if not normalized_name:
            raise ValueError(
                "ActionContext property 'current_agent_name' must not be empty."
            )

        return normalized_name

    def require_current_agent_name(self) -> str:
        """
        Return the current agent name or raise an error if missing.

        Raises:
            ValueError: If the current agent name is not configured.
        """
        agent_name = self.get_current_agent_name()
        if agent_name is None:
            raise ValueError("No current agent name found in ActionContext.")
        return agent_name

    def get_delegation_depth(self) -> int:
        """
        Return the current delegation depth.

        Defaults to 0 for the root agent.
        """
        depth = self.get("delegation_depth", 0)

        if not isinstance(depth, int):
            raise ValueError(
                "ActionContext property 'delegation_depth' must be an integer."
            )

        if depth < 0:
            raise ValueError(
                "ActionContext property 'delegation_depth' must not be negative."
            )

        return depth

    def get_max_delegation_depth(self) -> int:
        """
        Return the configured maximum delegation depth.

        Defaults to 3 if not explicitly configured.
        """
        max_depth = self.get("max_delegation_depth", 3)

        if not isinstance(max_depth, int):
            raise ValueError(
                "ActionContext property 'max_delegation_depth' must be an integer."
            )

        if max_depth < 0:
            raise ValueError(
                "ActionContext property 'max_delegation_depth' must not be negative."
            )

        return max_depth

    def get_delegation_path(self) -> list[str]:
        """
        Return the ordered delegation path.

        Example:
            ["coordinator_agent", "file_ops_agent"]
        """
        path = self.get("delegation_path", [])

        if not isinstance(path, list):
            raise ValueError(
                "ActionContext property 'delegation_path' must be a list[str]."
            )

        normalized_path: list[str] = []
        for item in path:
            if not isinstance(item, str):
                raise ValueError(
                    "ActionContext property 'delegation_path' must contain only strings."
                )
            normalized_item = item.strip()
            if not normalized_item:
                raise ValueError(
                    "ActionContext property 'delegation_path' must not contain empty strings."
                )
            normalized_path.append(normalized_item)

        return normalized_path

    def get_visited_agents(self) -> set[str]:
        """
        Return the set of visited agent names.

        Used for fast loop detection during delegation.
        """
        visited = self.get("visited_agents", set())

        if not isinstance(visited, set):
            raise ValueError(
                "ActionContext property 'visited_agents' must be a set[str]."
            )

        normalized_visited: set[str] = set()
        for item in visited:
            if not isinstance(item, str):
                raise ValueError(
                    "ActionContext property 'visited_agents' must contain only strings."
                )
            normalized_item = item.strip()
            if not normalized_item:
                raise ValueError(
                    "ActionContext property 'visited_agents' must not contain empty strings."
                )
            normalized_visited.add(normalized_item)

        return normalized_visited

    def get_relevant_memory(self) -> dict[str, Any]:
        """
        Return the task-shaped relevant memory for the current agent.

        This is intentionally lightweight and is primarily meant for delegated
        child agents, so they can receive only the subset of memory/context
        that matters for the delegated task.

        Defaults to an empty dict when not configured.
        """
        relevant_memory = self.get("relevant_memory", {})

        if not isinstance(relevant_memory, dict):
            raise ValueError(
                "ActionContext property 'relevant_memory' must be a dict[str, Any]."
            )

        return dict(relevant_memory)

    def set_relevant_memory(self, relevant_memory: dict[str, Any]) -> None:
        """
        Set the relevant memory packet for the current context.

        Raises:
            ValueError: If the value is not a dictionary.
        """
        if not isinstance(relevant_memory, dict):
            raise ValueError("relevant_memory must be a dict[str, Any].")

        self.set("relevant_memory", dict(relevant_memory))

    def spawn_child(self, **overrides: Any) -> ActionContext:
        """
        Create a child ActionContext with inherited properties plus overrides.

        This is useful for delegation scenarios where one agent invokes another
        and we want a fresh context object without mutating the parent context.

        Example:
            child_context = action_context.spawn_child(
                current_agent_name="test_design_agent",
                delegation_depth=1,
            )
        """
        child_props = self.to_dict()
        child_props.update(overrides)
        return ActionContext(props=child_props)

    def spawn_delegated_child(
        self,
        next_agent_name: str,
        relevant_memory: dict[str, Any] | None = None,
    ) -> ActionContext:
        """
        Create a child ActionContext for a delegated agent call.

        This propagates delegation tracking state and can optionally attach a
        task-shaped relevant memory packet for the delegated child.

        Selective memory is explicit: if relevant_memory is provided, it is set
        on the child context as-is. If omitted, the child receives an empty
        relevant_memory dict by default rather than inheriting broad prior memory.
        """
        if not isinstance(next_agent_name, str):
            raise ValueError("next_agent_name must be a string.")

        normalized_name = next_agent_name.strip()
        if not normalized_name:
            raise ValueError("next_agent_name must not be empty.")

        if relevant_memory is None:
            child_relevant_memory: dict[str, Any] = {}
        else:
            if not isinstance(relevant_memory, dict):
                raise ValueError("relevant_memory must be a dict[str, Any].")
            child_relevant_memory = dict(relevant_memory)

        current_path = self.get_delegation_path()
        current_visited = self.get_visited_agents()

        return self.spawn_child(
            current_agent_name=normalized_name,
            delegation_depth=self.get_delegation_depth() + 1,
            max_delegation_depth=self.get_max_delegation_depth(),
            delegation_path=[*current_path, normalized_name],
            visited_agents={*current_visited, normalized_name},
            relevant_memory=child_relevant_memory,
        )
