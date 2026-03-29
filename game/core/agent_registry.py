from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from game.core.memory import Memory


AgentRunCallable = Callable[..., Memory]


@dataclass(slots=True)
class RegisteredAgent:
    """
    Runtime representation of an agent that can be invoked by name.

    Attributes:
        name: Unique agent identifier used by the registry and call_agent tool.
        run_callable: The callable used to execute the agent.
        description: Optional human-readable description of the agent's purpose.
    """

    name: str
    run_callable: AgentRunCallable
    description: str = ""


@dataclass(slots=True)
class AgentRegistry:
    """
    Central runtime registry for all callable agents in the system.

    Responsibilities:
    - Register agents by name
    - Retrieve agents by name
    - Keep delegation permissions
    - Validate whether one agent may call another

    This object should be created at composition time (for example in a
    bootstrap module or main.py) and then injected into ActionContext.
    """

    _agents: dict[str, RegisteredAgent] = field(default_factory=dict)
    _delegation_permissions: dict[str, set[str]] = field(default_factory=dict)

    def register_agent(
        self,
        name: str,
        run_callable: AgentRunCallable,
        description: str = "",
    ) -> None:
        """
        Register an agent in the registry.

        Raises:
            ValueError: If the name is empty or already registered.
        """
        normalized_name = self._normalize_name(name)

        if normalized_name in self._agents:
            raise ValueError(f"Agent '{normalized_name}' is already registered.")

        self._agents[normalized_name] = RegisteredAgent(
            name=normalized_name,
            run_callable=run_callable,
            description=description,
        )

    def get_agent(self, name: str) -> RegisteredAgent | None:
        """
        Retrieve a registered agent by name.
        """
        normalized_name = self._normalize_name(name)
        return self._agents.get(normalized_name)

    def require_agent(self, name: str) -> RegisteredAgent:
        """
        Retrieve a registered agent by name or raise an error if missing.

        Raises:
            ValueError: If the agent is not registered.
        """
        agent = self.get_agent(name)
        if agent is None:
            normalized_name = self._normalize_name(name)
            raise ValueError(f"Agent '{normalized_name}' is not registered.")
        return agent

    def has_agent(self, name: str) -> bool:
        """
        Return True if an agent with the provided name exists.
        """
        normalized_name = self._normalize_name(name)
        return normalized_name in self._agents

    def list_agents(self) -> list[str]:
        """
        Return all registered agent names in sorted order.
        """
        return sorted(self._agents.keys())

    def allow_calls(self, caller_name: str, callee_names: list[str]) -> None:
        """
        Define which agents a caller is allowed to invoke.

        This replaces any previous permissions for the caller.

        Raises:
            ValueError: If caller or any callee is not registered.
        """
        normalized_caller = self._normalize_name(caller_name)
        self.require_agent(normalized_caller)

        normalized_callees = {
            self.require_agent(callee_name).name for callee_name in callee_names
        }

        self._delegation_permissions[normalized_caller] = normalized_callees

    def can_call(self, caller_name: str, callee_name: str) -> bool:
        """
        Return True if the caller is permitted to invoke the callee.

        Rules:
        - both caller and callee must be registered
        - caller must have an explicit permission entry
        - callee must be in the caller's allowed set
        """
        normalized_caller = self._normalize_name(caller_name)
        normalized_callee = self._normalize_name(callee_name)

        if normalized_caller not in self._agents:
            return False

        if normalized_callee not in self._agents:
            return False

        allowed_callees = self._delegation_permissions.get(normalized_caller, set())
        return normalized_callee in allowed_callees

    def require_can_call(self, caller_name: str, callee_name: str) -> None:
        """
        Validate delegation permission or raise an error.

        Raises:
            ValueError: If either agent is missing or the call is not allowed.
        """
        caller = self.require_agent(caller_name)
        callee = self.require_agent(callee_name)

        if not self.can_call(caller.name, callee.name):
            raise ValueError(
                f"Agent '{caller.name}' is not allowed to call agent '{callee.name}'."
            )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize agent names to reduce accidental mismatches.

        Raises:
            ValueError: If the provided name is empty after normalization.
        """
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Agent name must not be empty.")
        return normalized_name
