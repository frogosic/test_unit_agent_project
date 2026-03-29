from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from game.actions.action_context import ActionContext
from game.core.agent_registry import AgentRegistry
from game.core.memory import Memory
from game.actions.call_agent import call_agent


@dataclass
class DummyAgentEntry:
    name: str
    run_callable: Callable
    description: str = ""


class DummyTargetAgent:
    def __init__(self, name: str, response_text: str = "done") -> None:
        self.name = name
        self.response_text = response_text

    def run(
        self, user_input: str, memory: Memory, action_context: ActionContext
    ) -> Memory:
        memory.add_memory(
            {
                "role": "assistant",
                "content": f"{self.name} handled task: {user_input}",
            }
        )
        return memory


def _build_registry() -> AgentRegistry:
    registry = AgentRegistry()

    coordinator = DummyTargetAgent(name="coordinator_agent")
    file_ops = DummyTargetAgent(name="file_ops_agent")
    reporting = DummyTargetAgent(name="reporting_agent")
    test_design = DummyTargetAgent(name="test_design_agent")

    registry.register_agent(
        name=coordinator.name,
        run_callable=coordinator.run,
        description="Coordinator agent",
    )
    registry.register_agent(
        name=file_ops.name,
        run_callable=file_ops.run,
        description="File ops agent",
    )
    registry.register_agent(
        name=reporting.name,
        run_callable=reporting.run,
        description="Reporting agent",
    )
    registry.register_agent(
        name=test_design.name,
        run_callable=test_design.run,
        description="Test design agent",
    )

    registry.allow_calls(
        caller_name="coordinator_agent",
        callee_names=[
            "file_ops_agent",
            "reporting_agent",
            "test_design_agent",
        ],
    )

    return registry


def _build_action_context(
    *,
    registry: AgentRegistry,
    current_agent_name: str = "coordinator_agent",
    delegation_depth: int = 0,
    max_delegation_depth: int = 3,
    delegation_path: list[str] | None = None,
    visited_agents: set[str] | None = None,
) -> ActionContext:
    if delegation_path is None:
        delegation_path = [current_agent_name]

    if visited_agents is None:
        visited_agents = {current_agent_name}

    return ActionContext(
        props={
            "agent_registry": registry,
            "current_agent_name": current_agent_name,
            "delegation_depth": delegation_depth,
            "max_delegation_depth": max_delegation_depth,
            "delegation_path": delegation_path,
            "visited_agents": visited_agents,
        }
    )


def test_call_agent_allows_valid_delegation() -> None:
    registry = _build_registry()
    action_context = _build_action_context(registry=registry)

    result = call_agent(
        action_context=action_context,
        agent_name="file_ops_agent",
        task="list files in the project",
    )

    assert result["tool_executed"] is True
    assert result["success"] is True
    assert result["caller_agent"] == "coordinator_agent"
    assert result["called_agent"] == "file_ops_agent"
    assert result["delegation_depth"] == 1
    assert result["delegation_path"] == ["coordinator_agent", "file_ops_agent"]
    assert (
        "file_ops_agent handled task: list files in the project"
        in result["final_output"]
    )


def test_call_agent_blocks_when_max_depth_reached() -> None:
    registry = _build_registry()
    action_context = _build_action_context(
        registry=registry,
        delegation_depth=3,
        max_delegation_depth=3,
        delegation_path=[
            "coordinator_agent",
            "file_ops_agent",
            "reporting_agent",
            "test_design_agent",
        ],
        visited_agents={
            "coordinator_agent",
            "file_ops_agent",
            "reporting_agent",
            "test_design_agent",
        },
    )

    result = call_agent(
        action_context=action_context,
        agent_name="file_ops_agent",
        task="list files in the project",
    )

    assert result["tool_executed"] is False
    assert result["success"] is False
    assert "maximum delegation depth reached" in result["error"]


def test_call_agent_blocks_recursion_loop_when_agent_already_visited() -> None:
    registry = _build_registry()
    action_context = _build_action_context(
        registry=registry,
        current_agent_name="coordinator_agent",
        delegation_depth=1,
        max_delegation_depth=3,
        delegation_path=["coordinator_agent", "file_ops_agent"],
        visited_agents={"coordinator_agent", "file_ops_agent"},
    )

    result = call_agent(
        action_context=action_context,
        agent_name="file_ops_agent",
        task="list files in the project",
    )

    assert result["tool_executed"] is False
    assert result["success"] is False
    assert "recursion or delegation loop detected" in result["error"]


def test_call_agent_blocks_unauthorized_delegation() -> None:
    registry = _build_registry()
    action_context = _build_action_context(
        registry=registry,
        current_agent_name="file_ops_agent",
        delegation_depth=1,
        max_delegation_depth=3,
        delegation_path=["coordinator_agent", "file_ops_agent"],
        visited_agents={"coordinator_agent", "file_ops_agent"},
    )

    with pytest.raises(ValueError, match="not allowed to call"):
        call_agent(
            action_context=action_context,
            agent_name="reporting_agent",
            task="write report",
        )
