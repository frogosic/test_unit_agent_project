from __future__ import annotations

from dataclasses import dataclass

from game.actions.action_context import ActionContext
from game.core.agent_registry import AgentRegistry
from game.agents import (
    CoordinatorAgent,
    FileOpsAgent,
    TestDesignAgent,
    TestWritingAgent,
)
from game.core.core_action import ActionRegistry
from game.core.environment import Environment
from game.languages.tool_calling import AgentLanguage
from game.core.llm import LLM
from game.policies.file_security_policy import FileSecurityPolicy


@dataclass(slots=True)
class AgentSystem:
    """
    Fully assembled multi-agent runtime.
    """

    coordinator: CoordinatorAgent
    root_action_context: ActionContext


def _register_agents(
    *,
    agent_registry: AgentRegistry,
    coordinator: CoordinatorAgent,
    file_ops_agent: FileOpsAgent,
    test_design_agent: TestDesignAgent,
    test_writing_agent: TestWritingAgent,
) -> None:
    agent_registry.register_agent(
        name=coordinator.name,
        run_callable=coordinator.run,
        description="Top-level orchestrator agent.",
    )
    agent_registry.register_agent(
        name=file_ops_agent.name,
        run_callable=file_ops_agent.run,
        description="Specialist agent for file operations.",
    )
    agent_registry.register_agent(
        name=test_design_agent.name,
        run_callable=test_design_agent.run,
        description="Specialist agent for unit test design.",
    )
    agent_registry.register_agent(
        name=test_writing_agent.name,
        run_callable=test_writing_agent.run,
        description="Specialist agent for pytest unit test generation.",
    )


def _configure_delegation_permissions(
    *,
    agent_registry: AgentRegistry,
    coordinator: CoordinatorAgent,
    specialists: list[FileOpsAgent | TestDesignAgent | TestWritingAgent],
) -> None:
    agent_registry.allow_calls(
        caller_name=coordinator.name,
        callee_names=[agent.name for agent in specialists],
    )


def build_agent_system(
    *,
    agent_language: AgentLanguage,
    llm: LLM,
    environment: Environment,
    file_security_policy: FileSecurityPolicy,
    coordinator_action_registry: ActionRegistry,
    file_ops_action_registry: ActionRegistry,
    test_design_action_registry: ActionRegistry,
    test_writing_action_registry: ActionRegistry,
) -> AgentSystem:
    """
    Assemble the full multi-agent system.

    This version expects action registries to be built outside and injected here.
    That keeps this bootstrap module compatible with the project's real action
    registration mechanism.
    """
    file_ops_agent = FileOpsAgent(
        agent_language=agent_language,
        action_registry=file_ops_action_registry,
        llm=llm,
        environment=environment,
        file_security_policy=file_security_policy,
    )

    test_design_agent = TestDesignAgent(
        agent_language=agent_language,
        action_registry=test_design_action_registry,
        llm=llm,
        environment=environment,
        file_security_policy=file_security_policy,
    )

    test_writing_agent = TestWritingAgent(
        agent_language=agent_language,
        action_registry=test_writing_action_registry,
        llm=llm,
        environment=environment,
        file_security_policy=file_security_policy,
    )

    coordinator = CoordinatorAgent(
        agent_language=agent_language,
        action_registry=coordinator_action_registry,
        llm=llm,
        environment=environment,
        file_security_policy=file_security_policy,
        file_ops_agent=file_ops_agent,
        test_design_agent=test_design_agent,
        test_writing_agent=test_writing_agent,
    )

    agent_registry = AgentRegistry()

    _register_agents(
        agent_registry=agent_registry,
        coordinator=coordinator,
        file_ops_agent=file_ops_agent,
        test_design_agent=test_design_agent,
        test_writing_agent=test_writing_agent,
    )

    _configure_delegation_permissions(
        agent_registry=agent_registry,
        coordinator=coordinator,
        specialists=[file_ops_agent, test_design_agent, test_writing_agent],
    )

    root_action_context = ActionContext(
        props={
            "agent_registry": agent_registry,
            "current_agent_name": coordinator.name,
            "llm": llm,
            "file_security_policy": file_security_policy,
            "delegation_depth": 0,
            "max_delegation_depth": 3,
            "delegation_path": [coordinator.name],
            "visited_agents": {coordinator.name},
        }
    )

    return AgentSystem(
        coordinator=coordinator,
        root_action_context=root_action_context,
    )
