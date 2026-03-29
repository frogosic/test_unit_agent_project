from __future__ import annotations

import json
import logging
from typing import Any

from game.actions.action_context import ActionContext
from game.core.memory import Memory
from game.bootstrap.decorators import action

logger = logging.getLogger(__name__)


def _extract_structured_result(memory: Memory) -> dict[str, Any] | None:
    """
    Extract the structured result from a terminal action call in memory.

    Walks backwards through tool result messages looking for a payload
    with tool_executed=True and a result field. Returns the inner result
    dict if found, None otherwise.
    """
    for item in reversed(memory.get_memories()):
        if item.get("role") != "tool":
            continue

        try:
            payload = json.loads(item["content"])
        except (KeyError, json.JSONDecodeError):
            continue

        if not payload.get("tool_executed", False):
            continue

        result = payload.get("result")
        if result is not None:
            return result

    return None


def _extract_text_output(memory: Memory) -> str:
    """
    Fallback: extract the last meaningful assistant text message.
    Used when no structured result is found.
    """
    for item in reversed(memory.get_memories()):
        if item.get("role") == "assistant" and item.get("content"):
            return str(item["content"])

    if memory.get_memories():
        return str(memory.get_memories()[-1])

    return "Agent completed without producing output."


@action(
    name="call_agent",
    description=(
        "Delegate a task to a registered specialist agent and receive its structured result. "
        "Use file_ops_agent to discover and read Python source files. "
        "Use test_design_agent to analyze a source file and produce a structured test design. "
        "Use test_writing_agent to generate pytest code from a structured test design."
    ),
    parameters={
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "enum": [
                    "file_ops_agent",
                    "test_design_agent",
                    "test_writing_agent",
                ],
                "description": (
                    "The exact registered name of the specialist agent to call. "
                    "Use only one of: file_ops_agent, test_design_agent, test_writing_agent."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "A clear, self-contained description of the task to delegate. "
                    "Include all context the agent needs — file paths, source code, "
                    "or structured designs — since the agent has no prior memory."
                ),
            },
        },
        "required": ["agent_name", "task"],
        "additionalProperties": False,
    },
    terminal=False,
)
def call_agent(
    action_context: ActionContext,
    agent_name: str,
    task: str,
) -> dict[str, Any]:
    registry = action_context.require_agent_registry()
    caller_name = action_context.require_current_agent_name()

    target_agent = registry.require_agent(agent_name)
    registry.require_can_call(caller_name, target_agent.name)

    current_depth = action_context.get_delegation_depth()
    max_depth = action_context.get_max_delegation_depth()
    delegation_path = action_context.get_delegation_path()
    visited_agents = action_context.get_visited_agents()

    if current_depth >= max_depth:
        return {
            "tool_executed": False,
            "success": False,
            "caller_agent": caller_name,
            "called_agent": target_agent.name,
            "task": task,
            "error": (
                "Delegation denied: maximum delegation depth reached. "
                f"current_depth={current_depth}, "
                f"max_depth={max_depth}, "
                f"delegation_path={delegation_path}"
            ),
        }

    if target_agent.name in visited_agents:
        return {
            "tool_executed": False,
            "success": False,
            "caller_agent": caller_name,
            "called_agent": target_agent.name,
            "task": task,
            "error": (
                "Delegation denied: recursion or delegation loop detected. "
                f"Agent '{target_agent.name}' already in delegation path {delegation_path}."
            ),
        }

    relevant_memory = {
        "delegated_task": task,
        "caller_agent": caller_name,
        "delegation_path": delegation_path,
        "visited_agents": sorted(visited_agents),
    }

    child_context = action_context.spawn_delegated_child(
        next_agent_name=target_agent.name,
        relevant_memory=relevant_memory,
    )

    logger.info(
        "Delegating to %s (depth=%d): %s",
        target_agent.name,
        child_context.get_delegation_depth(),
        task[:120],
    )

    try:
        result_memory = target_agent.run_callable(
            user_input=task,
            memory=Memory(),
            action_context=child_context,
        )

        structured_result = _extract_structured_result(result_memory)

        if structured_result is not None:
            logger.info("Structured result received from %s", target_agent.name)
            return {
                "tool_executed": True,
                "success": True,
                "caller_agent": caller_name,
                "called_agent": target_agent.name,
                "task": task,
                "delegation_depth": child_context.get_delegation_depth(),
                "delegation_path": child_context.get_delegation_path(),
                "result": structured_result,
            }

        text_output = _extract_text_output(result_memory)
        logger.warning(
            "Agent %s returned text instead of structured result — treating as failure",
            target_agent.name,
            text_output[:500],
        )

        return {
            "tool_executed": True,
            "success": False,
            "caller_agent": caller_name,
            "called_agent": target_agent.name,
            "task": task,
            "delegation_depth": child_context.get_delegation_depth(),
            "delegation_path": child_context.get_delegation_path(),
            "error": (
                f"Agent '{target_agent.name}' returned text instead of calling "
                f"its required terminal action. Output: {text_output[:200]}"
            ),
        }

    except Exception as exc:
        logger.exception("Delegation to %s failed", target_agent.name)
        return {
            "tool_executed": False,
            "success": False,
            "caller_agent": caller_name,
            "called_agent": target_agent.name,
            "task": task,
            "delegation_depth": child_context.get_delegation_depth(),
            "delegation_path": child_context.get_delegation_path(),
            "error": str(exc),
        }
