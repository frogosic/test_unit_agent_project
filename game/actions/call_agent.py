from __future__ import annotations

from typing import Any

from game.actions.action_context import ActionContext
from game.core.memory import Memory
from game.bootstrap.decorators import action


def _extract_final_output(memory: Memory) -> str:
    """
    Extract the last meaningful assistant message from Memory.

    Your Memory structure:
    - stored in self._memories
    - accessed via get_memories()
    - assistant messages contain:
        {
            "role": "assistant",
            "content": ...
        }
    """
    memories = memory.get_memories()

    # Walk backwards to find last assistant message
    for item in reversed(memories):
        if item.get("role") == "assistant" and item.get("content"):
            return str(item["content"])

    # fallback: return last memory item if nothing else found
    if memories:
        return str(memories[-1])

    return "Agent completed without producing output."


@action(
    name="call_agent",
    description="Delegate a task to another registered specialist agent.",
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
                "description": "The task to delegate to the target agent.",
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
    """
    Delegate a task to another registered agent.

    Flow:
    - resolve caller from ActionContext
    - resolve registry from ActionContext
    - check delegation permission
    - enforce delegation safety rules
    - create isolated memory for the invoked agent
    - create child ActionContext for the target agent
    - run the target agent
    - return a structured summary of the result
    """
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
                f"Agent '{target_agent.name}' already exists in delegation path "
                f"{delegation_path}."
            ),
        }

    relevant_memory = {
        "delegated_task": task,
        "caller_agent": caller_name,
        "delegation_path": delegation_path,
        "visited_agents": sorted(visited_agents),
    }

    invoked_memory = Memory()
    child_context = action_context.spawn_delegated_child(
        next_agent_name=target_agent.name,
        relevant_memory=relevant_memory,
    )

    try:
        result_memory = target_agent.run_callable(
            user_input=task,
            memory=invoked_memory,
            action_context=child_context,
        )

        final_output = _extract_final_output(result_memory)

        return {
            "tool_executed": True,
            "success": True,
            "caller_agent": caller_name,
            "called_agent": target_agent.name,
            "task": task,
            "delegation_depth": child_context.get_delegation_depth(),
            "delegation_path": child_context.get_delegation_path(),
            "relevant_memory": child_context.get_relevant_memory(),
            "final_output": final_output,
        }

    except Exception as exc:
        return {
            "tool_executed": False,
            "success": False,
            "caller_agent": caller_name,
            "called_agent": target_agent.name,
            "task": task,
            "delegation_depth": child_context.get_delegation_depth(),
            "delegation_path": child_context.get_delegation_path(),
            "relevant_memory": child_context.get_relevant_memory(),
            "error": str(exc),
        }
