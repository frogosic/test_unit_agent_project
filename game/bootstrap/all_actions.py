from __future__ import annotations

from game.actions.call_agent import call_agent
from game.actions.expert_actions import prompt_expert
from game.actions.file_actions import (
    list_directories,
    list_files,
    read_file,
    terminate,
)
from game.actions.result_actions import (
    RETURN_TEST_DESIGN_RESULT_ACTION,
)
from game.core.core_action import ActionRegistry
from game.bootstrap.registry import build_action_registry


def build_coordinator_action_registry() -> ActionRegistry:
    """
    Coordinator/orchestrator actions only.
    """
    return build_action_registry(
        [
            call_agent,
            terminate,
        ]
    )


def build_file_ops_action_registry() -> ActionRegistry:
    """
    File operations specialist actions.
    """
    return build_action_registry(
        [
            list_directories,
            list_files,
            read_file,
        ]
    )


def build_test_design_action_registry() -> ActionRegistry:
    """
    Test design specialist actions.
    """
    return build_action_registry(
        [
            prompt_expert,
            RETURN_TEST_DESIGN_RESULT_ACTION,
        ]
    )


def build_test_writing_action_registry() -> ActionRegistry:
    """
    Test writing specialist actions.
    """
    return build_action_registry([])
