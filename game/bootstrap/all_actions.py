from __future__ import annotations

from game.actions.expert_actions import prompt_expert
from game.actions.file_actions import (
    list_directories,
    list_files,
    read_file,
    terminate,
)
from game.actions.result_actions import (
    RETURN_GENERATED_TEST_FILE_ACTION,
    RETURN_TEST_DESIGN_RESULT_ACTION,
)
from game.core.core_action import ActionRegistry
from game.bootstrap.registry import build_action_registry


def build_coordinator_action_registry() -> ActionRegistry:
    """
    Coordinator executes deterministically via Python.
    terminate is kept as a safety valve if the LLM loop is ever entered.
    """
    return build_action_registry(
        [
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
            read_file,
            prompt_expert,
            RETURN_TEST_DESIGN_RESULT_ACTION,
        ]
    )


def build_test_writing_action_registry() -> ActionRegistry:
    """
    Test writing specialist actions.
    """
    return build_action_registry(
        [
            read_file,
            RETURN_GENERATED_TEST_FILE_ACTION,
        ]
    )
