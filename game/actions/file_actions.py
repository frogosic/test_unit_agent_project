from __future__ import annotations

import os

from game.actions.action_context import ActionContext
from game.core.core_action import ActionRegistry
from game.bootstrap.decorators import action
from game.bootstrap.registry import build_action_registry


def _get_file_security_policy(action_context: ActionContext):
    """Retrieve the file security policy from the action context."""
    policy = action_context.get("file_security_policy")

    if policy is None:
        raise ValueError("Missing file_security_policy in action context.")

    return policy


@action(tags=["file_operations"])
def list_directories(
    action_context: ActionContext,
    directory: str = ".",
) -> list[str]:
    """
    List visible, safe directories in the specified directory. Defaults to the current directory.

    Args:
        action_context: Runtime execution context containing the file security policy.
        directory: The directory to inspect. Defaults to the current directory.

    Returns:
        A sorted list of visible directory names, excluding hidden and restricted directories.
    """
    policy = _get_file_security_policy(action_context)
    path = policy.ensure_safe_directory_path(directory)

    return sorted(
        name
        for name in os.listdir(path)
        if (path / name).is_dir()
        and not name.startswith(".")
        and not policy.should_hide_name(name)
    )


@action(tags=["file_operations"])
def list_files(
    action_context: ActionContext,
    directory: str = ".",
) -> list[dict[str, str]]:
    """
    List visible, safe files and directories in the specified directory. Defaults to the current directory.

    Args:
        action_context: Runtime execution context containing the file security policy.
        directory: The directory to inspect. Defaults to the current directory.

    Returns:
        A list of dictionaries with item name and type, excluding hidden and restricted items.
    """
    policy = _get_file_security_policy(action_context)
    path = policy.ensure_safe_directory_path(directory)
    items: list[dict[str, str]] = []

    for name in sorted(os.listdir(path)):
        if name.startswith(".") or policy.should_hide_name(name):
            continue

        item_path = path / name
        item_type = "file" if item_path.is_file() else "directory"
        items.append({"name": name, "type": item_type})

    return items


@action(tags=["file_operations"])
def read_file(action_context: ActionContext, file_name: str) -> str:
    """
    Read the contents of a specific file.

    Args:
        action_context: Runtime execution context containing the file security policy.
        file_name: The name or path of the file to read.

    Returns:
        The contents of the file as a string.
    """
    policy = _get_file_security_policy(action_context)
    path = policy.ensure_safe_read_path(file_name)
    return path.read_text(encoding="utf-8")


@action(terminal=True, tags=["control"])
def terminate(message: str) -> str:
    """
    End the agent loop with a short final answer.

    Args:
        message: A short final response for the user.

    Returns:
        The same final message.
    """
    return message
