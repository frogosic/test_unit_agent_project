from __future__ import annotations

from game.actions.action_context import ActionContext
from game.bootstrap.decorators import action


def _get_file_security_policy(action_context: ActionContext):
    """Retrieve the file security policy from the action context."""
    policy = action_context.get("file_security_policy")

    if policy is None:
        raise ValueError("Missing file_security_policy in action context.")

    return policy


def _write_file(
    action_context: ActionContext,
    file_name: str,
    content: str,
    allowed_suffixes: set[str],
) -> str:
    """Write a file safely if its suffix is allowed."""
    policy = _get_file_security_policy(action_context)
    path = policy.ensure_safe_write_path(file_name)

    if path.suffix not in allowed_suffixes:
        raise ValueError(
            f"Invalid file type: {path.suffix}. Allowed: {', '.join(sorted(allowed_suffixes))}"
        )

    path.write_text(content, encoding="utf-8")
    return f"Wrote to {file_name}"


@action(tags=["testing", "test_writing"])
def write_pytest_file(
    action_context: ActionContext,
    file_name: str,
    content: str,
) -> str:
    """
    Write content to a pytest file, ensuring the path is safe.

    Args:
        action_context: Runtime execution context containing the file security policy.
        file_name: The name or path of the pytest file to write. Must end with .py.
        content: The content to write to the file.

    Returns:
        A confirmation message indicating the file was written.
    """
    return _write_file(
        action_context,
        file_name,
        content,
        allowed_suffixes={".py"},
    )
