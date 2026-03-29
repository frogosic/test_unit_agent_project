from __future__ import annotations

from collections.abc import Iterable

from game.core.core_action import Action, ActionRegistry


def build_action_registry(functions: Iterable) -> ActionRegistry:
    """
    Build an ActionRegistry from decorated functions.
    """
    registry = ActionRegistry()

    for func in functions:
        meta = getattr(func, "_action_meta", None)
        if not meta:
            raise ValueError(f"{func.__name__} is missing @action decorator")

        registry.register(
            Action(
                name=meta["name"],
                function=func,
                description=meta["description"],
                parameters=meta["parameters"],
                terminal=meta["terminal"],
            )
        )

    return registry
