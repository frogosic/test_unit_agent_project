from __future__ import annotations

from collections.abc import Iterable

from game.core.core_action import Action, ActionRegistry


def build_action_registry(items: Iterable) -> ActionRegistry:
    """
    Build an ActionRegistry from a mixture of:
    - decorated functions carrying _action_meta
    - prebuilt Action objects
    """
    registry = ActionRegistry()

    for item in items:
        if isinstance(item, Action):
            registry.register(item)
            continue

        meta = getattr(item, "_action_meta", None)
        if not meta:
            item_name = getattr(item, "__name__", repr(item))
            raise ValueError(f"{item_name} is missing @action decorator")

        registry.register(
            Action(
                name=meta["name"],
                function=item,
                description=meta["description"],
                parameters=meta["parameters"],
                terminal=meta["terminal"],
            )
        )

    return registry
