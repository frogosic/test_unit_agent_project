from __future__ import annotations

from game.actions.action_context import ActionContext
from game.bootstrap.decorators import action


@action()
def prompt_expert(
    action_context: ActionContext,
    description_of_expert: str,
    prompt: str,
) -> str:
    """
    Generate a response from an expert persona.

    Args:
        description_of_expert: Detailed description of the expert's background and expertise.
        prompt: The specific question or task for the expert.

    Returns:
        The expert's response.
    """
    llm = action_context.get("llm")
    if llm is None:
        raise ValueError("No LLM available in action_context.")

    if not hasattr(llm, "generate_text"):
        raise TypeError("Configured LLM does not expose a 'generate_text' method.")

    messages = [
        {
            "role": "system",
            "content": (
                "Act as the following expert and respond accordingly:\n\n"
                f"{description_of_expert}"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return llm.generate_text(messages)
