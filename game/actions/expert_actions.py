from __future__ import annotations

from game.actions.action_context import ActionContext
from game.bootstrap.decorators import action


@action(tags=["experts"])
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


@action(tags=["testing", "experts"])
def design_test_suite(
    action_context: ActionContext,
    feature_description: str,
) -> str:
    """
    Design a comprehensive test suite by consulting a senior QA engineer.

    Args:
        feature_description: Description of the feature or code to test.

    Returns:
        A structured test plan covering major testing layers and edge cases.
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
You are a senior QA engineer with 12 years of experience in test design and automation.

Your expertise includes:
- Unit, integration, and end-to-end testing
- API and UI automation strategy
- Edge case discovery
- Failure-mode analysis
- Performance and security testing
- Writing practical, implementation-aware test plans

You are thorough, pragmatic, and focused on real risks rather than boilerplate.
You explain clearly and structure your output so it can be used directly by engineers.
        """.strip(),
        prompt=f"""
Please design a practical and concise test suite for the following feature or code:

{feature_description}

Requirements:
- Keep the response compact and structured.
- Use markdown headings and bullet points only.
- Do not include code blocks or long examples.
- Do not invent functions, classes, or APIs not present in the provided code.
- Base your analysis only on the provided code.
- Use only provided code for test design; do not assume additional context.
- If code is orchestration only, focus tests on orchestration logic.
- Keep the output under 700 words.

Cover:
1. Unit tests
2. Integration tests
3. End-to-end tests if relevant
4. Edge cases and failure scenarios
5. Test data needs
6. Key risks
""".strip(),
    )
