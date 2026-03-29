from __future__ import annotations

import ast
import logging
import json
import re
from textwrap import dedent

from game.actions.action_context import ActionContext
from game.agents.base_agent import BaseAgent
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.core.llm import LLM
from game.core.memory import Memory
from game.languages.tool_calling import AgentLanguage
from game.models.unit_test_models import GeneratedTestFile, TestWritingTask
from game.policies.file_security_policy import FileSecurityPolicy

logger = logging.getLogger(__name__)


class TestWritingAgent(BaseAgent):
    AGENT_NAME = "test_writing_agent"

    def __init__(
        self,
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        llm: LLM,
        environment: Environment,
        file_security_policy: FileSecurityPolicy,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="write_pytest_unit_tests",
                    description=(
                        "Generate high-quality pytest unit tests grounded in real source code "
                        "and structured test design. "
                        "Finish by calling `return_generated_test_file` with valid Python code."
                    ),
                )
            ],
            agent_language=agent_language,
            action_registry=action_registry,
            llm=llm,
            environment=environment,
            file_security_policy=file_security_policy,
            max_iterations=max_iterations,
        )

    def run_and_parse(
        self,
        task: TestWritingTask,
        memory: Memory | None = None,
        action_context: ActionContext | None = None,
    ) -> GeneratedTestFile:
        prompt = self._build_test_writing_prompt(task)

        run_memory = self.run(
            user_input=prompt,
            memory=memory,
            action_context=action_context,
        )
        return self._extract_generated_test_file(run_memory)

    def _extract_generated_test_file(self, memory: Memory) -> GeneratedTestFile:
        for item in reversed(memory.get_memories()):
            if item.get("role") != "tool":
                continue

            try:
                payload = json.loads(item["content"])
            except (KeyError, json.JSONDecodeError):
                continue

            inner_result = payload.get("result", {})
            if inner_result.get("result_type") != "generated_test_file":
                continue

            try:
                pytest_code = inner_result["pytest_code"]
                stripped = _strip_markdown_fences(pytest_code)
                _validate_python(stripped, inner_result["source_file_path"])

                return GeneratedTestFile(
                    source_file_path=inner_result["source_file_path"],
                    test_file_path=inner_result["test_file_path"],
                    pytest_code=stripped,
                )
            except KeyError as e:
                raise ValueError(
                    f"TestWritingAgent result missing required field: {e}"
                ) from e

        raise ValueError(
            "TestWritingAgent did not produce a valid generated_test_file result."
        )

    def _format_test_targets(self, task: TestWritingTask) -> str:
        sections: list[str] = []

        for target in task.test_targets:
            lines = [
                f"TARGET NAME: {target.name}",
                f"TARGET TYPE: {target.target_type}",
                f"DEPENDENCIES TO MOCK: {target.dependencies_to_mock or []}",
            ]

            for scenario in target.scenarios:
                lines.extend(
                    [
                        "",
                        f"  SCENARIO NAME: {scenario.name}",
                        f"  DESCRIPTION: {scenario.description}",
                        f"  INPUTS: {scenario.inputs}",
                        f"  ASSERTIONS: {scenario.assertions}",
                        f"  MOCK TARGETS: {scenario.mock_targets}",
                    ]
                )

            if target.notes:
                lines.extend(["", f"NOTES: {target.notes}"])

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _build_test_writing_prompt(self, task: TestWritingTask) -> str:
        formatted_test_targets = self._format_test_targets(task)

        return dedent(
            f"""
            You are a senior Python test engineer generating pytest unit tests from real source code.

            SOURCE FILE PATH
            {task.source_file_path}

            SOURCE CODE
            {task.source_code}

            MODULE SUMMARY
            {task.module_summary}

            STRUCTURED TEST TARGETS
            {formatted_test_targets}

            RULES
            - Write only unit tests
            - Use only behaviors supported by the provided source code and structured design
            - Every generated test must map to a real scenario from the structured test targets
            - Use the scenario INPUTS as hints for setup and invocation
            - Use the scenario ASSERTIONS as the basis for test expectations
            - Use the scenario MOCK TARGETS when patching collaborators or dependencies
            - Do not invent return values, fields, exception messages, or object structures
            - Do not invent dependencies not present in the source code or structured design
            - No placeholder assertions
            - No markdown fences
            - Prefer a few grounded tests over many speculative tests
            - Call `return_generated_test_file` with raw Python source code when complete
            """
        ).strip()


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _validate_python(code: str, source_path: str) -> None:
    """
    Validate that the generated code is syntactically valid Python.

    Raises:
        ValueError: If the code cannot be parsed.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f"TestWritingAgent generated syntactically invalid Python "
            f"for {source_path}: {e}"
        ) from e
