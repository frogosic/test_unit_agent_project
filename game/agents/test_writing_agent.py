from __future__ import annotations

import logging
import re
from textwrap import dedent

from game.actions.action_context import ActionContext
from game.agents.base_agent import BaseAgent
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.core.llm import LLM
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
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="write_pytest_unit_tests",
                    description="Generate high-quality pytest unit tests grounded in real source code and structured design.",
                )
            ],
            agent_language=agent_language,
            action_registry=action_registry,
            llm=llm,
            environment=environment,
            file_security_policy=file_security_policy,
        )

    def run_and_parse(
        self,
        task: TestWritingTask,
    ) -> GeneratedTestFile:
        prompt = self._build_test_writing_prompt(task)

        logger.debug("TestWritingAgent generating tests for: %s", task.source_file_path)

        raw = self.llm.generate_text([{"role": "user", "content": prompt}])
        code = _strip_markdown_fences(raw)

        if not code.strip():
            raise ValueError(
                f"TestWritingAgent produced empty output for: {task.source_file_path}"
            )

        test_file_path = _derive_test_path(task.source_file_path)

        return GeneratedTestFile(
            source_file_path=task.source_file_path,
            test_file_path=test_file_path,
            pytest_code=code,
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
            - Do not invent dependencies that are not present in the source code or structured design
            - No placeholder assertions
            - No markdown fences
            - Prefer a few grounded tests over many speculative tests
            - Return only raw Python source code, no explanations, no markdown
            """
        ).strip()


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _derive_test_path(source_file_path: str) -> str:
    """
    Derive a test file path from a source file path.

    Example:
        game/core/llm.py -> tests/game/core/test_llm.py
    """
    from pathlib import Path

    source = Path(source_file_path)
    return str(Path("tests") / source.parent / f"test_{source.name}")
