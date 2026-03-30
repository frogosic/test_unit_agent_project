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
from game.services.pytest_runner import validate_generated_test
from game.services.code_fixups import _fix_mock_name_pattern

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
        max_iterations: int = 15,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="write_pytest_unit_tests",
                    description=(
                        "Generate pytest unit tests from the provided source code and test design. "
                        "You MUST finish by calling `return_generated_test_file` with the generated code. "
                        "Do NOT respond with text. Do NOT explain your work. "
                        "Call `return_generated_test_file` as your only and final action."
                    ),
                )
            ],
            agent_language=agent_language,
            action_registry=action_registry,
            llm=llm,
            environment=environment,
            file_security_policy=file_security_policy,
            max_iterations=max_iterations,
            max_retries=max_retries,
        )

    def run_and_parse(
        self,
        task: TestWritingTask,
        action_context: ActionContext | None = None,
    ) -> GeneratedTestFile:
        prompt = self._build_test_writing_prompt(task)
        return self._run_with_retry(
            user_input=prompt,
            extract=self._extract_generated_test_file,
            validate=validate_generated_test,
            transform=self._transform_generated_test_file,
            action_context=action_context,
        )

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

    def _transform_generated_test_file(
        self, generated: GeneratedTestFile
    ) -> GeneratedTestFile:
        return GeneratedTestFile(
            source_file_path=generated.source_file_path,
            test_file_path=generated.test_file_path,
            pytest_code=_fix_mock_name_pattern(generated.pytest_code),
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
            You are a senior Python test engineer generating pytest unit tests.

            SOURCE FILE PATH
            {task.source_file_path}

            INSTRUCTIONS
            - Use the read_file action to read the source file at the path above
            - Use the structured test design below to write grounded pytest tests
            - Finish by calling `return_generated_test_file` with raw Python code

            MODULE SUMMARY
            {task.module_summary}

            STRUCTURED TEST TARGETS
            {formatted_test_targets}

            COMMON MISTAKES TO AVOID
            - When using @patch as a decorator, always add a corresponding positional
            argument to the test function for each patch applied, in reverse order.
            If the test does not need to interact with the mock, still declare the argument:

            # WRONG - missing argument, will raise TypeError at runtime:
            @patch('game.core.llm.completion')
            def test_something():
                ...

            # CORRECT - one argument per @patch, in reverse decorator order:
            @patch('game.core.llm.completion')
            def test_something(mock_completion):
                ...

            - Only apply @patch when the scenario's MOCK TARGETS list is non-empty.
            If MOCK TARGETS is empty, do not add any @patch decorators:

            # WRONG - patching when mock_targets is []:
            @patch('game.core.llm.completion')
            def test_llm_init():
                llm = LLM()
                assert llm.model == 'openai/gpt-4o-mini'

            # CORRECT - no patch when nothing needs to be mocked:
            def test_llm_init():
                llm = LLM()
                assert llm.model == 'openai/gpt-4o-mini'

            - When mocking objects with a 'name' attribute, never pass name= to MagicMock.
            The name= constructor argument sets the mock's internal identifier and corrupts
            the .name attribute — even if you assign .name afterward, it will not work:

            # WRONG - name= constructor argument corrupts .name even with reassignment:
            mock = MagicMock(name='action1')
            mock.name = 'action1'  # does NOT fix it, .name still returns a child mock

            # CORRECT - omit name= entirely, only set the attribute:
            mock = MagicMock()
            mock.name = 'action1'  # works correctly

            - When asserting mock call arguments, always use keyword arguments in
            assert_called_once_with if the actual function call uses keyword arguments:

            # WRONG - positional when the real call uses keyword args:
            mock_fn.assert_called_once_with(prompt)

            # CORRECT - mirrors the actual call signature:
            mock_fn.assert_called_once_with(prompt=prompt, max_tokens=None)

            RULES
            - Write only unit tests
            - Use only behaviors supported by the source code and structured design
            - Every test must map to a real scenario from the structured test targets
            - Use scenario INPUTS as hints for setup and invocation
            - Use scenario ASSERTIONS as the basis for test expectations
            - Use scenario MOCK TARGETS when patching collaborators
            - Do not invent return values, fields, or exception messages
            - No placeholder assertions
            - No markdown fences
            - When testing filesystem-dependent code, use pytest's tmp_path fixture
            and initialize path-dependent objects with tmp_path as the base directory
            - Call `return_generated_test_file` with raw Python source code when complete
            - When patching imported functions, patch at the module where they are imported,
            not where they are defined. For example, if module 'game.core.llm' imports
            'completion' from 'litellm', the correct patch target is 'game.core.llm.completion',
            not 'litellm.completion'
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
