from __future__ import annotations

import json

from game.actions.action_context import ActionContext
from game.agents.base_agent import BaseAgent
from game.constants.testing import UNIT_TEST_DEFINITION
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.core.llm import LLM
from game.core.memory import Memory
from game.languages.tool_calling import AgentLanguage
from game.models.unit_test_models import GeneratedTestFile
from game.policies.file_security_policy import FileSecurityPolicy


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
                    description=(
                        "Generate pytest unit test code from a structured unit test design.\n\n"
                        f"{UNIT_TEST_DEFINITION}\n\n"
                        "Rules:\n"
                        "- Write only unit tests\n"
                        "- Use mocks, stubs, or patching for external collaborators where appropriate\n"
                        "- Keep tests readable and behavior-focused\n"
                        "- Do not drift into integration or end-to-end testing\n"
                        "- Return generated pytest code and the intended test file path\n"
                        "- When the task is complete, you must finish by calling "
                        "`return_generated_test_file` with the final structured result"
                    ),
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
        user_input: str,
        memory: Memory | None = None,
        max_iterations: int = 20,
        action_context: ActionContext | None = None,
    ) -> GeneratedTestFile:
        run_memory = self.run(
            user_input=user_input,
            memory=memory,
            max_iterations=max_iterations,
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

            if payload.get("result_type") != "generated_test_file":
                continue

            return GeneratedTestFile(
                source_file_path=payload["source_file_path"],
                test_file_path=payload["test_file_path"],
                pytest_code=payload["pytest_code"],
            )

        raise ValueError(
            "TestWritingAgent did not produce a valid generated_test_file result."
        )
