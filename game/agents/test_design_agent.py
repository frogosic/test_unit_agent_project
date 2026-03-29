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
from game.models.unit_test_models import (
    TestDesignResult,
    TestScenario,
    TestTarget,
)
from game.policies.file_security_policy import FileSecurityPolicy


class TestDesignAgent(BaseAgent):
    AGENT_NAME = "test_design_agent"

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
                    name="design_unit_tests",
                    description=(
                        "Analyze a Python source file and produce a structured unit test design.\n\n"
                        f"{UNIT_TEST_DEFINITION}\n\n"
                        "Rules:\n"
                        "- Identify unit-testable functions, methods, or classes\n"
                        "- Distinguish business logic from integration boundaries\n"
                        "- Identify dependencies that should be mocked or isolated\n"
                        "- Propose happy path, edge case, and failure scenarios\n"
                        "- Do not write pytest code yet\n"
                        "- When the task is complete, you must finish by calling "
                        "`return_test_design_result` with the final structured result"
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
    ) -> TestDesignResult:
        run_memory = self.run(
            user_input=user_input,
            memory=memory,
            max_iterations=max_iterations,
            action_context=action_context,
        )
        return self._extract_test_design_result(run_memory)

    def _extract_test_design_result(self, memory: Memory) -> TestDesignResult:
        for item in reversed(memory.get_memories()):
            if item.get("role") != "tool":
                continue

            try:
                payload = json.loads(item["content"])
            except (KeyError, json.JSONDecodeError):
                continue

            if payload.get("result_type") != "test_design":
                continue

            test_targets = []
            for target_data in payload.get("test_targets", []):
                scenarios = [
                    TestScenario(
                        name=scenario["name"],
                        description=scenario["description"],
                    )
                    for scenario in target_data.get("scenarios", [])
                ]

                test_targets.append(
                    TestTarget(
                        name=target_data["name"],
                        target_type=target_data["target_type"],
                        dependencies_to_mock=target_data.get(
                            "dependencies_to_mock", []
                        ),
                        scenarios=scenarios,
                        notes=target_data.get("notes", []),
                    )
                )

            return TestDesignResult(
                file_path=payload["file_path"],
                module_summary=payload["module_summary"],
                test_targets=test_targets,
            )

        raise ValueError("TestDesignAgent did not produce a valid test_design result.")
