from __future__ import annotations

import json
from textwrap import dedent

from game.actions.action_context import ActionContext
from game.agents.base_agent import BaseAgent
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.core.llm import LLM
from game.core.memory import Memory
from game.languages.tool_calling import AgentLanguage
from game.models.unit_test_models import (
    TestDesignResult,
    TestDesignTask,
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
                        "Analyze Python source code and produce a structured, implementation-grounded "
                        "unit test design. Finish by calling `return_test_design_result`."
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
        task: TestDesignTask,
        memory: Memory | None = None,
        max_iterations: int = 20,
        action_context: ActionContext | None = None,
    ) -> TestDesignResult:
        prompt = self._build_test_design_prompt(task)

        run_memory = self.run(
            user_input=prompt,
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

            inner_result = payload.get("result", {})
            if inner_result.get("result_type") != "test_design":
                continue

            test_targets = []
            for target_data in inner_result.get("test_targets", []):
                scenarios = [
                    TestScenario(
                        name=scenario["name"],
                        description=scenario["description"],
                        inputs=scenario.get("inputs", []),
                        assertions=scenario.get("assertions", []),
                        mock_targets=scenario.get("mock_targets", []),
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

            try:
                return TestDesignResult(
                    file_path=inner_result["file_path"],
                    module_summary=inner_result["module_summary"],
                    test_targets=test_targets,
                )
            except KeyError as e:
                raise ValueError(
                    f"TestDesignAgent result is missing required field: {e}"
                ) from e

        raise ValueError("TestDesignAgent did not produce a valid test_design result.")

    def _build_test_design_prompt(self, task: TestDesignTask) -> str:
        return dedent(
            f"""
            You are a Python unit test design specialist.

            FILE PATH
            {task.file_path}

            SOURCE CODE
            {task.source_code}

            RULES
            - Identify only real unit-testable functions, methods, or classes visible in the source code
            - Distinguish business logic from integration boundaries
            - Identify real collaborators or dependencies that may need mocking
            - Propose only scenarios grounded in the implementation
            - Do not invent hidden behavior
            - Do not write pytest code
            - Prefer fewer high-confidence scenarios over many speculative ones
            - For each scenario, provide:
            - a clear scenario name
            - a short description
            - concrete inputs or setup hints as a list of strings
            - observable assertions as a list of strings
            - mock targets as a list of strings
            - Assertions must reflect only behavior visible in the code
            - Mock targets must refer only to real collaborators/dependencies visible in the code
            - Finish by calling `return_test_design_result` with the final structured result
            """
        ).strip()
