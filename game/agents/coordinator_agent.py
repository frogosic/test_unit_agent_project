from __future__ import annotations

from game.actions.action_context import ActionContext
from game.agents.base_agent import BaseAgent
from game.agents.file_ops_agent import FileOpsAgent
from game.agents.test_design_agent import TestDesignAgent
from game.agents.test_writing_agent import TestWritingAgent
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.core.llm import LLM
from game.languages.tool_calling import AgentLanguage
from game.models.unit_test_models import (
    GeneratedTestFile,
    TestDesignResult,
    UnitTestGenerationResult,
)
from game.policies.file_security_policy import FileSecurityPolicy


class CoordinatorAgent(BaseAgent):
    """
    High-level orchestrator agent for the unit test generation workflow.

    Responsibilities:
    - Coordinate source file discovery
    - Coordinate unit test design
    - Coordinate pytest test generation
    - Combine outputs into a final structured result
    """

    AGENT_NAME = "coordinator_agent"

    def __init__(
        self,
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        llm: LLM,
        environment: Environment,
        file_security_policy: FileSecurityPolicy,
        file_ops_agent: FileOpsAgent,
        test_design_agent: TestDesignAgent,
        test_writing_agent: TestWritingAgent,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="coordinate_unit_test_generation",
                    description=(
                        "Coordinate the end-to-end workflow for generating pytest unit tests "
                        "from Python source files.\n\n"
                        "Workflow responsibilities:\n"
                        "- Discover relevant Python source files in the requested directory\n"
                        "- Analyze each file for unit-testable targets\n"
                        "- Design structured unit test scenarios\n"
                        "- Generate pytest unit test code\n"
                        "- Combine all structured outputs into a final result\n\n"
                        "Rules:\n"
                        "- Focus on unit tests only\n"
                        "- Prefer structured outputs from specialist agents\n"
                        "- Keep orchestration deterministic and pipeline-driven\n"
                        "- Do not treat filenames alone as sufficient evidence of behavior\n"
                        "- Base analysis on inspected file content"
                    ),
                )
            ],
            agent_language=agent_language,
            action_registry=action_registry,
            llm=llm,
            environment=environment,
            file_security_policy=file_security_policy,
        )
        self.file_ops_agent = file_ops_agent
        self.test_design_agent = test_design_agent
        self.test_writing_agent = test_writing_agent

    def run_unit_test_generation(
        self,
        target_directory: str,
        action_context: ActionContext | None = None,
    ) -> UnitTestGenerationResult:
        """
        Run the full unit test generation workflow for a target directory.
        """
        execution_context = self._build_execution_context(action_context)

        discovery_result = self.file_ops_agent.run_and_parse(
            user_input=(
                f"Inspect the directory '{target_directory}', identify relevant Python "
                "source files for unit test generation, read their contents, and return "
                "the final structured file discovery result."
            ),
            action_context=execution_context,
        )

        test_designs: list[TestDesignResult] = []
        generated_tests: list[GeneratedTestFile] = []

        for source_file in discovery_result.files:
            test_design = self.test_design_agent.run_and_parse(
                user_input=(
                    f"Analyze the following Python source file for unit-testable behavior "
                    f"and return a structured unit test design.\n\n"
                    f"File path: {source_file.path}\n\n"
                    f"Source code:\n{source_file.content}"
                ),
                action_context=execution_context,
            )
            test_designs.append(test_design)

            generated_test = self.test_writing_agent.run_and_parse(
                user_input=(
                    "Generate pytest unit tests from the following structured test design "
                    "and return the final structured generated test file result.\n\n"
                    f"Source file path: {test_design.file_path}\n"
                    f"Module summary: {test_design.module_summary}\n"
                    f"Test targets: {test_design.test_targets}"
                ),
                action_context=execution_context,
            )
            generated_tests.append(generated_test)

        return UnitTestGenerationResult(
            target_directory=target_directory,
            discovered_files=discovery_result.files,
            test_designs=test_designs,
            generated_tests=generated_tests,
            summary=(
                f"Discovered {len(discovery_result.files)} source files, "
                f"designed {len(test_designs)} unit test suites, "
                f"generated {len(generated_tests)} pytest test files."
            ),
        )
