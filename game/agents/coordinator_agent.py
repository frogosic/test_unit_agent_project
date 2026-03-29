from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

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
    TestDesignTask,
    TestWritingTask,
    UnitTestGenerationResult,
)
from game.policies.file_security_policy import FileSecurityPolicy
from game.services.generated_test_file_writer import write_generated_test_file

logger = logging.getLogger(__name__)


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
        logger.info("Starting unit test generation for: %s", target_directory)

        execution_context = self._build_execution_context(action_context)

        discovery_result = self.file_ops_agent.run_and_parse(
            target_directory=target_directory,
            action_context=execution_context,
        )

        logger.info("Discovered %d source files", len(discovery_result.files))

        test_designs: list[TestDesignResult] = []
        generated_tests: list[GeneratedTestFile] = []

        for source_file in discovery_result.files:
            logger.info("Processing: %s", source_file.path)

            test_design = self.test_design_agent.run_and_parse(
                task=TestDesignTask(
                    file_path=source_file.path,
                    source_code=source_file.content,
                ),
                action_context=execution_context,
            )
            test_designs.append(test_design)
            logger.info("Test design complete: %s", source_file.path)

            generated_test = self.test_writing_agent.run_and_parse(
                task=TestWritingTask(
                    source_file_path=test_design.file_path,
                    source_code=source_file.content,
                    module_summary=test_design.module_summary,
                    test_targets=test_design.test_targets,
                ),
            )
            generated_tests.append(generated_test)
            logger.info("Tests generated: %s", source_file.path)

            write_generated_test_file(
                generated_test=generated_test,
                file_security_policy=self.file_security_policy,
            )

            self._write_debug_snapshot(
                source_file_path=source_file.path,
                test_design=test_design,
                generated_test=generated_test,
            )

        logger.info(
            "Generation complete — files: %d, designs: %d, tests: %d",
            len(discovery_result.files),
            len(test_designs),
            len(generated_tests),
        )

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

    def _write_debug_snapshot(
        self,
        source_file_path: str,
        test_design: TestDesignResult,
        generated_test: GeneratedTestFile,
    ) -> None:
        debug_dir = Path("debug/unit_test_generation")
        debug_dir.mkdir(parents=True, exist_ok=True)

        source_stem = Path(source_file_path).stem

        design_path = debug_dir / f"{source_stem}_design.json"
        generated_test_path = debug_dir / f"{source_stem}_generated_test.py"

        design_path.write_text(
            json.dumps(asdict(test_design), indent=2),
            encoding="utf-8",
        )
        generated_test_path.write_text(
            generated_test.pytest_code,
            encoding="utf-8",
        )
