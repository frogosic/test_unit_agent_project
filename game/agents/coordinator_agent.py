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
    FileProcessingStep,
    GeneratedTestFile,
    SourceFile,
    TestDesignResult,
    TestDesignTask,
    TestWritingTask,
    UnitTestGenerationPlan,
    UnitTestGenerationResult,
)
from game.policies.file_security_policy import FileSecurityPolicy
from game.services.generated_test_file_writer import write_generated_test_file

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    High-level orchestrator for unit test generation.

    Uses a two-phase approach:
    1. Plan — discover files and build a lightweight plan (file paths only)
    2. Execute — process each file deterministically via specialist agents
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
        max_iterations: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="coordinate_unit_test_generation",
                    description=(
                        "Coordinate end-to-end pytest unit test generation.\n\n"
                        "Your only job is to discover Python source files "
                        "in the target path and return their file paths.\n\n"
                        "Call file_ops_agent with the target path.\n"
                        "Once you have the file list, call return_unit_test_generation_result "
                        "with the list of file paths.\n\n"
                        "Do not design tests. Do not generate code. "
                        "Only discover files and return the plan."
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
        self.file_ops_agent = file_ops_agent
        self.test_design_agent = test_design_agent
        self.test_writing_agent = test_writing_agent

    def run_unit_test_generation(
        self,
        target_directory: str,
        action_context: ActionContext | None = None,
    ) -> UnitTestGenerationResult:
        logger.info("Starting unit test generation for: %s", target_directory)

        execution_context = self._build_execution_context(action_context)

        plan = self._build_plan(target_directory, execution_context)
        logger.info("Plan built — %d files to process", len(plan.steps))

        return self._execute_plan(plan, execution_context)

    def _build_plan(
        self,
        target_directory: str,
        action_context: ActionContext,
    ) -> UnitTestGenerationPlan:
        """
        Phase 1 — discover files and build a lightweight plan.
        Uses FileOpsAgent directly since we own it.
        """
        discovery_result = self.file_ops_agent.run_and_parse(
            target_directory=target_directory,
            action_context=action_context,
        )

        steps = [
            FileProcessingStep(file_path=source_file.path)
            for source_file in discovery_result.files
        ]

        return UnitTestGenerationPlan(
            target=target_directory,
            steps=steps,
            status="executing",
        )

    def _execute_plan(
        self,
        plan: UnitTestGenerationPlan,
        action_context: ActionContext,
    ) -> UnitTestGenerationResult:
        """
        Phase 2 — execute each step deterministically.
        Specialists fetch their own source code via read_file.
        """
        for step in plan.steps:
            logger.info("Processing: %s", step.file_path)
            step.status = "designing"

            try:
                test_design = self.test_design_agent.run_and_parse(
                    task=TestDesignTask(file_path=step.file_path),
                    action_context=action_context,
                )
                step.test_design = test_design
                step.status = "writing"
                logger.info("Design complete: %s", step.file_path)

            except Exception as e:
                logger.error("Design failed for %s: %s", step.file_path, e)
                step.status = "failed"
                step.error = str(e)
                continue

            try:
                generated_test = self.test_writing_agent.run_and_parse(
                    task=TestWritingTask(
                        source_file_path=test_design.file_path,
                        module_summary=test_design.module_summary,
                        test_targets=test_design.test_targets,
                    ),
                    action_context=action_context,
                )
                step.generated_test = generated_test
                step.status = "complete"
                logger.info("Tests generated: %s", step.file_path)

            except Exception as e:
                logger.error("Writing failed for %s: %s", step.file_path, e)
                step.status = "failed"
                step.error = str(e)
                continue

            try:
                write_generated_test_file(
                    generated_test=generated_test,
                    file_security_policy=self.file_security_policy,
                )
                logger.info("Written: %s", generated_test.test_file_path)
            except Exception as e:
                import traceback as tb

                logger.error(
                    "Writing failed for %s: %s\n%s",
                    step.file_path,
                    e,
                    tb.format_exc(),
                )
                step.status = "failed"
                step.error = str(e)
                continue

        plan.status = "complete"
        return self._build_result(plan)

    def _build_result(
        self,
        plan: UnitTestGenerationPlan,
    ) -> UnitTestGenerationResult:
        completed = plan.completed_steps()
        failed = plan.failed_steps()

        discovered_files = [
            SourceFile(path=step.file_path, content="", file_type="python")
            for step in plan.steps
        ]

        generated_tests = [
            step.generated_test for step in completed if step.generated_test is not None
        ]

        summary_parts = [
            f"Discovered {len(plan.steps)} source files.",
            f"Generated {len(generated_tests)} test files.",
        ]
        if failed:
            summary_parts.append(
                f"{len(failed)} file(s) failed: "
                + ", ".join(s.file_path for s in failed)
            )

        return UnitTestGenerationResult(
            target_directory=plan.target,
            discovered_files=discovered_files,
            test_designs=[
                step.test_design for step in completed if step.test_design is not None
            ],
            generated_tests=generated_tests,
            summary=" ".join(summary_parts),
        )
