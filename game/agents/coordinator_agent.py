from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from game.actions.action_context import ActionContext
from game.models.unit_test_models import SourceFile
from game.agents.base_agent import BaseAgent
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

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    High-level orchestrator agent for the unit test generation workflow.

    Delegates all work to specialist agents via call_agent.
    Assembles the final structured result from delegation outputs.
    """

    AGENT_NAME = "coordinator_agent"

    def __init__(
        self,
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        llm: LLM,
        environment: Environment,
        file_security_policy: FileSecurityPolicy,
        max_iterations: int = 50,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="coordinate_unit_test_generation",
                    description=(
                        "Coordinate end-to-end pytest unit test generation for a target directory.\n\n"
                        "WORKFLOW — follow this exact sequence:\n\n"
                        "1. DISCOVER FILES\n"
                        "   Call file_ops_agent with the target directory.\n"
                        "   The result contains a list of source files with their paths and content.\n\n"
                        "2. FOR EACH SOURCE FILE — repeat steps 2a and 2b:\n\n"
                        "   2a. DESIGN TESTS\n"
                        "       Call test_design_agent.\n"
                        "       Include the full file_path and source_code in the task.\n"
                        "       The result contains a structured test design.\n\n"
                        "   2b. GENERATE TESTS\n"
                        "       Call test_writing_agent.\n"
                        "       Include the full file_path, source_code, module_summary, "
                        "and all test_targets from the design result in the task.\n"
                        "       The result contains the generated pytest code.\n\n"
                        "3. FINALIZE\n"
                        "   Once ALL files have been processed, call return_unit_test_generation_result\n"
                        "   with the complete list of generated test results and a short summary.\n\n"
                        "RULES\n"
                        "- Process every discovered file — do not skip any\n"
                        "- Each task string must be fully self-contained\n"
                        "- Do not call return_unit_test_generation_result until all files are done\n"
                        "- Do not call terminate — use return_unit_test_generation_result to finish"
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

    def run_unit_test_generation(
        self,
        target_directory: str,
        action_context: ActionContext | None = None,
    ) -> UnitTestGenerationResult:
        """
        Entry point for unit test generation.
        Runs the coordinator agent loop and assembles the final result.
        """
        logger.info("Starting unit test generation for: %s", target_directory)

        run_memory = self.run(
            user_input=(
                f"Generate pytest unit tests for all Python source files "
                f"in the following directory: {target_directory}"
            ),
            action_context=action_context,
        )

        return self._extract_unit_test_generation_result(
            run_memory=run_memory,
            target_directory=target_directory,
        )

    def _extract_unit_test_generation_result(
        self,
        run_memory,
        target_directory: str,
    ) -> UnitTestGenerationResult:
        for item in reversed(run_memory.get_memories()):
            if item.get("role") != "tool":
                continue

            try:
                payload = json.loads(item["content"])
            except (KeyError, json.JSONDecodeError):
                continue

            inner = payload.get("result", {})
            if inner.get("result_type") != "unit_test_generation_result":
                continue

            generated_tests = [
                GeneratedTestFile(
                    source_file_path=t["source_file_path"],
                    test_file_path=t["test_file_path"],
                    pytest_code=t["pytest_code"],
                )
                for t in inner.get("generated_tests", [])
            ]

            discovered_files = [
                SourceFile(
                    path=t.source_file_path,
                    content="",
                    file_type="python",
                )
                for t in generated_tests
            ]

            return UnitTestGenerationResult(
                target_directory=target_directory,
                discovered_files=discovered_files,
                test_designs=[],
                generated_tests=generated_tests,
                summary=inner.get("summary", ""),
            )

        logger.warning(
            "Coordinator did not produce a structured result — returning empty result."
        )
        return UnitTestGenerationResult(
            target_directory=target_directory,
            summary="Generation did not complete successfully.",
        )

    def _write_debug_snapshot(
        self,
        source_file_path: str,
        test_design: TestDesignResult,
        generated_test: GeneratedTestFile,
    ) -> None:
        debug_dir = Path("debug/unit_test_generation")
        debug_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(source_file_path)
        safe_stem = "_".join([*source_path.parent.parts, source_path.stem]).lstrip("_")

        design_path = debug_dir / f"{safe_stem}_design.json"
        generated_test_path = debug_dir / f"{safe_stem}_generated_test.py"

        design_path.write_text(
            json.dumps(asdict(test_design), indent=2),
            encoding="utf-8",
        )
        generated_test_path.write_text(
            generated_test.pytest_code,
            encoding="utf-8",
        )
