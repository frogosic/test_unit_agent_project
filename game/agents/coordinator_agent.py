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
        max_iterations: int = 80,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="coordinate_unit_test_generation",
                    description=(
                        "Coordinate end-to-end pytest unit test generation for a target path.\n\n"
                        "WORKFLOW — follow this exact sequence:\n\n"
                        "1. DISCOVER FILES\n"
                        "   Call file_ops_agent with the target path.\n"
                        "   The result contains a list of source files, each with 'path' and 'content'.\n\n"
                        "2. FOR EACH SOURCE FILE — repeat steps 2a and 2b for every file:\n\n"
                        "   2a. DESIGN TESTS\n"
                        "       Call test_design_agent.\n"
                        "       The task MUST include:\n"
                        "       - FILE PATH: the exact file path\n"
                        "       - SOURCE CODE: the complete source code content from the discovery result\n\n"
                        "   2b. GENERATE TESTS\n"
                        "       Call test_writing_agent.\n"
                        "       The task MUST include:\n"
                        "       - FILE PATH: the exact file path\n"
                        "       - SOURCE CODE: the complete source code content\n"
                        "       - MODULE SUMMARY: from the test design result\n"
                        "       - TEST TARGETS: the full test_targets list from the test design result\n\n"
                        "3. FINALIZE\n"
                        "   Once ALL files are processed, call return_unit_test_generation_result\n"
                        "   with every generated test result and a short summary.\n\n"
                        "RULES\n"
                        "- Always include full source code in tasks — agents have no memory of prior calls\n"
                        "- Process every discovered file — do not skip any\n"
                        "- Do not call return_unit_test_generation_result until all files are done\n"
                        "- Do not call terminate — use return_unit_test_generation_result to finish\n"
                        "- If the target path is a single file, process only that file"
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
                f"Generate pytest unit tests for the following target path: {target_directory}\n\n"
                f"If it is a single Python file, process only that file.\n"
                f"If it is a directory, process all relevant Python source files found."
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
