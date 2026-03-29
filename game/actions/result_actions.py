from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from game.actions.action_context import ActionContext
from game.core.core_action import Action
from game.models.unit_test_models import GeneratedTestFile
from game.policies.file_security_policy import FileSecurityPolicy
from game.services.generated_test_file_writer import write_generated_test_file

logger = logging.getLogger(__name__)


def _write_all_test_files(
    generated_tests: list[dict[str, Any]],
    file_security_policy: FileSecurityPolicy,
) -> list[str]:
    """
    Write all generated test files to disk.
    Returns list of successfully written paths.
    """
    written: list[str] = []

    for test in generated_tests:
        try:
            generated = GeneratedTestFile(
                source_file_path=test["source_file_path"],
                test_file_path=test["test_file_path"],
                pytest_code=test["pytest_code"],
            )
            write_generated_test_file(
                generated_test=generated,
                file_security_policy=file_security_policy,
            )
            written.append(test["test_file_path"])
            logger.info("Written: %s", test["test_file_path"])
        except Exception as e:
            logger.warning(
                "Failed to write test file %s: %s",
                test.get("test_file_path", "unknown"),
                e,
            )

    return written


def _write_debug_snapshots(
    generated_tests: list[dict[str, Any]],
) -> None:
    """
    Write debug snapshots for each generated test file.
    Failures are logged but do not interrupt the finalization step.
    """
    debug_dir = Path("debug/unit_test_generation")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for test in generated_tests:
        try:
            source_path = Path(test["source_file_path"])
            safe_stem = "_".join([*source_path.parent.parts, source_path.stem]).lstrip(
                "_"
            )

            (debug_dir / f"{safe_stem}_generated_test.py").write_text(
                test["pytest_code"],
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(
                "Failed to write debug snapshot for %s: %s",
                test.get("source_file_path", "unknown"),
                e,
            )


def _return_unit_test_generation_result(
    action_context: ActionContext,
    generated_tests: list[dict[str, Any]],
    summary: str,
) -> dict[str, Any]:
    policy = action_context.get("file_security_policy")
    if policy is None:
        raise ValueError("Missing file_security_policy in action_context.")

    written = _write_all_test_files(generated_tests, policy)
    _write_debug_snapshots(generated_tests)

    return {
        "result_type": "unit_test_generation_result",
        "summary": summary,
        "generated_tests": generated_tests,
        "written_files": written,
        "total_written": len(written),
    }


def return_test_design_result(
    file_path: str,
    module_summary: str,
    test_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool_executed": True,
        "success": True,
        "result_type": "test_design",
        "file_path": file_path,
        "module_summary": module_summary,
        "test_targets": test_targets,
    }


def return_generated_test_file(
    source_file_path: str,
    test_file_path: str,
    pytest_code: str,
) -> dict[str, Any]:
    return {
        "tool_executed": True,
        "success": True,
        "result_type": "generated_test_file",
        "source_file_path": source_file_path,
        "test_file_path": test_file_path,
        "pytest_code": pytest_code,
    }


RETURN_TEST_DESIGN_RESULT_ACTION = Action(
    name="return_test_design_result",
    function=return_test_design_result,
    description=(
        "Return the final structured unit test design for a source file. "
        "Use this only when test design is complete."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The analyzed source file path.",
            },
            "module_summary": {
                "type": "string",
                "description": "A concise summary of the module.",
            },
            "test_targets": {
                "type": "array",
                "description": "Structured unit test targets for the file.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "target_type": {"type": "string"},
                        "dependencies_to_mock": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "scenarios": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "inputs": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "assertions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "mock_targets": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "name",
                                    "description",
                                    "inputs",
                                    "assertions",
                                    "mock_targets",
                                ],
                            },
                        },
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "target_type", "scenarios"],
                },
            },
        },
        "required": ["file_path", "module_summary", "test_targets"],
    },
    terminal=True,
)


RETURN_GENERATED_TEST_FILE_ACTION = Action(
    name="return_generated_test_file",
    function=return_generated_test_file,
    description=(
        "Return the final generated pytest unit test file content. "
        "Use this only when test generation is complete."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source_file_path": {
                "type": "string",
                "description": "The original source file path.",
            },
            "test_file_path": {
                "type": "string",
                "description": "The destination path for the generated test file.",
            },
            "pytest_code": {
                "type": "string",
                "description": "The generated pytest code.",
            },
        },
        "required": ["source_file_path", "test_file_path", "pytest_code"],
    },
    terminal=True,
)


RETURN_UNIT_TEST_GENERATION_RESULT_ACTION = Action(
    name="return_unit_test_generation_result",
    function=_return_unit_test_generation_result,
    description=(
        "Signal completion of the unit test generation workflow. "
        "Call this once ALL source files have been processed — "
        "after calling test_design_agent and test_writing_agent for every discovered file. "
        "Provide the full list of generated test results. "
        "This action writes all test files to disk and finalizes the run."
    ),
    parameters={
        "type": "object",
        "properties": {
            "generated_tests": {
                "type": "array",
                "description": (
                    "List of all generated test file results. "
                    "Each entry must come from a test_writing_agent result."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source_file_path": {"type": "string"},
                        "test_file_path": {"type": "string"},
                        "pytest_code": {"type": "string"},
                    },
                    "required": ["source_file_path", "test_file_path", "pytest_code"],
                },
            },
            "summary": {
                "type": "string",
                "description": "Short human-readable summary of what was generated.",
            },
        },
        "required": ["generated_tests", "summary"],
        "additionalProperties": False,
    },
    terminal=True,
)
