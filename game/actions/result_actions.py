from __future__ import annotations

from typing import Any

from game.core.core_action import Action


def return_file_discovery_result(
    target_directory: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool_executed": True,
        "success": True,
        "result_type": "file_discovery",
        "target_directory": target_directory,
        "files": files,
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
                                },
                                "required": ["name", "description"],
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
