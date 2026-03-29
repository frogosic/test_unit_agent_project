from __future__ import annotations

import json
from dataclasses import asdict

from game.bootstrap.all_actions import (
    build_coordinator_action_registry,
    build_file_ops_action_registry,
    build_test_design_action_registry,
    build_test_writing_action_registry,
)
from game.bootstrap.agent_system import build_agent_system
from game.core.environment import Environment
from game.languages.tool_calling import ToolCallingAgentLanguage
from game.core.llm import LLM
from game.policies.file_security_policy import FileSecurityPolicy


def main() -> None:
    llm = LLM(model="openai/gpt-4o-mini")
    environment = Environment()
    file_security_policy = FileSecurityPolicy()
    agent_language = ToolCallingAgentLanguage()

    coordinator_action_registry = build_coordinator_action_registry()
    file_ops_action_registry = build_file_ops_action_registry()
    test_design_action_registry = build_test_design_action_registry()
    test_writing_action_registry = build_test_writing_action_registry()

    agent_system = build_agent_system(
        agent_language=agent_language,
        llm=llm,
        environment=environment,
        file_security_policy=file_security_policy,
        coordinator_action_registry=coordinator_action_registry,
        file_ops_action_registry=file_ops_action_registry,
        test_design_action_registry=test_design_action_registry,
        test_writing_action_registry=test_writing_action_registry,
    )

    target_directory = input(
        "Which directory should I inspect for unit test generation? "
    ).strip()

    result = agent_system.coordinator.run_unit_test_generation(
        target_directory=target_directory,
        action_context=agent_system.root_action_context,
    )

    print("\n=== UNIT TEST GENERATION SUMMARY ===")
    print(result.summary)
    print(f"Discovered files: {len(result.discovered_files)}")
    print(f"Test designs: {len(result.test_designs)}")
    print(f"Generated tests: {len(result.generated_tests)}")

    if result.discovered_files:
        print("\n=== FIRST DISCOVERED FILE ===")
        print(result.discovered_files[0].path)

    if result.test_designs:
        first_design = result.test_designs[0]
        print("\n=== FIRST TEST DESIGN ===")
        print(f"File: {first_design.file_path}")
        print(f"Summary: {first_design.module_summary}")
        print(f"Targets: {len(first_design.test_targets)}")

    if result.generated_tests:
        first_generated = result.generated_tests[0]
        print("\n=== FIRST GENERATED TEST FILE ===")
        print(f"Path: {first_generated.test_file_path}")
        print(first_generated.pytest_code[:1500])

    output_data = asdict(result)
    with open("unit_test_generation_result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\nStructured result has been written to unit_test_generation_result.json.")


if __name__ == "__main__":
    main()
