from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

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

    logger.info("Starting unit test generation for directory: %s", target_directory)

    result = agent_system.coordinator.run_unit_test_generation(
        target_directory=target_directory,
        action_context=agent_system.root_action_context,
    )

    logger.info(
        "Generation complete — files: %d, designs: %d, tests: %d",
        len(result.discovered_files),
        len(result.test_designs),
        len(result.generated_tests),
    )
    logger.info("Summary: %s", result.summary)

    output_data = asdict(result)
    with open("unit_test_generation_result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Structured result written to unit_test_generation_result.json")


if __name__ == "__main__":
    main()
