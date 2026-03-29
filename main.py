from __future__ import annotations

import json

from game.bootstrap.all_actions import (
    build_coordinator_action_registry,
    build_file_ops_action_registry,
    build_reporting_action_registry,
    build_test_design_action_registry,
)
from game.bootstrap.agent_system import build_agent_system
from game.core.environment import Environment
from game.languages.tool_calling import ToolCallingAgentLanguage
from game.core.llm import LLM
from game.core.memory import Memory
from game.policies.file_security_policy import FileSecurityPolicy


def main() -> None:
    llm = LLM(model="openai/gpt-4o-mini")
    environment = Environment()
    file_security_policy = FileSecurityPolicy()
    agent_language = ToolCallingAgentLanguage()

    coordinator_action_registry = build_coordinator_action_registry()
    file_ops_action_registry = build_file_ops_action_registry()
    test_design_action_registry = build_test_design_action_registry()
    reporting_action_registry = build_reporting_action_registry()

    agent_system = build_agent_system(
        agent_language=agent_language,
        llm=llm,
        environment=environment,
        file_security_policy=file_security_policy,
        coordinator_action_registry=coordinator_action_registry,
        file_ops_action_registry=file_ops_action_registry,
        test_design_action_registry=test_design_action_registry,
        reporting_action_registry=reporting_action_registry,
    )

    user_task = input("What would you like me to do? ").strip()

    memory = agent_system.coordinator.run(
        user_input=user_task,
        memory=Memory(),
        action_context=agent_system.root_action_context,
    )

    print("\nFinal memory:")
    for item in memory.get_memories():
        print(json.dumps(item, indent=2, ensure_ascii=False))

    output_data = {"memories": memory.get_memories()}
    with open("final_memory.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print("\nAll memories have been written to final_memory.json.")


if __name__ == "__main__":
    main()
