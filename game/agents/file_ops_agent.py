from __future__ import annotations

import json

from game.actions.action_context import ActionContext
from game.agents.base_agent import BaseAgent
from game.core.core_action import ActionRegistry, Goal
from game.core.environment import Environment
from game.core.llm import LLM
from game.core.memory import Memory
from game.languages.tool_calling import AgentLanguage
from game.models.unit_test_models import FileDiscoveryResult, SourceFile
from game.policies.file_security_policy import FileSecurityPolicy


class FileOpsAgent(BaseAgent):
    """
    Specialist agent responsible for safe file discovery and file reading
    within the allowed project scope.
    """

    AGENT_NAME = "file_ops_agent"

    def __init__(
        self,
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        llm: LLM,
        environment: Environment,
        file_security_policy: FileSecurityPolicy,
    ) -> None:
        super().__init__(
            name=self.AGENT_NAME,
            goals=[
                Goal(
                    priority=1,
                    name="inspect_source_files",
                    description=(
                        "Inspect a requested directory, identify relevant Python source files, "
                        "and read only the files needed to support downstream analysis.\n\n"
                        "Rules:\n"
                        "- Focus on Python source files relevant to the task\n"
                        "- Avoid irrelevant files such as caches, virtual environments, and test files unless explicitly needed\n"
                        "- Prefer targeted inspection over broad exploration\n"
                        "- Ground conclusions in actual file contents, not filenames alone\n"
                        "- Stay within the allowed file-access policy\n"
                        "- When the task is complete, you must finish by calling "
                        "`return_file_discovery_result` with the final structured result"
                    ),
                )
            ],
            agent_language=agent_language,
            action_registry=action_registry,
            llm=llm,
            environment=environment,
            file_security_policy=file_security_policy,
        )

    def run_and_parse(
        self,
        user_input: str,
        memory: Memory | None = None,
        max_iterations: int = 20,
        action_context: ActionContext | None = None,
    ) -> FileDiscoveryResult:
        run_memory = self.run(
            user_input=user_input,
            memory=memory,
            max_iterations=max_iterations,
            action_context=action_context,
        )
        return self._extract_file_discovery_result(run_memory)

    def _extract_file_discovery_result(self, memory: Memory) -> FileDiscoveryResult:
        for item in reversed(memory.get_memories()):
            if item.get("role") != "tool":
                continue

            try:
                payload = json.loads(item["content"])
            except (KeyError, json.JSONDecodeError):
                continue

            if payload.get("result_type") != "file_discovery":
                continue

            files = [
                SourceFile(
                    path=file_data["path"],
                    content=file_data["content"],
                    file_type=file_data.get("file_type", "python"),
                )
                for file_data in payload.get("files", [])
            ]

            return FileDiscoveryResult(
                target_directory=payload["target_directory"],
                files=files,
            )

        raise ValueError("FileOpsAgent did not produce a valid file_discovery result.")
