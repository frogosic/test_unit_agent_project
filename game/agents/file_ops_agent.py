from __future__ import annotations

import json
from pathlib import Path

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
                        "Inspect a requested file or directory, identify relevant Python source files, "
                        "and read only the files needed to support downstream analysis.\n\n"
                        "Rules:\n"
                        "- If the user provides a specific Python file path, focus on that file only\n"
                        "- Avoid irrelevant files such as __init__.py, caches, virtual environments, and test files unless explicitly needed\n"
                        "- Prefer targeted inspection over broad exploration\n"
                        "- Ground conclusions in actual file contents, not filenames alone\n"
                        "- Stay within the allowed file-access policy\n"
                        "- When enough relevant files have been read, stop"
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
        memories = memory.get_memories()

        requested_path: str | None = None
        read_results_by_tool_id: dict[str, str] = {}
        tool_name_by_id: dict[str, str] = {}
        tool_args_by_id: dict[str, dict] = {}

        for item in memories:
            if item.get("role") == "user" and requested_path is None:
                requested_path = self._extract_requested_path(item.get("content", ""))

            if item.get("role") == "assistant":
                for tool_call in item.get("tool_calls", []):
                    tool_id = tool_call["id"]
                    tool_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"]

                    try:
                        parsed_args = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                    except json.JSONDecodeError:
                        parsed_args = {}

                    tool_name_by_id[tool_id] = tool_name
                    tool_args_by_id[tool_id] = parsed_args or {}

            elif item.get("role") == "tool":
                tool_id = item.get("tool_call_id")
                if not tool_id:
                    continue

                try:
                    payload = json.loads(item["content"])
                except (KeyError, json.JSONDecodeError):
                    continue

                if not payload.get("tool_executed", False):
                    continue

                if tool_name_by_id.get(tool_id) == "read_file":
                    result = payload.get("result", "")
                    if isinstance(result, str):
                        read_results_by_tool_id[tool_id] = result

        files: list[SourceFile] = []

        for tool_id, content in read_results_by_tool_id.items():
            args = tool_args_by_id.get(tool_id, {})
            file_name = args.get("file_name")
            if not file_name:
                continue

            if not file_name.endswith(".py"):
                continue

            if Path(file_name).name == "__init__.py":
                continue

            files.append(
                SourceFile(
                    path=file_name,
                    content=content,
                    file_type="python",
                )
            )

        if not files:
            raise ValueError(
                "FileOpsAgent did not produce any readable Python source files."
            )

        target_directory = requested_path or str(Path(files[0].path).parent)
        return FileDiscoveryResult(
            target_directory=target_directory,
            files=files,
        )

    def _extract_requested_path(self, user_input: str) -> str | None:
        marker = "'"
        if marker in user_input:
            parts = user_input.split(marker)
            if len(parts) >= 2:
                return parts[1].strip()
        return None
