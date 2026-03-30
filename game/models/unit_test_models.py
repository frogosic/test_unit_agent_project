from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FileProcessingStep:
    file_path: str
    status: Literal["pending", "designing", "writing", "complete", "failed"] = "pending"
    test_design: TestDesignResult | None = None
    generated_test: GeneratedTestFile | None = None
    error: str | None = None


@dataclass
class UnitTestGenerationPlan:
    target: str
    steps: list[FileProcessingStep] = field(default_factory=list)
    status: Literal["planning", "executing", "complete", "failed"] = "planning"

    def pending_steps(self) -> list[FileProcessingStep]:
        return [s for s in self.steps if s.status == "pending"]

    def completed_steps(self) -> list[FileProcessingStep]:
        return [s for s in self.steps if s.status == "complete"]

    def failed_steps(self) -> list[FileProcessingStep]:
        return [s for s in self.steps if s.status == "failed"]


@dataclass
class SourceFile:
    path: str
    content: str
    file_type: Literal["python"] = "python"


@dataclass
class FileDiscoveryResult:
    target_directory: str
    files: list[SourceFile] = field(default_factory=list)


@dataclass
class TestScenario:
    name: str
    description: str
    inputs: list[str]
    assertions: list[str]
    mock_targets: list[str]


@dataclass
class TestTarget:
    name: str
    target_type: Literal["function", "class", "method"]
    dependencies_to_mock: list[str] = field(default_factory=list)
    scenarios: list[TestScenario] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class TestWritingTask:
    source_file_path: str
    module_summary: str
    test_targets: list[TestTarget]
    source_code: str = ""


@dataclass
class TestDesignResult:
    file_path: str
    module_summary: str
    test_targets: list[TestTarget] = field(default_factory=list)


@dataclass
class TestDesignTask:
    file_path: str
    source_code: str = ""


@dataclass
class GeneratedTestFile:
    source_file_path: str
    test_file_path: str
    pytest_code: str


@dataclass
class UnitTestGenerationResult:
    target_directory: str
    discovered_files: list[SourceFile] = field(default_factory=list)
    test_designs: list[TestDesignResult] = field(default_factory=list)
    generated_tests: list[GeneratedTestFile] = field(default_factory=list)
    summary: str = ""
