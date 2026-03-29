from dataclasses import dataclass, field
from typing import Literal


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
    source_code: str
    module_summary: str
    test_targets: list[TestTarget]


@dataclass
class TestDesignResult:
    file_path: str
    module_summary: str
    test_targets: list[TestTarget] = field(default_factory=list)


@dataclass
class TestDesignTask:
    file_path: str
    source_code: str


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
