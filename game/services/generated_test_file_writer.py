from __future__ import annotations

from game.models.unit_test_models import GeneratedTestFile
from game.policies.file_security_policy import FileSecurityPolicy


def write_generated_test_file(
    generated_test: GeneratedTestFile,
    file_security_policy: FileSecurityPolicy,
) -> None:
    path = file_security_policy.ensure_safe_write_path(generated_test.test_file_path)

    if path.suffix != ".py":
        raise ValueError(f"Invalid file type: {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_test.pytest_code, encoding="utf-8")
