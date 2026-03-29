from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile

from game.models.unit_test_models import GeneratedTestFile

logger = logging.getLogger(__name__)


def validate_generated_test(generated: GeneratedTestFile) -> str | None:
    """
    Validate a generated test file by running pytest on it.

    Returns None if all tests pass, or an error string describing the failures.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="test_validate_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(generated.pytest_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                tmp_path,
                "-v",
                "--tb=short",
                "--no-header",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info("Pytest validation passed for: %s", generated.source_file_path)
            return None

        output = (result.stdout + result.stderr).strip()
        logger.warning(
            "Pytest validation failed for %s:\n%s",
            generated.source_file_path,
            output[:500],
        )
        return output

    except subprocess.TimeoutExpired:
        return "Pytest timed out after 30 seconds."
    except Exception as e:
        return f"Pytest runner error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
