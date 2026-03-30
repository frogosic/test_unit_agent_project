from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile

from game.models.unit_test_models import GeneratedTestFile

logger = logging.getLogger(__name__)


def _validate_imports(code: str, source_path: str) -> str | None:
    """
    Validate that the generated code has no import errors.
    Runs a syntax check first, then attempts a dry-run import.
    Returns None if clean, or an error string describing the problem.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error in generated code: {e}"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="test_import_validate_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import importlib.util; spec = importlib.util.spec_from_file_location('_validate', '{tmp_path}'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )

        if result.returncode == 0:
            return None

        error = (result.stdout + result.stderr).strip()
        logger.warning("Import validation failed for %s:\n%s", source_path, error)
        return f"Import error in generated code:\n{error}"

    except subprocess.TimeoutExpired:
        return "Import validation timed out."
    except Exception as e:
        return f"Import validation error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def validate_generated_test(generated: GeneratedTestFile) -> str | None:
    """
    Validate a generated test file by checking imports then running pytest.
    Returns None if all tests pass, or an error string describing the failures.
    """
    import_error = _validate_imports(generated.pytest_code, generated.source_file_path)
    if import_error is not None:
        return import_error

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
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )

        if result.returncode == 0:
            logger.info("Pytest validation passed for: %s", generated.source_file_path)
            return None

        output = (result.stdout + result.stderr).strip()
        logger.warning(
            "Pytest validation failed for %s:\n%s",
            generated.source_file_path,
            output,
        )
        logger.debug("Generated test code that failed:\n%s", generated.pytest_code)
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
