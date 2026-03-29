from __future__ import annotations

from pathlib import Path


class FileSecurityPolicy:
    """Controls safe filesystem access within a restricted working directory."""

    def __init__(
        self,
        base_dir: Path | None = None,
        restricted_names: set[str] | None = None,
        restricted_dirs: set[str] | None = None,
    ) -> None:
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.restricted_names = restricted_names or {".env"}
        self.restricted_dirs = restricted_dirs or {
            ".venv",
            "__pycache__",
            ".git",
        }

    def _resolve_path(self, path_name: str) -> Path:
        """Resolve a path name relative to the base directory."""
        return (self.base_dir / path_name).resolve()

    def _is_within_base_dir(self, path: Path) -> bool:
        """Check whether a resolved path stays inside the base directory."""
        return self.base_dir == path or self.base_dir in path.parents

    def _ensure_safe_path(self, path_name: str) -> Path:
        """Ensure a path stays within the allowed base directory and avoids restricted names."""
        path = self._resolve_path(path_name)

        if not self._is_within_base_dir(path):
            raise ValueError("Access outside working directory is forbidden.")

        if any(part in self.restricted_dirs for part in path.parts):
            raise ValueError("Access to restricted directories is forbidden.")

        if path.name in self.restricted_names:
            raise ValueError("Access to restricted files is forbidden.")

        return path

    def ensure_safe_read_path(self, file_name: str) -> Path:
        """Ensure a file path is safe and valid for reading."""
        path = self._ensure_safe_path(file_name)

        if not path.exists():
            raise FileNotFoundError(f"{file_name} not found.")

        if not path.is_file():
            raise ValueError(f"{file_name} is not a file.")

        return path

    def ensure_safe_write_path(self, file_name: str) -> Path:
        """Ensure a file path is safe and valid for writing."""
        path = self._ensure_safe_path(file_name)

        if path.exists() and not path.is_file():
            raise ValueError(f"{file_name} is not a file.")

        return path

    def ensure_safe_directory_path(self, directory_name: str = ".") -> Path:
        """Ensure a directory path is safe and valid to inspect."""
        path = self._ensure_safe_path(directory_name)

        if not path.exists():
            raise FileNotFoundError(f"{directory_name} not found.")

        if not path.is_dir():
            raise ValueError(f"{directory_name} is not a directory.")

        return path

    def should_hide_name(self, name: str) -> bool:
        """Check whether a file or directory name should be hidden from listings."""
        return name in self.restricted_dirs or name in self.restricted_names
