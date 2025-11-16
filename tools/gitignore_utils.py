#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from pathspec import PathSpec


class GitignoreError(RuntimeError):
    """Raised when .gitignore is missing or invalid."""


def gitignore_path(project_root: Path) -> Path:
    return project_root / ".gitignore"


def load_gitignore_spec(project_root: Path) -> PathSpec:
    path = gitignore_path(project_root)
    if not path.exists():
        raise GitignoreError(
            f".gitignore not found in {project_root}. "
            "Run 'pyrefine --create' to scaffold one."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    return PathSpec.from_lines("gitwildmatch", lines)


def is_gitignored(path: Path, project_root: Path, spec: PathSpec | None) -> bool:
    if spec is None:
        return False
    try:
        relative = path.resolve().relative_to(project_root)
    except ValueError:
        return False
    if not relative.parts:
        return False
    return spec.match_file(relative.as_posix())

