#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from gitignore_utils import GitignoreError, is_gitignored, load_gitignore_spec

CLUTTER_DIR_PATTERNS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".ipynb_checkpoints",
    ".eggs",
    "*.egg-info",
    "build",
    "dist",
)
CLUTTER_FILE_PATTERNS: tuple[str, ...] = ("*.pyc", "*.pyo", "*.pyd")


def run_clean(
    project_root: Path,
    target_arg: str | None,
    format_script: Path,
    flake8_template: Path,
) -> None:
    try:
        gitignore_spec = load_gitignore_spec(project_root)
    except GitignoreError as exc:
        print(f"[pyrefine] {exc}", file=sys.stderr)
        sys.exit(1)
    target = target_arg if target_arg is not None else "."
    clean_target(
        project_root=project_root,
        target_arg=target,
        gitignore_spec=gitignore_spec,
        format_script=format_script,
        flake8_template=flake8_template,
    )


def clean_target(
    project_root: Path,
    target_arg: str,
    gitignore_spec,
    format_script: Path,
    flake8_template: Path,
) -> None:
    target_path = ensure_absolute(Path(target_arg), project_root)
    if is_gitignored(target_path, project_root, gitignore_spec):
        print(
            f"[pyrefine] '{target_path}' is ignored via .gitignore; skipping."
        )
        return
    if target_arg == "." or target_path == project_root:
        remove_clutter(project_root, project_root, gitignore_spec)
        ensure_flake8_config(project_root, flake8_template)
        run_formatter("all", project_root, format_script)
        return

    if not target_path.exists():
        raise FileNotFoundError(f"Target '{target_path}' does not exist.")

    if target_path.is_file():
        if target_path.suffix != ".py":
            raise ValueError("Only Python files can be formatted directly.")
        run_formatter(str(target_path), project_root, format_script)
        return

    if target_path.is_dir():
        remove_clutter(target_path, project_root, gitignore_spec)
        ensure_flake8_config(project_root, flake8_template)
        run_formatter(str(target_path), project_root, format_script)
        return

    raise ValueError(f"Unsupported target: {target_path}")


def ensure_absolute(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def ensure_flake8_config(project_root: Path, flake8_template: Path) -> None:
    if not flake8_template.exists():
        return
    target = project_root / ".flake8"
    if not target.exists():
        shutil.copy2(flake8_template, target)


def remove_clutter(path: Path, project_root: Path, gitignore_spec) -> None:
    removed_any = False
    for pattern in CLUTTER_DIR_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_dir() and not is_gitignored(
                item, project_root, gitignore_spec
            ):
                shutil.rmtree(item, ignore_errors=True)
                removed_any = True
    for pattern in CLUTTER_FILE_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_file() and not is_gitignored(
                item, project_root, gitignore_spec
            ):
                try:
                    item.unlink()
                    removed_any = True
                except OSError:
                    continue
    if removed_any:
        print("Removed cache/build artefacts before formatting.")


def run_formatter(
    target: str,
    project_root: Path,
    format_script: Path,
) -> None:
    if not format_script.exists():
        print("Formatter script missing from resources; skipping formatting.")
        return

    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)

    if hasattr(sys, "_MEIPASS"):
        previous_environ = os.environ.copy()
        previous_argv = sys.argv[:]
        try:
            os.environ.update(env)
            sys.argv = [str(format_script), target]
            print(f"Running formatter via embedded script: {format_script}")
            runpy.run_path(str(format_script), run_name="__main__")
        finally:
            os.environ.clear()
            os.environ.update(previous_environ)
            sys.argv = previous_argv
    else:
        command = [sys.executable, str(format_script), target]
        print(f"Running formatter: {' '.join(command)}")
        subprocess.run(command, check=False, env=env)
