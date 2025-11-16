#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

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
    ignored_dirs = load_gitignore_directories(project_root)
    target = target_arg if target_arg is not None else "."
    clean_target(
        project_root=project_root,
        target_arg=target,
        ignored_dirs=ignored_dirs,
        format_script=format_script,
        flake8_template=flake8_template,
    )


def clean_target(
    project_root: Path,
    target_arg: str,
    ignored_dirs: set[str],
    format_script: Path,
    flake8_template: Path,
) -> None:
    target_path = ensure_absolute(Path(target_arg), project_root)
    if target_arg == "." or target_path == project_root:
        remove_clutter(project_root, project_root, ignored_dirs)
        ensure_flake8_config(project_root, flake8_template)
        run_formatter("all", project_root, format_script, ignored_dirs)
        return

    if not target_path.exists():
        raise FileNotFoundError(f"Target '{target_path}' does not exist.")

    if target_path.is_file():
        if target_path.suffix != ".py":
            raise ValueError("Only Python files can be formatted directly.")
        if path_in_gitignore(target_path, project_root, ignored_dirs):
            print(
                f"[pyrefine] '{target_path}' is ignored via .gitignore; skipping."
            )
            return
        run_formatter(
            str(target_path), project_root, format_script, ignored_dirs
        )
        return

    if target_path.is_dir():
        if path_in_gitignore(target_path, project_root, ignored_dirs):
            print(
                f"[pyrefine] '{target_path}' is ignored via .gitignore; skipping."
            )
            return
        remove_clutter(target_path, project_root, ignored_dirs)
        ensure_flake8_config(project_root, flake8_template)
        run_formatter(
            str(target_path), project_root, format_script, ignored_dirs
        )
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


def remove_clutter(
    path: Path, project_root: Path, ignored_dirs: set[str]
) -> None:
    removed_any = False
    for pattern in CLUTTER_DIR_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_dir() and not path_in_gitignore(
                item, project_root, ignored_dirs
            ):
                shutil.rmtree(item, ignore_errors=True)
                removed_any = True
    for pattern in CLUTTER_FILE_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_file() and not path_in_gitignore(
                item, project_root, ignored_dirs
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
    ignored_dirs: set[str] | None = None,
) -> None:
    if not format_script.exists():
        print("Formatter script missing from resources; skipping formatting.")
        return

    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)
    if ignored_dirs:
        env["PYREFINE_IGNORE_DIRS"] = ",".join(sorted(ignored_dirs))

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


def load_gitignore_directories(project_root: Path) -> set[str]:
    gitignore = project_root / ".gitignore"
    if not gitignore.exists():
        print(
            "[pyrefine] .gitignore not found in "
            f"{project_root}. Please create one before running --clean.",
            file=sys.stderr,
        )
        sys.exit(1)
    ignored: set[str] = set()
    for raw_line in gitignore.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(ch in line for ch in {"*", "!", "?", "[", "]"}):
            continue
        line = line.strip("/\\")
        if not line:
            continue
        first_component = line.split("/", 1)[0].split("\\", 1)[0]
        if first_component:
            ignored.add(first_component)
    return ignored


def path_in_gitignore(
    path: Path, project_root: Path, ignored_dirs: set[str]
) -> bool:
    try:
        relative = path.resolve().relative_to(project_root)
    except ValueError:
        return False
    if not relative.parts:
        return False
    return relative.parts[0] in ignored_dirs

