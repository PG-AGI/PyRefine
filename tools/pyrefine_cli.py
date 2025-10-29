#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.setup_workspace import (  # type: ignore
    build_extensions,
    build_settings,
    formatter_reference,
)


PYREFINE_ROOT = Path(__file__).resolve().parents[1]
FORMAT_SCRIPT = PYREFINE_ROOT / "tools" / "format.py"


ESSENTIAL_DIR_NAMES: set[str] = {"concrete_tools", "src", "utils", "tests", "app", "services"}
EXCLUDE_ROOT_NAMES: set[str] = {"PyRefine", ".venv", "venv", "env"}
CLUTTER_DIR_PATTERNS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".ipynb_checkpoints",
    ".idea",
    "build",
    "dist",
    ".eggs",
    "*.egg-info",
)
CLUTTER_FILE_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyRefine interactive CLI for repository setup and formatting."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Path to the target project root (defaults to the current working directory).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept defaults for prompts when possible.",
    )
    return parser.parse_args()


def resolve_project_root(value: Path | None) -> Path:
    if value:
        return value.resolve()
    return Path.cwd()


def prompt_choice(
    prompt: str,
    choices: list[str],
    default: int | None = None,
    auto_accept: bool = False,
) -> int:
    if auto_accept:
        if default is not None:
            return default
        return 1
    while True:
        for idx, label in enumerate(choices, start=1):
            print(f"{idx}. {label}")
        suffix = f" [{default}]" if default is not None else ""
        selection = input(f"{prompt}{suffix}: ").strip()
        if not selection:
            if default is not None:
                return default
            continue
        if selection.isdigit():
            numeric = int(selection)
            if 1 <= numeric <= len(choices):
                return numeric
        print("Please enter a valid option number.")


def is_under_excluded(path: Path, project_root: Path) -> bool:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return False
    return any(part in EXCLUDE_ROOT_NAMES for part in relative.parts)


def should_preserve(path: Path, project_root: Path, preserve_essential: bool) -> bool:
    if not preserve_essential:
        return False
    try:
        top_level = path.relative_to(project_root).parts[0]
    except (ValueError, IndexError):
        return False
    return top_level in ESSENTIAL_DIR_NAMES


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            return


def clean_repository(
    project_root: Path,
    preserve_essential: bool,
) -> None:
    removed = []
    for pattern in CLUTTER_DIR_PATTERNS:
        for directory in project_root.glob(f"**/{pattern}"):
            if not directory.exists():
                continue
            if is_under_excluded(directory, project_root):
                continue
            if should_preserve(directory, project_root, preserve_essential):
                continue
            removed.append(directory)
            remove_path(directory)
    for pattern in CLUTTER_FILE_PATTERNS:
        for file_path in project_root.glob(f"**/{pattern}"):
            if not file_path.exists():
                continue
            if is_under_excluded(file_path, project_root):
                continue
            removed.append(file_path)
            remove_path(file_path)
    if removed:
        print("Removed the following items:")
        for path in removed:
            print(f"  - {path.relative_to(project_root)}")
    else:
        print("No cleanup required; repository is already tidy.")


def ensure_virtual_environment(project_root: Path) -> None:
    candidates = [project_root / ".venv", project_root / "venv"]
    if any((env_dir / "Scripts").exists() or (env_dir / "bin").exists() for env_dir in candidates):
        return
    print("WARNING: No virtual environment detected (.venv/ or venv/).")
    print("         Consider creating one before running formatters.")


def merge_dict(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in updates.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dict(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                existing = result[key]
                for item in value:
                    if item not in existing:
                        existing.append(item)
                result[key] = existing
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def merge_run_on_save(existing: dict[str, object], project_root: Path) -> dict[str, object]:
    if "emeraldwalk.runonsave" not in existing:
        return existing
    commands = existing["emeraldwalk.runonsave"]
    if not isinstance(commands, dict):
        return existing
    command_list = commands.get("commands")
    if not isinstance(command_list, list):
        return existing
    expected_cmd = f'python "{formatter_reference(project_root)}" "${{file}}"'
    for command in command_list:
        if isinstance(command, dict) and command.get("cmd") == expected_cmd:
            return existing
    command_list.append(
        {
            "match": "\\.py$",
            "cmd": expected_cmd,
            "runIn": "terminal",
        }
    )
    return existing


def integrate_vscode_settings(project_root: Path) -> None:
    settings_dir = project_root / ".vscode"
    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    extensions_path = settings_dir / "extensions.json"

    desired_settings = build_settings(project_root)
    desired_extensions = build_extensions()

    if settings_path.exists():
        try:
            existing_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_settings = {}
        merged = merge_dict(existing_settings, desired_settings)
        merged = merge_run_on_save(merged, project_root)
        settings_path.write_text(json.dumps(merged, indent=4) + "\n", encoding="utf-8")
        print(f"Merged PyRefine settings into {settings_path}")
    else:
        settings_path.write_text(json.dumps(desired_settings, indent=4) + "\n", encoding="utf-8")
        print(f"Created VS Code settings at {settings_path}")

    if extensions_path.exists():
        try:
            existing_extensions = json.loads(extensions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_extensions = {}
        merged_extensions = merge_dict(existing_extensions, desired_extensions)
        extensions_path.write_text(
            json.dumps(merged_extensions, indent=4) + "\n",
            encoding="utf-8",
        )
        print(f"Merged PyRefine extension recommendations into {extensions_path}")
    else:
        extensions_path.write_text(
            json.dumps(desired_extensions, indent=4) + "\n",
            encoding="utf-8",
        )
        print(f"Created VS Code extension recommendations at {extensions_path}")


def run_formatter(target: str, project_root: Path, lint_only: bool = False) -> None:
    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)
    command: list[str] = [sys.executable, str(FORMAT_SCRIPT)]
    if lint_only:
        command.append("--lint-only")
    command.append(target)
    print(f"Running formatter: {' '.join(command)}")
    subprocess.run(command, check=False, env=env)


def prompt_formatting(project_root: Path, auto_accept: bool = False) -> None:
    options = [
        "Format a single Python file",
        "Format all files in a specific folder",
        "Format the entire repository",
        "Skip formatting",
    ]
    choice = prompt_choice(
        "Choose a formatting option",
        options,
        default=4,
        auto_accept=auto_accept,
    )
    if choice == 4:
        print("Skipping formatting.")
        return
    if choice == 1:
        path_input = input("Enter the absolute path to the Python file: ").strip()
        target_path = Path(path_input).expanduser()
        if not target_path.is_absolute():
            print("Please provide an absolute path.")
            return
        if not target_path.exists():
            print("The specified file does not exist.")
            return
        run_formatter(str(target_path), project_root)
    elif choice == 2:
        path_input = input("Enter the absolute path to the folder: ").strip()
        target_path = Path(path_input).expanduser()
        if not target_path.is_absolute():
            print("Please provide an absolute path.")
            return
        if not target_path.exists():
            print("The specified folder does not exist.")
            return
        run_formatter(str(target_path), project_root)
    elif choice == 3:
        run_formatter("all", project_root)


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    print(f"Detected project root: {project_root}")

    ensure_virtual_environment(project_root)

    print("\nStep 1 - Repository cleaning:")
    clean_options = [
        "Clean existing repository (remove caches, build artefacts, compiled files).",
        "Preserve essential folders (e.g. concrete_tools/, src/, utils/) while cleaning clutter.",
        "Skip cleaning.",
    ]
    clean_choice = prompt_choice(
        "Select a cleaning strategy",
        clean_options,
        default=2 if args.yes else None,
        auto_accept=args.yes,
    )
    if clean_choice == 1:
        clean_repository(project_root, preserve_essential=False)
    elif clean_choice == 2:
        clean_repository(project_root, preserve_essential=True)
    else:
        print("Skipping cleanup.")

    print("\nStep 2 - VS Code settings integration:")
    integrate_vscode_settings(project_root)

    print("\nStep 3 - Manual formatting options:")
    prompt_formatting(project_root, auto_accept=args.yes)

    print("\nPyRefine CLI complete.")


if __name__ == "__main__":
    main()


