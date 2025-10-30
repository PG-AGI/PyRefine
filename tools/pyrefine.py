#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PYREFINE_ROOT = Path(__file__).resolve().parents[1]
FORMAT_SCRIPT = PYREFINE_ROOT / "tools" / "format.py"

# Directories we treat as part of the canonical scaffold
TEMPLATE_DIRECTORIES: tuple[str, ...] = ("src", "tests", "configs", "scripts")
TEMPLATE_FILES: tuple[tuple[str, str], ...] = (
    ("src/__init__.py", ""),
    ("tests/__init__.py", ""),
    ("configs/.gitkeep", ""),
    ("scripts/.gitkeep", ""),
)

# Items to prune when cleaning a repository or folder
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
CLUTTER_FILE_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyRefine command-line utility for scaffolding, cleaning, and VS Code setup."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root folder (defaults to the current working directory).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the standard Python project scaffold (src/, tests/, configs/, scripts/).",
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const=".",
        metavar="PATH",
        help="Format a file, folder, or the entire project (default: current project).",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create or merge .vscode settings and extension recommendations.",
    )
    args = parser.parse_args()

    actions = sum(
        [
            1 if args.create else 0,
            1 if args.clean is not None else 0,
            1 if args.setup else 0,
        ]
    )
    if actions == 0:
        parser.error("Please specify one of --create, --clean, or --setup.")
    if actions > 1:
        parser.error("Please choose only one action at a time (--create, --clean, or --setup).")
    return args


def ensure_absolute(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def create_scaffold(project_root: Path) -> None:
    for directory in TEMPLATE_DIRECTORIES:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    for relative_path, contents in TEMPLATE_FILES:
        file_path = project_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(contents, encoding="utf-8")
    print(f"Created scaffold under {project_root}")


def remove_clutter(path: Path) -> None:
    removed_any = False
    for pattern in CLUTTER_DIR_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                removed_any = True
    for pattern in CLUTTER_FILE_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_file():
                try:
                    item.unlink()
                    removed_any = True
                except OSError:
                    continue
    if removed_any:
        print("Removed cache/build artefacts before formatting.")


def run_formatter(target: str, project_root: Path) -> None:
    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)
    command = [sys.executable, str(FORMAT_SCRIPT), target]
    print(f"Running formatter: {' '.join(command)}")
    subprocess.run(command, check=False, env=env)


def clean_target(project_root: Path, target_arg: str) -> None:
    target_path = ensure_absolute(Path(target_arg), project_root)
    if target_arg == "." or target_path == project_root:
        remove_clutter(project_root)
        run_formatter("all", project_root)
        return

    if not target_path.exists():
        raise FileNotFoundError(f"Target '{target_path}' does not exist.")

    if target_path.is_file():
        if target_path.suffix != ".py":
            raise ValueError("Only Python files can be formatted directly.")
        run_formatter(str(target_path), project_root)
        return

    if target_path.is_dir():
        remove_clutter(target_path)
        run_formatter(str(target_path), project_root)
        return

    raise ValueError(f"Unsupported target: {target_path}")


def build_settings_payload(project_root: Path) -> dict[str, object]:
    command = f'python "{formatter_reference(project_root)}" "${{file}}"'
    return {
        "editor.formatOnSave": True,
        "[python]": {
            "editor.formatOnSave": True,
            "editor.defaultFormatter": "ms-python.black-formatter",
            "editor.codeActionsOnSave": {
                "source.organizeImports": True,
                "source.fixAll": True,
            },
        },
        "python.languageServer": "Jedi",
        "python.terminal.activateEnvironment": True,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length", "79"],
        "python.sortImports.args": [
            "--profile",
            "black",
            "--line-length",
            "79",
        ],
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.mypyEnabled": False,
        "python.linting.banditEnabled": False,
        "python.linting.ignorePatterns": [
            "env/**",
            "**/__pycache__/**",
        ],
        "emeraldwalk.runonsave": {
            "commands": [
                {
                    "match": "\\.py$",
                    "cmd": command,
                    "runIn": "terminal",
                }
            ],
        },
    }


def build_extensions_payload() -> dict[str, object]:
    return {
        "recommendations": [
            "ms-python.python",
            "ms-python.black-formatter",
            "ms-python.isort",
            "ms-python.flake8",
            "emeraldwalk.runonsave",
        ],
        "unwantedRecommendations": [
            "ms-python.pylint",
            "ms-python.vscode-pylance",
            "charliermarsh.ruff",
        ],
    }


def formatter_reference(project_root: Path) -> str:
    try:
        relative = FORMAT_SCRIPT.relative_to(project_root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return FORMAT_SCRIPT.as_posix()


def merge_dict(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            combined = list(result[key])
            for item in value:
                if item not in combined:
                    combined.append(item)
            result[key] = combined
        else:
            result[key] = value
    return result


def merge_run_on_save(settings: dict[str, object], project_root: Path) -> dict[str, object]:
    commands_obj = settings.get("emeraldwalk.runonsave")
    if not isinstance(commands_obj, dict):
        return settings
    commands = commands_obj.get("commands")
    if not isinstance(commands, list):
        return settings
    desired_cmd = f'python "{formatter_reference(project_root)}" "${{file}}"'
    if any(isinstance(entry, dict) and entry.get("cmd") == desired_cmd for entry in commands):
        return settings
    commands.append(
        {
            "match": "\\.py$",
            "cmd": desired_cmd,
            "runIn": "terminal",
        }
    )
    return settings


def integrate_vscode(project_root: Path) -> None:
    settings_dir = project_root / ".vscode"
    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    extensions_path = settings_dir / "extensions.json"

    desired_settings = build_settings_payload(project_root)
    desired_extensions = build_extensions_payload()

    if settings_path.exists():
        try:
            existing_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_settings = {}
        merged_settings = merge_run_on_save(
            merge_dict(existing_settings, desired_settings),
            project_root,
        )
        settings_path.write_text(json.dumps(merged_settings, indent=4) + "\n", encoding="utf-8")
        print(f"Merged settings into {settings_path}")
    else:
        settings_path.write_text(json.dumps(desired_settings, indent=4) + "\n", encoding="utf-8")
        print(f"Created {settings_path}")

    if extensions_path.exists():
        try:
            existing_extensions = json.loads(extensions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_extensions = {}
        merged_extensions = merge_dict(existing_extensions, desired_extensions)
        extensions_path.write_text(json.dumps(merged_extensions, indent=4) + "\n", encoding="utf-8")
        print(f"Merged extensions into {extensions_path}")
    else:
        extensions_path.write_text(json.dumps(desired_extensions, indent=4) + "\n", encoding="utf-8")
        print(f"Created {extensions_path}")


def handle_create(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    create_scaffold(project_root)
    print("Scaffold complete. Run 'python PyRefine/tools/pyrefine.py setup' to configure VS Code.")


def handle_clean(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    try:
        target = args.clean if args.clean is not None else "."
        clean_target(project_root, target)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[pyrefine] {exc}", file=sys.stderr)
        sys.exit(1)


def handle_setup(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    integrate_vscode(project_root)


def main() -> None:
    args = parse_args()
    if args.create:
        handle_create(args)
    elif args.clean is not None:
        handle_clean(args)
    elif args.setup:
        handle_setup(args)
    else:
        raise AssertionError("Unreachable: at least one action must be specified.")


if __name__ == "__main__":
    main()
