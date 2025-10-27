#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PYREFINE_ROOT = Path(__file__).resolve().parents[1]
FORMAT_SCRIPT = PYREFINE_ROOT / "tools" / "format.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write VS Code workspace settings for PyRefine."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (defaults to PyRefine parent).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    return parser.parse_args()


def formatter_reference(project_root: Path) -> str:
    try:
        relative = FORMAT_SCRIPT.relative_to(project_root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return FORMAT_SCRIPT.as_posix()


def build_settings(project_root: Path) -> dict[str, object]:
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


def build_extensions() -> dict[str, object]:
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


def confirm(path: Path, force: bool, label: str) -> bool:
    if force or not path.exists():
        return True
    prompt = f"{label} already exists. Overwrite? [y/N]: "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root or PYREFINE_ROOT.parent
    project_root = project_root.resolve()

    settings_dir = project_root / ".vscode"
    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    if confirm(settings_path, args.force, "settings.json"):
        write_json(settings_path, build_settings(project_root))

    extensions_path = settings_dir / "extensions.json"
    if confirm(extensions_path, args.force, "extensions.json"):
        write_json(extensions_path, build_extensions())

    print(f"Workspace settings written to {settings_dir}")


if __name__ == "__main__":
    main()
