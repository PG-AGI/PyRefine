#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_resource_root() -> Path:
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidate = Path(bundle_dir) / "PyRefine"
        if candidate.exists():
            return candidate
        return Path(bundle_dir)
    return Path(__file__).resolve().parents[1]


PYLANCE_EXTENSION_ID = "ms-python.vscode-pylance"


def pylance_installed() -> bool:
    for cmd in ("code", "code-insiders"):
        exe = shutil.which(cmd)
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, "--list-extensions"],
                capture_output=True,
                text=True,
                check=True,
            )
            if PYLANCE_EXTENSION_ID.lower() in result.stdout.lower():
                return True
            return False
        except subprocess.CalledProcessError:
            continue

    candidates = []
    home = Path.home()
    candidates.append(home / ".vscode" / "extensions")
    candidates.append(home / ".vscode-insiders" / "extensions")
    user_profile = Path(os.environ.get("USERPROFILE", home))
    candidates.append(user_profile / ".vscode" / "extensions")
    for base in candidates:
        if base.exists():
            for child in base.iterdir():
                if child.is_dir() and child.name.startswith(PYLANCE_EXTENSION_ID):
                    return True
    return False


def notify_pylance_missing() -> None:
    message = (
        "The Pylance extension (ms-python.vscode-pylance) was not detected.\n"
        "Installing it is highly recommended for richer IntelliSense, "
        "smarter autocompletion, and an improved Python development "
        "experience.\n\n"
        "Marketplace link:\n"
        "https://marketplace.visualstudio.com/items?itemName="
        "ms-python.vscode-pylance"
    )
    print(f"NOTICE: {message}")
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                None,
                message,
                "PyRefine Recommendation",
                0x00000040,
            )
        except Exception:
            pass
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Write VS Code workspace settings for PyRefine.")
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help=(
            "Project root directory (defaults to the current working "
            "directory)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    return parser.parse_args()


def formatter_reference(project_root: Path, format_script: Path) -> str:
    try:
        relative = format_script.relative_to(project_root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return format_script.as_posix()


def build_settings(
    project_root: Path, format_script: Path
) -> dict[str, object]:
    formatter_cmd = formatter_reference(project_root, format_script)
    command = f'python "{formatter_cmd}" "${{file}}"'
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
    project_root = (args.project_root or Path.cwd()).resolve()
    resource_root = get_resource_root()
    format_script = resource_root / "tools" / "format.py"

    settings_dir = project_root / ".vscode"
    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    if confirm(settings_path, args.force, "settings.json"):
        write_json(settings_path, build_settings(project_root, format_script))

    extensions_path = settings_dir / "extensions.json"
    if confirm(extensions_path, args.force, "extensions.json"):
        write_json(extensions_path, build_extensions())

    if not pylance_installed():
        notify_pylance_missing()
    print(f"Workspace settings written to {settings_dir}")


if __name__ == "__main__":
    main()
