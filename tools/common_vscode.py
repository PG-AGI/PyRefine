#!/usr/bin/env python3
from __future__ import annotations

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

    candidates: list[Path] = []
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
        "Installing it is highly recommended for richer IntelliSense, smarter autocompletion, and an improved Python development experience.\n\n"
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


def formatter_reference(project_root: Path, format_script: Path) -> str:
    try:
        relative = format_script.relative_to(project_root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return format_script.as_posix()


def build_settings_payload(project_root: Path, format_script: Path) -> dict[str, object]:
    command = f'python "{formatter_reference(project_root, format_script)}" "${{file}}"'
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
            "ms-python.vscode-pylance",
            "emeraldwalk.runonsave",
        ],
        "unwantedRecommendations": [
            "ms-python.pylint",
            "charliermarsh.ruff",
        ],
    }


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


def merge_run_on_save(
    settings: dict[str, object], project_root: Path, format_script: Path
) -> dict[str, object]:
    commands_obj = settings.get("emeraldwalk.runonsave")
    if not isinstance(commands_obj, dict):
        return settings
    commands = commands_obj.get("commands")
    if not isinstance(commands, list):
        return settings
    desired_cmd = (
        f'python "{formatter_reference(project_root, format_script)}" "${{file}}"'
    )
    if any(
        isinstance(entry, dict) and entry.get("cmd") == desired_cmd for entry in commands
    ):
        return settings
    commands.append({"match": "\\.py$", "cmd": desired_cmd, "runIn": "terminal"})
    return settings

