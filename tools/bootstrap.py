#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PYREFINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = PYREFINE_ROOT.parent

FORMAT_SCRIPT = PYREFINE_ROOT / "tools" / "format.py"

RECOMMENDED_EXTENSIONS: Sequence[str] = (
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "emeraldwalk.runonsave",
)

UNWANTED_EXTENSIONS: Sequence[str] = (
    "ms-python.pylint",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
)

STRUCTURE_DIRECTORIES: Sequence[str] = ("src", "tests", "configs", "scripts")
STRUCTURE_FILES: Sequence[tuple[str, str]] = (
    ("src/__init__.py", ""),
    ("tests/__init__.py", ""),
    ("configs/.gitkeep", ""),
    ("scripts/.gitkeep", ""),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap or clean a Python project using the PyRefine template."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("clean", "create"),
        help="Skip prompts by specifying the desired action up front.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically accept the default answer for prompts.",
    )
    parser.add_argument(
        "--skip-extensions",
        action="store_true",
        help="Do not attempt to install or remove VS Code extensions.",
    )
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Skip automated formatting steps.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation (pip install).",
    )
    return parser.parse_args()


def ask_choice(
    question: str,
    choices: Sequence[str],
    assume_default: bool,
) -> str:
    if not choices:
        raise ValueError("choices must not be empty")

    if assume_default:
        return choices[0]

    prompt = f"{question} ({'/'.join(choices)}): "
    while True:
        response = input(prompt).strip().lower()
        if not response:
            response = choices[0]
        if response in choices:
            return response
        print(f"Please enter one of {', '.join(choices)}.")


def ask_yes_no(
    question: str,
    assume_default: bool,
    default: bool = True,
) -> bool:
    if assume_default:
        return default

    suffix = "Y/n" if default else "y/N"
    prompt = f"{question} ({suffix}): "
    while True:
        response = input(prompt).strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_for_project_root(default_root: Path, assume_default: bool) -> Path:
    message = "Detected project root (parent directory of PyRefine): "
    print(f"{message}{default_root}")
    if ask_yes_no("Use this directory?", assume_default, default=True):
        return default_root

    while True:
        prompt = "Enter the absolute path to the project root: "
        user_input = input(prompt).strip()
        if not user_input:
            print("Please provide a valid absolute path.")
            continue
        candidate = Path(user_input).expanduser()
        if not candidate.is_absolute():
            print("The provided path is not absolute. Please try again.")
            continue
        if not candidate.exists():
            if ask_yes_no(
                f"{candidate} does not exist. Create it?",
                assume_default,
                default=True,
            ):
                candidate.mkdir(parents=True, exist_ok=True)
            else:
                continue
        return candidate.resolve()


def rel_script_reference(project_root: Path) -> str:
    try:
        relative = FORMAT_SCRIPT.relative_to(project_root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return FORMAT_SCRIPT.as_posix()


def build_settings_payload(project_root: Path) -> dict[str, object]:
    formatter_reference = rel_script_reference(project_root)
    command = f'python "{formatter_reference}" "${{file}}"'
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
        "python.linting.ignorePatterns": ["env/**", "**/__pycache__/**"],
        "emeraldwalk.runonsave": {
            "commands": [
                {
                    "match": "\\.py$",
                    "cmd": command,
                    "runIn": "terminal",
                }
            ]
        },
    }


def build_extensions_payload() -> dict[str, object]:
    return {
        "recommendations": list(RECOMMENDED_EXTENSIONS),
        "unwantedRecommendations": list(UNWANTED_EXTENSIONS),
    }


def run_process(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def manage_extensions(skip_extensions: bool) -> None:
    if skip_extensions:
        return

    code_cli = shutil.which("code") or shutil.which("code.cmd")
    if code_cli is None:
        print(
            "[bootstrap] VS Code CLI ('code') not found. "
            "Skipping extension management.",
        )
        return

    for extension in RECOMMENDED_EXTENSIONS:
        try:
            run_process([code_cli, "--install-extension", extension])
        except subprocess.CalledProcessError:
            print(
                f"[bootstrap] Failed to install extension {extension}.",
                file=sys.stderr,
            )

    for extension in UNWANTED_EXTENSIONS:
        try:
            run_process([code_cli, "--uninstall-extension", extension])
        except subprocess.CalledProcessError:
            pass


def ensure_project_structure(project_root: Path) -> None:
    for directory in STRUCTURE_DIRECTORIES:
        (project_root / directory).mkdir(parents=True, exist_ok=True)

    for relative_path, contents in STRUCTURE_FILES:
        file_path = project_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(contents, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def ensure_workspace_settings(
    project_root: Path,
    assume_default: bool,
) -> None:
    settings_dir = project_root / ".vscode"
    if settings_dir.exists():
        message = f"Update existing VS Code workspace files in {settings_dir}?"
    else:
        message = f"Create VS Code workspace settings in {settings_dir}?"

    if not ask_yes_no(message, assume_default, default=True):
        return

    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    extensions_path = settings_dir / "extensions.json"

    if settings_path.exists():
        overwrite_settings = ask_yes_no(
            f"{settings_path} already exists. "
            "Overwrite with PyRefine defaults?",
            assume_default,
            default=False,
        )
    else:
        overwrite_settings = True

    if overwrite_settings:
        write_json(settings_path, build_settings_payload(project_root))

    if extensions_path.exists():
        overwrite_extensions = ask_yes_no(
            f"{extensions_path} already exists. "
            "Overwrite with PyRefine defaults?",
            assume_default,
            default=False,
        )
    else:
        overwrite_extensions = True

    if overwrite_extensions:
        write_json(extensions_path, build_extensions_payload())


def install_dependencies() -> None:
    requirements = PYREFINE_ROOT / "requirements.txt"
    if not requirements.exists():
        return
    run_process(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
        ]
    )


def format_entire_project(project_root: Path) -> None:
    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)
    run_process([sys.executable, str(FORMAT_SCRIPT), "all"], env=env)


def format_specific_path(target: Path, project_root: Path) -> None:
    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)
    run_process([sys.executable, str(FORMAT_SCRIPT), str(target)], env=env)


def interactive_formatting(
    skip_format: bool,
    assume_default: bool,
    project_root: Path,
) -> None:
    if skip_format:
        return

    while True:
        if not ask_yes_no(
            "Format another file or folder?",
            assume_default,
            default=False,
        ):
            break
        target_input = input("Enter the absolute path to format: ").strip()
        if not target_input:
            continue
        candidate = Path(target_input).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        try:
            format_specific_path(candidate, project_root)
        except subprocess.CalledProcessError as error:
            print(
                "[bootstrap] Formatting failed for "
                f"{candidate} (exit code {error.returncode}).",
                file=sys.stderr,
            )


def main() -> None:
    args = parse_args()

    print("== PyRefine Bootstrap ==")

    question = "Do you want to clean an existing repository "
    question += "or create a new project structure?"
    action = args.mode or ask_choice(
        question,
        ("clean", "create"),
        args.yes,
    )
    project_root = prompt_for_project_root(DEFAULT_PROJECT_ROOT, args.yes)
    print(f"[bootstrap] Working against project root: {project_root}")

    ensure_workspace_settings(project_root, args.yes)
    manage_extensions(args.skip_extensions)

    if not args.skip_deps:
        if ask_yes_no(
            "Install Python dependencies from requirements.txt?",
            args.yes,
            default=True,
        ):
            try:
                install_dependencies()
            except subprocess.CalledProcessError as error:
                print(
                    "[bootstrap] Dependency installation failed "
                    f"(exit code {error.returncode}).",
                    file=sys.stderr,
                )

    if action == "create":
        if ask_yes_no(
            "Generate the default project structure (src/, tests/, configs/,"
            " scripts/)?",
            args.yes,
            default=True,
        ):
            ensure_project_structure(project_root)
            print("[bootstrap] Project structure created.")
        if not args.skip_format:
            if ask_yes_no(
                "Format the new project now?",
                args.yes,
                default=True,
            ):
                try:
                    format_entire_project(project_root)
                except subprocess.CalledProcessError as error:
                    message = "[bootstrap] Formatting failed "
                    message += f"(exit code {error.returncode})."
                    print(message, file=sys.stderr)
    else:
        print("[bootstrap] Cleaning existing repository.")
        if not args.skip_format:
            try:
                format_entire_project(project_root)
            except subprocess.CalledProcessError as error:
                print(
                    "[bootstrap] Formatting failed "
                    f"(exit code {error.returncode}).",
                    file=sys.stderr,
                )

    interactive_formatting(args.skip_format, args.yes, project_root)

    print("Bootstrap complete. Happy coding!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bootstrap] Aborted by user.", file=sys.stderr)
        sys.exit(1)
