#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
FORMATTER_SCRIPT = ROOT / "tools" / "formatting.py"

RECOMMENDED_EXTENSIONS: Sequence[str] = (
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
)

UNWANTED_EXTENSIONS: Sequence[str] = (
    "ms-python.pylint",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
)

DEFAULT_DIRECTORIES: Sequence[Path] = (
    ROOT / "src",
    ROOT / "tests",
    ROOT / "configs",
    ROOT / "scripts",
)

DEFAULT_FILES: Sequence[tuple[Path, str]] = (
    (ROOT / "src" / "__init__.py", ""),
    (ROOT / "tests" / "__init__.py", ""),
    (ROOT / "configs" / ".gitkeep", ""),
    (ROOT / "scripts" / ".gitkeep", ""),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the Python starter template after cloning."
    )
    parser.add_argument(
        "--mode",
        choices=("new", "existing"),
        help="Skip prompts and pick the project mode up front.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume the default answer for all prompts.",
    )
    parser.add_argument(
        "--skip-extensions",
        action="store_true",
        help="Do not attempt to manage VS Code extensions automatically.",
    )
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Do not trigger any formatting during bootstrap.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation (pip install).",
    )
    return parser.parse_args()


def ask_choice(question: str, choices: Sequence[str], assume_default: bool) -> str:
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


def ask_yes_no(question: str, assume_default: bool, default: bool = True) -> bool:
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


def ensure_project_structure() -> None:
    for directory in DEFAULT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    for file_path, contents in DEFAULT_FILES:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(contents, encoding="utf-8")


def run_process(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def manage_extensions(skip_extensions: bool) -> None:
    if skip_extensions:
        return

    code_cli = shutil.which("code")
    if code_cli is None:
        # Windows ships code.cmd; shutil.which will find it under "code"
        code_cli = shutil.which("code.cmd")

    if code_cli is None:
        print(
            "[bootstrap] VS Code CLI ('code') not found. Skipping extension management.",
            file=sys.stderr,
        )
        return

    for extension in RECOMMENDED_EXTENSIONS:
        try:
            run_process([code_cli, "--install-extension", extension])
        except subprocess.CalledProcessError:
            print(f"[bootstrap] Failed to install extension {extension}.", file=sys.stderr)

    for extension in UNWANTED_EXTENSIONS:
        try:
            run_process([code_cli, "--uninstall-extension", extension])
        except subprocess.CalledProcessError:
            # Ext might already be absent; ignore.
            pass


def install_dependencies() -> None:
    requirements = ROOT / "requirements.txt"
    if not requirements.exists():
        return

    try:
        run_process([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    except subprocess.CalledProcessError:
        print("[bootstrap] Failed to install dependencies via pip.", file=sys.stderr)
        raise


def format_entire_workspace() -> None:
    run_process([sys.executable, str(FORMATTER_SCRIPT), "--all"])


def format_specific_path(target: str) -> None:
    run_process([sys.executable, str(FORMATTER_SCRIPT), target])


def interactive_formatting(skip_format: bool, assume_default: bool) -> None:
    if skip_format:
        return

    if ask_yes_no("Format the entire workspace now?", assume_default, default=True):
        try:
            format_entire_workspace()
        except subprocess.CalledProcessError as error:
            print(f"[bootstrap] Formatting failed (exit code {error.returncode}).", file=sys.stderr)

    while True:
        if not ask_yes_no("Format an additional file or folder?", assume_default, default=False):
            break
        target = input("Enter the relative path to format: ").strip()
        if not target:
            continue
        try:
            format_specific_path(target)
        except subprocess.CalledProcessError as error:
            print(
                f"[bootstrap] Formatting failed for {target!r} (exit code {error.returncode}).",
                file=sys.stderr,
            )


def main() -> None:
    args = parse_args()

    print("== Python Starter Template Bootstrap ==")

    mode = args.mode or ask_choice(
        "Are you configuring an existing codebase or starting a new project?",
        ("existing", "new"),
        args.yes,
    )
    manage_extensions(args.skip_extensions)

    if mode == "new":
        if ask_yes_no(
            "Generate the default project structure (src/, tests/, configs/, scripts/)?",
            args.yes,
            default=True,
        ):
            ensure_project_structure()
            print("[bootstrap] Project structure ensured.")
    else:
        print("[bootstrap] Existing project mode selected.")

    if not args.skip_deps:
        if ask_yes_no("Install Python dependencies from requirements.txt?", args.yes, default=True):
            try:
                install_dependencies()
            except subprocess.CalledProcessError:
                pass

    interactive_formatting(args.skip_format, args.yes)

    print("Bootstrap complete. Happy coding!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bootstrap] Aborted by user.", file=sys.stderr)
        sys.exit(1)
