#!/usr/bin/env python3
from __future__ import annotations

"""
Module Purpose:
    Acts as the primary CLI entry point that routes user flags to scaffolding,
     cleaning, setup, coverage, and self-update workflows.

Key Components:
    - parse_args: Validates mutually exclusive flags and default behaviors.
    - handle_create / handle_clean / handle_setup / handle_test_coverage /
     handle_update:
      Dispatchers that invoke the respective managers.
    - main: Bootstraps the CLI by resolving arguments and calling the chosen
     handler.

Project Contribution:
    Provides the cohesive command surface for PyRefine so every automation
     task can be triggered with consistent flags across platforms and packaging
     modes.

"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from commands.clean import clean_manager
from commands.create import scaffold_manager
from commands.setup import setup_manager
from commands.test_coverage import coverage_runner
from commands.update import update_manager
from shared.gitignore_utils import (
    GitignoreError,
    is_gitignored,
    load_gitignore_spec,
)

APP_VERSION = "1.0"
DEFAULT_MANIFEST_URL = os.environ.get(
    "PYREFINE_UPDATE_URL",
    (
        "https://raw.githubusercontent.com/PG-AGI/PyRefine/"
        "pyrefine.exe/release/manifest.json"
    ),
)
TEST_COVERAGE_ALL = "__PYREFINE_TEST_COVERAGE_ALL__"


def _execution_project_root(project_root: Path) -> Path:
    """
    When frozen as pyrefine.exe, treat the executable's directory as the
     project root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root.resolve()


def get_resource_root() -> Path:
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidate = Path(bundle_dir) / "PyRefine"
        if candidate.exists():
            return candidate
        return Path(bundle_dir)
    return Path(__file__).resolve().parents[1]


RESOURCE_ROOT = get_resource_root()
APP_VERSION = "unknown"


def _load_app_version(resource_root: Path) -> str:
    manifest_path = resource_root / "release" / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    version = data.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "unknown"


APP_VERSION = _load_app_version(RESOURCE_ROOT)
FORMAT_SCRIPT = RESOURCE_ROOT / "commands" / "clean" / "format.py"
FLAKE8_TEMPLATE = RESOURCE_ROOT / "configs" / ".flake8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PyRefine CLI for scaffolding, cleanup, environments, and updates."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Project root folder (defaults to the current working "
            "directory)."
        ),
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help=(
            "Create the standard Python project scaffold "
            "(src/, tests/, configs/, scripts/, utils/, services/)."
        ),
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const=".",
        metavar="PATH",
        help=(
            "Format a file, folder, or the entire project (default: current "
            "project)."
        ),
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Configure VS Code and provision pip/UV virtual environments.",
    )
    parser.add_argument(
        "--test-coverage",
        nargs="?",
        const=TEST_COVERAGE_ALL,
        metavar="PROJECT_PATH",
        help=(
            "Run pytest with coverage enabled and store reports under "
            "pyrefine_artifacts/<project>/coverage."
        ),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Check the update manifest and replace the current executable if "
            "a newer version is available."
        ),
    )
    parser.add_argument(
        "--manifest-url",
        help=(
            "Override the update manifest URL. Defaults to the value of the "
            "PYREFINE_UPDATE_URL environment variable (if set)."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the current PyRefine version and exit.",
    )
    args = parser.parse_args()

    actions = sum(
        [
            1 if args.create else 0,
            1 if args.clean is not None else 0,
            1 if args.setup else 0,
            1 if args.test_coverage is not None else 0,
            1 if args.update else 0,
            1 if args.version else 0,
        ]
    )
    if actions > 1:
        parser.error(
            "Please choose only one action at a time "
            "(--create, --clean, --setup, --test-coverage, --update, or --version)."
        )
    if args.version:
        return args
    if args.update:
        return args  # no default action when explicitly updating
    if actions == 0:
        args.clean = "."
    return args


def handle_create(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    scaffold_manager.ensure_scaffold(project_root, RESOURCE_ROOT)
    print("Scaffold complete.")
    print(
        "Run 'python PyRefine/cli/pyrefine.py --setup' to configure VS Code."
    )


def handle_clean(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    clean_manager.run_clean(
        project_root, args.clean, FORMAT_SCRIPT, FLAKE8_TEMPLATE
    )


def handle_setup(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    setup_manager.run_setup(project_root, RESOURCE_ROOT)


def handle_test_coverage(args: argparse.Namespace) -> None:
    root = _execution_project_root(args.project_root)
    try:
        gitignore_spec = load_gitignore_spec(root)
    except GitignoreError as exc:
        print(f"[pyrefine] {exc}", file=sys.stderr)
        sys.exit(1)

    target_arg = args.test_coverage
    if target_arg is None or target_arg == TEST_COVERAGE_ALL:
        project_path = root
    else:
        project_path = clean_manager.ensure_absolute(Path(target_arg), root)

    if not project_path.exists():
        print(
            f"[pyrefine] Target '{project_path}' does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not project_path.is_dir():
        print(
            f"[pyrefine] '{project_path}' is not a directory.", file=sys.stderr
        )
        sys.exit(1)
    if is_gitignored(project_path, root, gitignore_spec):
        message = (
            f"[pyrefine] '{project_path}' is ignored via .gitignore; "
            "skipping."
        )
        print(message, file=sys.stderr)
        sys.exit(1)

    projects = [project_path]

    try:
        coverage_runner.run_for_projects(projects)
    except coverage_runner.CoverageError as exc:
        print(f"[pyrefine] {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(
            "[pyrefine] Coverage command failed "
            f"(exit code {exc.returncode}).",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)


def handle_update(args: argparse.Namespace) -> None:
    update_manager.handle_update(
        args=args,
        current_version=APP_VERSION,
        default_manifest_url=DEFAULT_MANIFEST_URL,
    )


def main() -> None:
    args = parse_args()
    if args.version:
        print(f"PyRefine version {APP_VERSION}")
        return
    if args.create:
        handle_create(args)
    elif args.clean is not None:
        handle_clean(args)
    elif args.setup:
        handle_setup(args)
    elif args.test_coverage is not None:
        handle_test_coverage(args)
    elif args.update:
        handle_update(args)
    else:
        raise AssertionError(
            "Unreachable: at least one action must be specified."
        )


if __name__ == "__main__":
    main()
