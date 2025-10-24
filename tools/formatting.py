#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "env"
ENV_BIN_SUBDIR = "Scripts" if os.name == "nt" else "bin"
ENV_BIN_DIR = ENV_DIR / ENV_BIN_SUBDIR

IGNORED_DIR_NAMES: set[str] = {
    ".git",
    ".vscode",
    ".idea",
    "__pycache__",
    "env",
    ".env",
    ".venv",
    "venv",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
}

DEFAULT_DIRECTORIES: tuple[Path, ...] = tuple(
    (ROOT / name) for name in ("src", "tests", "configs", "scripts")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format and lint Python code using Black, Isort, and Flake8."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional file or directory paths to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every recognised project directory (src, tests, configs, scripts) that exists.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run in check mode (no files modified).",
    )
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Skip formatting and run Flake8 linting only.",
    )
    parser.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip the Flake8 linting step.",
    )
    return parser.parse_args()


def resolve_paths(user_paths: Iterable[str], include_defaults: bool) -> list[Path]:
    resolved: list[Path] = []

    if include_defaults:
        for default_path in DEFAULT_DIRECTORIES:
            if default_path.exists():
                resolved.append(default_path)

    for raw_path in user_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            print(f"[formatting] Skipping {raw_path!r} (path not found).", file=sys.stderr)
            continue
        resolved.append(path)

    if not resolved:
        resolved.append(ROOT)

    unique_paths = []
    seen = set()
    for path in resolved:
        if path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


def expand_targets(paths: Iterable[Path]) -> list[Path]:
    targets: list[Path] = []

    for path in paths:
        path = path.resolve()
        if path.is_file():
            if path.suffix == ".py":
                targets.append(path)
            continue

        if path.name in IGNORED_DIR_NAMES and path != ROOT:
            continue

        if path == ROOT:
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    if child.name in IGNORED_DIR_NAMES:
                        continue
                    targets.append(child)
                elif child.suffix == ".py":
                    targets.append(child)
            continue

        targets.append(path)

    unique_targets: list[Path] = []
    seen = set()
    for target in targets:
        if target in seen:
            continue
        unique_targets.append(target)
        seen.add(target)
    return unique_targets


def find_executable(name: str) -> str | None:
    candidate_names = [name]
    if os.name == "nt":
        candidate_names.extend([f"{name}.exe", f"{name}.bat", f"{name}.cmd"])

    search_dirs: list[Path] = []
    if ENV_BIN_DIR.exists():
        search_dirs.append(ENV_BIN_DIR)

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_dir = Path(venv) / ENV_BIN_SUBDIR
        if venv_dir.exists():
            search_dirs.append(venv_dir)

    for directory in search_dirs:
        for candidate in candidate_names:
            candidate_path = directory / candidate
            if candidate_path.exists():
                return str(candidate_path)

    return shutil.which(name)


def run_tool(executable: str | None, args: Sequence[str], targets: Sequence[Path]) -> None:
    if executable is None:
        raise RuntimeError("Required executable not found.")

    if not targets:
        return

    command = [executable, *args, *(str(target) for target in targets)]
    subprocess.run(command, cwd=ROOT, check=True)


def run_formatters(targets: Sequence[Path], check_mode: bool) -> None:
    black = find_executable("black")
    isort = find_executable("isort")

    black_args = ["--line-length", "100"]
    isort_args = ["--profile", "black"]

    if check_mode:
        black_args.append("--check")
        isort_args.append("--check-only")

    run_tool(black, black_args, targets)
    run_tool(isort, isort_args, targets)


def run_flake8(targets: Sequence[Path]) -> None:
    executable = find_executable("flake8")
    if executable is None:
        raise RuntimeError(
            "flake8 executable not found. Install it or activate the project environment."
        )

    exclude = ",".join(sorted(IGNORED_DIR_NAMES))
    command = [executable, f"--exclude={exclude}"]

    command.extend(str(target) for target in targets)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    resolved_paths = resolve_paths(args.paths, args.all)
    targets = expand_targets(resolved_paths)

    try:
        if not args.lint_only:
            run_formatters(targets, check_mode=args.check)
        if not args.no_lint:
            run_flake8(targets)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    except RuntimeError as error:
        print(f"[formatting] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
