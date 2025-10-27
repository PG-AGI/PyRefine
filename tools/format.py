#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple


PYREFINE_ROOT = Path(__file__).resolve().parents[1]
PYREFINE_DIRNAME = PYREFINE_ROOT.name


def determine_project_root() -> Path:
    env_root = os.environ.get("PYREFINE_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
    return PYREFINE_ROOT.parent


PROJECT_ROOT = determine_project_root()

IGNORED_DIR_NAMES: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".idea",
    ".vscode",
    PYREFINE_DIRNAME,
}

BLACK_LINE_LENGTH = 79


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Format and lint Python code. "
            "Run without arguments to do nothing. "
            "Pass 'all' to process the entire project, or an absolute path to a file or directory."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Either 'all' or an absolute path to a Python file or directory.",
    )
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Skip formatting steps and run Flake8 only.",
    )
    return parser.parse_args()


def ensure_absolute_path(path: str) -> Path:
    target_path = Path(path)
    if not target_path.is_absolute():
        raise ValueError(f"{path!r} is not an absolute path.")
    return target_path.resolve()


def partition_paths(paths: Iterable[Path]) -> Tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = []
    for path in paths:
        if path.is_dir():
            directories.append(path)
        else:
            files.append(path)
    return files, directories


def gather_all_targets() -> list[Path]:
    candidates: list[Path] = []

    for child in PROJECT_ROOT.iterdir():
        if child.name in IGNORED_DIR_NAMES:
            continue
        if child.is_dir() or (child.is_file() and child.suffix == ".py"):
            candidates.append(child)

    return sorted(candidates)


def gather_targets(target: Path) -> list[Path]:
    if target.is_dir():
        return [target]
    if target.is_file() and target.suffix == ".py":
        return [target]
    raise ValueError(f"{target} is neither a Python file nor a directory.")


def find_executable(name: str) -> str | None:
    candidate_names = [name]
    if os.name == "nt":
        candidate_names.extend([f"{name}.exe", f"{name}.bat", f"{name}.cmd"])

    search_dirs: list[Path] = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        search_dirs.append(Path(venv) / ("Scripts" if os.name == "nt" else "bin"))
    env_candidates = [
        PROJECT_ROOT / "env",
        PROJECT_ROOT / ".venv",
        PYREFINE_ROOT / "env",
        PYREFINE_ROOT / ".venv",
    ]
    for env_dir in env_candidates:
        if env_dir.exists():
            search_dirs.append(env_dir / ("Scripts" if os.name == "nt" else "bin"))

    for directory in search_dirs:
        if directory.exists():
            for candidate in candidate_names:
                candidate_path = directory / candidate
                if candidate_path.exists():
                    return str(candidate_path)

    return shutil.which(name)


def run_subprocess(command: Sequence[str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def run_autoflake(targets: list[Path]) -> None:
    executable = find_executable("autoflake")
    if executable is None:
        return

    files, directories = partition_paths(targets)
    if directories:
        run_subprocess(
            [
                executable,
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                "--recursive",
                *map(str, directories),
            ]
        )
    if files:
        run_subprocess(
            [
                executable,
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                *map(str, files),
            ]
        )


def run_isort(targets: list[Path]) -> None:
    executable = find_executable("isort")
    if executable is None:
        raise RuntimeError("isort executable not found. Install dependencies first.")

    run_subprocess(
        [
            executable,
            "--profile",
            "black",
            "--line-length",
            str(BLACK_LINE_LENGTH),
            "--atomic",
            *map(str, targets),
        ]
    )


def run_autopep8(targets: list[Path]) -> None:
    executable = find_executable("autopep8")
    if executable is None:
        return

    files, directories = partition_paths(targets)

    if directories:
        run_subprocess(
            [
                executable,
                "--in-place",
                "--aggressive",
                "--aggressive",
                "--max-line-length",
                str(BLACK_LINE_LENGTH),
                "--recursive",
                *map(str, directories),
            ]
        )
    if files:
        run_subprocess(
            [
                executable,
                "--in-place",
                "--aggressive",
                "--aggressive",
                "--max-line-length",
                str(BLACK_LINE_LENGTH),
                *map(str, files),
            ]
        )


def run_black(targets: list[Path]) -> None:
    executable = find_executable("black")
    if executable is None:
        raise RuntimeError("black executable not found. Install dependencies first.")

    run_subprocess(
        [
            executable,
            "--line-length",
            str(BLACK_LINE_LENGTH),
            *map(str, targets),
        ]
    )


def run_flake8(targets: list[Path]) -> None:
    executable = find_executable("flake8")
    if executable is None:
        raise RuntimeError("flake8 executable not found. Install dependencies first.")

    run_subprocess([executable, *map(str, targets)])


def deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def main() -> None:
    args = parse_args()

    if args.target is None:
        print("[format] No action requested. Pass 'all' or an absolute path.")
        return

    try:
        if args.target.lower() == "all":
            targets = deduplicate_paths(gather_all_targets())
        else:
            target_path = ensure_absolute_path(args.target)
            targets = deduplicate_paths(gather_targets(target_path))
    except ValueError as error:
        print(f"[format] {error}", file=sys.stderr)
        sys.exit(1)

    if not targets:
        print("[format] No Python files found to process.")
        return

    try:
        if not args.lint_only:
            run_autoflake(targets)
            run_isort(targets)
            run_autopep8(targets)
            run_black(targets)
        run_flake8(targets)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    except RuntimeError as error:
        print(f"[format] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
