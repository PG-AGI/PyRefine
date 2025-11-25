#!/usr/bin/env python3
from __future__ import annotations

"""
Module Purpose:
    Implements the formatter pipeline backing `--clean` and Run On Save
     workflows,
    handling target discovery, gitignore filtering, and tool invocation.

Key Components:
    - gather_all_targets / gather_targets: Collect Python paths honoring
     .gitignore.
    - run_autoflake / run_isort / run_black / run_flake8: Sequentially apply
     each tool.
    - main: Parses CLI args to run the formatter in “all” or single-target
     modes.

Project Contribution:
    Ensures every PyRefine-managed repository adheres to consistent style and
     linting rules, enabling automated hygiene across projects and CI
     environments.

"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from shared.gitignore_utils import (
    GitignoreError,
    is_gitignored,
    load_gitignore_spec,
)

PYREFINE_ROOT = Path(__file__).resolve().parents[1]
PYREFINE_DIRNAME = PYREFINE_ROOT.name
FORMAT_BATCH_SIZE = 200
GITIGNORE_SPEC = None


def _load_dynamic_ignore_names() -> set[str]:
    raw = os.environ.get("PYREFINE_IGNORE_DIRS", "")
    if not raw:
        return set()
    return {entry.strip() for entry in raw.split(",") if entry.strip()}


DYNAMIC_IGNORE_NAMES: set[str] = _load_dynamic_ignore_names()


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


def ensure_gitignore_spec():
    global GITIGNORE_SPEC
    if GITIGNORE_SPEC is None:
        try:
            GITIGNORE_SPEC = load_gitignore_spec(PROJECT_ROOT)
        except GitignoreError as exc:
            print(f"[format] {exc}", file=sys.stderr)
            sys.exit(1)
    return GITIGNORE_SPEC


def gitignored(path: Path) -> bool:
    return is_gitignored(path, PROJECT_ROOT, ensure_gitignore_spec())


def should_skip_dir(path: Path) -> bool:
    if path.resolve() == PROJECT_ROOT.resolve():
        return False
    if path.name in IGNORED_DIR_NAMES or path.name in DYNAMIC_IGNORE_NAMES:
        return True
    return gitignored(path)


def should_include_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    return not gitignored(path)


def iter_python_files(sources: Iterable[Path]) -> Iterator[Path]:
    for source in sources:
        if source.is_file():
            if should_include_file(source):
                yield source.resolve()
            continue
        if not source.is_dir():
            continue
        for root, dirs, files in os.walk(source):
            root_path = Path(root)
            if should_skip_dir(root_path):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not should_skip_dir(root_path / d)]
            for filename in files:
                candidate = root_path / filename
                if should_include_file(candidate):
                    yield candidate.resolve()


def collect_targets(paths: Iterable[Path]) -> list[Path]:
    files = deduplicate_paths(iter_python_files(paths))
    return sorted(files)


def chunked_targets(targets: list[Path], chunk_size: int = FORMAT_BATCH_SIZE):
    for index in range(0, len(targets), chunk_size):
        yield targets[index : index + chunk_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Format and lint Python code. "
            "Run without arguments to do nothing. "
            "Pass 'all' to process the entire project, or an absolute path "
            "to a file or directory."
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


def gather_all_targets() -> list[Path]:
    return collect_targets([PROJECT_ROOT])


def gather_targets(target: Path) -> list[Path]:
    if target.is_dir():
        if should_skip_dir(target):
            return []
        return collect_targets([target])
    if target.is_file():
        if target.suffix != ".py":
            raise ValueError(
                f"{target} is neither a Python file nor a directory."
            )
        if gitignored(target):
            return []
        return [target.resolve()]
    raise ValueError(f"{target} is neither a Python file nor a directory.")


def find_executable(name: str) -> str | None:
    candidate_names = [name]
    if os.name == "nt":
        candidate_names.extend([f"{name}.exe", f"{name}.bat", f"{name}.cmd"])

    search_dirs: list[Path] = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        search_dirs.append(
            Path(venv) / ("Scripts" if os.name == "nt" else "bin")
        )
    env_candidates = [
        PROJECT_ROOT / "env",
        PROJECT_ROOT / ".venv",
        PYREFINE_ROOT / "env",
        PYREFINE_ROOT / ".venv",
    ]
    for env_dir in env_candidates:
        if env_dir.exists():
            search_dirs.append(
                env_dir / ("Scripts" if os.name == "nt" else "bin")
            )

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

    for batch in chunked_targets(targets):
        run_subprocess(
            [
                executable,
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                *map(str, batch),
            ]
        )


def run_isort(targets: list[Path]) -> None:
    executable = find_executable("isort")
    if executable is None:
        raise RuntimeError(
            "isort executable not found. Install dependencies first."
        )

    for batch in chunked_targets(targets):
        run_subprocess(
            [
                executable,
                "--profile",
                "black",
                "--line-length",
                str(BLACK_LINE_LENGTH),
                "--atomic",
                *map(str, batch),
            ]
        )


def run_autopep8(targets: list[Path]) -> None:
    executable = find_executable("autopep8")
    if executable is None:
        return

    for batch in chunked_targets(targets):
        run_subprocess(
            [
                executable,
                "--in-place",
                "--aggressive",
                "--aggressive",
                "--max-line-length",
                str(BLACK_LINE_LENGTH),
                *map(str, batch),
            ]
        )


def run_black(targets: list[Path]) -> None:
    executable = find_executable("black")
    if executable is None:
        raise RuntimeError(
            "black executable not found. Install dependencies first."
        )

    for batch in chunked_targets(targets):
        run_subprocess(
            [
                executable,
                "--line-length",
                str(BLACK_LINE_LENGTH),
                *map(str, batch),
            ]
        )


def run_flake8(targets: list[Path]) -> None:
    executable = find_executable("flake8")
    if executable is None:
        raise RuntimeError(
            "flake8 executable not found. Install dependencies first."
        )

    for batch in chunked_targets(targets):
        run_subprocess([executable, *map(str, batch)])


def run_string_fixer(targets: list[Path]) -> None:
    try:
        from string_fixer import FixStats, fix_file
    except ImportError as exc:
        print(
            f"[string-fixer] Unable to import string_fixer: {exc}",
            file=sys.stderr,
        )
        return

    stats = FixStats()
    for file_path in targets:
        fix_file(file_path, stats=stats, create_backup=False)

    if stats.strings_fixed > 0:
        print(
            f"[string-fixer] Fixed {stats.strings_fixed} long strings in {stats.files_modified} file(s)"
        )
    else:
        print("[string-fixer] No long strings found to fix.")
    if stats.errors:
        print(
            f"[string-fixer] Encountered {len(stats.errors)} issue(s):"
        )
        for error in stats.errors[:3]:
            print(f"    - {error}")
        if len(stats.errors) > 3:
            print("    - ...")


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

    # Step 0: Run string fixer before all other formatters
    try:
        run_string_fixer(targets)
    except Exception as error:
        print(f"[string-fixer] Error: {error}", file=sys.stderr)

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
