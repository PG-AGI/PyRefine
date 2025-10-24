#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "env"
ENV_BIN_SUBDIR = "Scripts" if os.name == "nt" else "bin"
ENV_BIN_DIR = ENV_DIR / ENV_BIN_SUBDIR


ToolConfig = tuple[str, Sequence[str], bool]

TOOL_CHAIN: Sequence[ToolConfig] = (
    ("black", ("--quiet",), True),
    ("isort", ("--profile", "black", "--quiet"), True),
    ("autoflake", ("--in-place", "--remove-all-unused-imports"), False),
    ("autopep8", ("--in-place", "--aggressive"), False),
    ("unimport", (), False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository's Python formatting and import-cleanup tooling on one or more paths. "
            "Primarily used for VS Code's on-save hook, but can also drive full-repo formatting."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="File or directory to process. Defaults to the main backend package when omitted.",
    )
    parser.add_argument(
        "--workspace",
        action="store_true",
        help=(
            "Run the toolchain against the default project folders "
            "(crack-dev-backend and test_py_scripts)."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the tooling in check mode where supported (no files are modified).",
    )
    return parser.parse_args()


def iter_python_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if p.is_file()))
    unique_files = sorted(dict.fromkeys(files))
    return unique_files


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

    resolved = shutil.which(name)
    if resolved:
        return resolved

    for candidate in candidate_names:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def build_command(
    executable: str,
    base_args: Sequence[str],
    target: Path,
    check_mode: bool,
) -> List[str]:
    cmd = [executable, *base_args]
    if check_mode:
        if os.path.basename(executable).startswith("black"):
            cmd.append("--check")
        elif os.path.basename(executable).startswith("isort"):
            cmd.append("--check-only")
    cmd.append(str(target))
    return cmd


def run_toolchain(files: Sequence[Path], check_mode: bool) -> None:
    if not files:
        return

    for tool, args, supports_check in TOOL_CHAIN:
        if check_mode and not supports_check:
            continue
        executable = find_executable(tool)
        if not executable:
            print(f"[toolchain] Skipping '{tool}': executable not found in PATH or env.", file=sys.stderr)
            continue

        for file_path in files:
            command = build_command(executable, args, file_path, check_mode)
            try:
                subprocess.run(command, check=True, cwd=ROOT)
            except subprocess.CalledProcessError as error:
                try:
                    display_path = file_path.relative_to(ROOT)
                except ValueError:
                    display_path = file_path
                print(
                    f"[toolchain] '{tool}' failed for {display_path} with exit code {error.returncode}.",
                    file=sys.stderr,
                )
                raise


def main() -> None:
    args = parse_args()

    targets: list[Path] = []
    if args.workspace:
        targets.extend(
            path
            for path in (
                ROOT / "crack-dev-backend",
                ROOT / "test_py_scripts",
            )
            if path.exists()
        )

    if args.target:
        target_path = Path(args.target)
        if not target_path.is_absolute():
            target_path = (Path.cwd() / target_path).resolve()
        targets.append(target_path)

    if not targets:
        default_target = ROOT / "crack-dev-backend"
        if default_target.exists():
            targets.append(default_target)
        else:
            print("[toolchain] Nothing to format: could not resolve a target path.", file=sys.stderr)
            sys.exit(0)

    files = iter_python_files(targets)
    if not files:
        print("[toolchain] No Python files found to process.", file=sys.stderr)
        return

    try:
        run_toolchain(files, args.check)
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
