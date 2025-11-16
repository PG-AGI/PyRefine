#!/usr/bin/env python3
from __future__ import annotations

"""
Module Purpose:
    Discovers project directories and runs pytest with coverage reporting while
    saving artifacts into a consistent `pyrefine_artifacts/<project>/coverage` tree.

Key Components:
    - _python_executable: Chooses the correct interpreter when PyRefine runs as an exe.
    - run_pytest_with_coverage: Executes coverage commands and writes XML/HTML outputs.
    - run_for_projects: Iterates over targets, surfacing errors and success paths.

Project Contribution:
    Powers the `--test-coverage` command so teams can enforce testing standards,
    archive reports for CI, and keep coverage automation cross-platform.

"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from shared.gitignore_utils import is_gitignored

ARTIFACTS_DIRNAME = "pyrefine_artifacts"
COVERAGE_SUBDIR = "coverage"
IGNORED_PROJECT_NAMES = {
    ARTIFACTS_DIRNAME,
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "env",
    ".venv",
    "venv",
    "PyRefine",
}
PROJECT_MARKERS = (
    "tests",
    "src",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
)


class CoverageError(RuntimeError):
    """Raised when coverage execution cannot proceed."""


def _python_executable() -> str:
    """
    Coverage needs a real Python interpreter; the packaged exe cannot run modules.
    """
    if getattr(sys, "frozen", False):
        for candidate in ("python.exe", "python", "python3"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise CoverageError(
            "Cannot locate a system Python interpreter for coverage runs."
        )
    return sys.executable


def looks_like_project(path: Path) -> bool:
    for marker in PROJECT_MARKERS:
        candidate = path / marker
        if candidate.is_dir() or candidate.is_file():
            return True
    return False


def discover_projects(root: Path, gitignore_spec) -> List[Path]:
    projects: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in IGNORED_PROJECT_NAMES or child.name.startswith("."):
            continue
        if child.name.lower().startswith("pyrefine"):
            continue
        if is_gitignored(child, root, gitignore_spec):
            continue
        if looks_like_project(child):
            projects.append(child)
    return projects


def project_artifact_dir(project_dir: Path) -> Path:
    base = project_dir.parent / ARTIFACTS_DIRNAME / project_dir.name / COVERAGE_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_pytest_with_coverage(project_dir: Path) -> Path:
    coverage_dir = project_artifact_dir(project_dir)
    shutil.rmtree(coverage_dir, ignore_errors=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)

    python_exec = _python_executable()
    env = os.environ.copy()
    run_cmd = [python_exec, "-m", "coverage", "run", "-m", "pytest"]
    subprocess.run(run_cmd, cwd=project_dir, check=True, env=env)

    xml_path = coverage_dir / "coverage.xml"
    subprocess.run(
        [python_exec, "-m", "coverage", "xml", "-o", str(xml_path)],
        cwd=project_dir,
        check=True,
    )

    html_dir = coverage_dir / "coverage_html_report"
    subprocess.run(
        [python_exec, "-m", "coverage", "html", "-d", str(html_dir)],
        cwd=project_dir,
        check=True,
    )

    report = subprocess.run(
        [python_exec, "-m", "coverage", "report"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    (coverage_dir / "summary.txt").write_text(report.stdout, encoding="utf-8")

    coverage_file = project_dir / ".coverage"
    if coverage_file.exists():
        shutil.copy2(coverage_file, coverage_dir / ".coverage")

    return coverage_dir


def run_for_projects(projects: Iterable[Path]) -> None:
    ran_any = False
    for project_dir in projects:
        ran_any = True
        print(f"[coverage] Running tests for {project_dir}")
        run_pytest_with_coverage(project_dir)
        print(
            "[coverage] Reports saved to "
            f"{project_artifact_dir(project_dir)}"
        )
    if not ran_any:
        raise CoverageError("No projects were provided for coverage.")

