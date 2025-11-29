#!/usr/bin/env python3
from __future__ import annotations

"""
Module Purpose:
    Discovers project directories and runs pytest with coverage reporting while
     saving artifacts into a consistent `pyrefine_artifacts/<project>/coverage`
     tree.

Key Components:
    - _python_executable: Chooses the correct interpreter when PyRefine runs
     as an exe.
    - run_pytest_with_coverage: Executes coverage commands and writes XML/HTML
     outputs.
    - run_for_projects: Iterates over targets, surfacing errors and success
     paths.

Project Contribution:
    Powers the `--test-coverage` command so teams can enforce testing
    standards, archive reports for CI, and keep coverage automation
    cross-platform.

"""

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from shared.gitignore_utils import is_gitignored

ARTIFACTS_DIRNAME = "pyrefine_artifacts"
COVERAGE_SUBDIR = "coverage"
PUBLISHED_HTML_DIRNAME = "pyrefine-test-coverage"
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
    Coverage needs a real Python interpreter; the packaged exe cannot run
     modules.
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
    base = (
        project_dir.parent
        / ARTIFACTS_DIRNAME
        / project_dir.name
        / COVERAGE_SUBDIR
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _coverage_run_dir(coverage_dir: Path) -> Path:
    """
    Create a unique subdirectory for a single coverage run so previous runs
     remain available. The parent coverage directory itself is never removed.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = coverage_dir / f"run-{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = coverage_dir / f"run-{timestamp}-{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _published_html_root(project_dir: Path) -> Path:
    """
    Ensure the user-facing coverage folder exists at the project root.

    The layout is:
        <project>/pyrefine-test-coverage/
            index.html
            report/  (full HTML report and assets)
    """
    root = project_dir / PUBLISHED_HTML_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _publish_html_report(project_dir: Path, html_dir: Path) -> Path:
    """
    Copy the generated HTML report into the public `pyrefine-test-coverage`
     folder so users have a stable entry point.

    Inside `pyrefine-test-coverage` we keep only:
        - index.html
        - a single `report/` subfolder containing the full coverage HTML tree.
    """
    if not html_dir.exists():
        raise CoverageError(
            f"Expected coverage HTML directory at {html_dir}, but it "
            "was not created."
        )

    published_root = _published_html_root(project_dir)

    # Clear previous contents inside the published root, but never remove
    # the root directory itself.
    for child in published_root.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

    report_dir = published_root / "report"
    shutil.copytree(html_dir, report_dir)

    index_path = published_root / "index.html"
    index_html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <title>PyRefine Test Coverage</title>\n"
        "  <meta http-equiv=\"refresh\" content=\"0; url=report/index.html\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />\n"
        "</head>\n"
        "<body>\n"
        "  <p>If you are not redirected automatically, "
        '<a href="report/index.html">open the coverage report</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )
    index_path.write_text(index_html, encoding="utf-8")

    return published_root


def run_pytest_with_coverage(project_dir: Path) -> Path:
    coverage_dir = project_artifact_dir(project_dir)
    run_dir = _coverage_run_dir(coverage_dir)

    python_exec = _python_executable()
    env = os.environ.copy()
    run_cmd = [python_exec, "-m", "coverage", "run", "-m", "pytest"]
    subprocess.run(run_cmd, cwd=project_dir, check=True, env=env)

    xml_path = run_dir / "coverage.xml"
    subprocess.run(
        [python_exec, "-m", "coverage", "xml", "-o", str(xml_path)],
        cwd=project_dir,
        check=True,
    )

    html_dir = run_dir / "coverage_html_report"
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
    (run_dir / "summary.txt").write_text(report.stdout, encoding="utf-8")

    coverage_file = project_dir / ".coverage"
    if coverage_file.exists():
        shutil.copy2(coverage_file, run_dir / ".coverage")

    _publish_html_report(project_dir, html_dir)

    return run_dir


def run_for_projects(projects: Iterable[Path]) -> None:
    ran_any = False
    for project_dir in projects:
        ran_any = True
        print(f"[coverage] Running tests for {project_dir}")
        run_dir = run_pytest_with_coverage(project_dir)
        print(
            "[coverage] Reports saved to "
            f"{project_artifact_dir(project_dir)} (latest run: {run_dir})"
        )
    if not ran_any:
        raise CoverageError("No projects were provided for coverage.")
