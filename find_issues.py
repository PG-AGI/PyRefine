#!/usr/bin/env python3
"""
lint_suite.py — Recursively lint Python files and produce JSON reports.

What we will use:
- Ruff   : fast, modern linter (style & many rules)
- Pylint : deep static analysis, code smells
- Mypy   : type checker (optional but recommended if you use hints)
- Bandit : security lints (optional)

Outputs:
- lint_report.json (master combined report + metadata)
- lint_reports/<per-file>.json (one file per source)

How to run:
    python lint_suite.py

Notes:
- Tools are auto-detected in PATH; missing tools are skipped gracefully.
- Non-zero exit codes from linters are expected; we always parse their outputs.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ========= EDIT THIS LINE =========
ROOT_DIR = r"H:\PGAGI\PyRefine\crack-dev-backend"  # <-- change to your target directory if needed
# ==================================

# Files & dirs
OUTPUT_DIR = Path("crackDev_lint_reports_after")
MASTER_REPORT = Path("lint_report_a2.json")

# File extensions we’ll treat as “Python”
PYTHON_EXTS = {".py", ".pyw", ".pyi"}

# Exclusions (folder names)
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    ".tox",
    ".idea",
    ".vscode",
}

# ---------- data models ----------
@dataclass
class Issue:
    tool: str
    code: Optional[str]
    message: str
    path: str
    line: Optional[int]
    column: Optional[int]
    severity: Optional[str]
    extra: Dict[str, Any]

@dataclass
class FileReport:
    path: str
    issues: List[Issue]
    summary: Dict[str, Any]

@dataclass
class MasterReport:
    root_dir: str
    scanned_files: List[str]
    tool_versions: Dict[str, str]
    totals: Dict[str, Any]
    generated_at: str
    environment: Dict[str, str]
    per_file: List[FileReport]


# ---------- helpers ----------
def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(args: List[str]) -> Tuple[int, str, str]:
    """Run command, return (returncode, stdout, stderr) without raising."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"

def gather_python_files(root_dir: str) -> List[str]:
    files: List[str] = []
    root = Path(root_dir)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in PYTHON_EXTS:
                files.append(str(p))
    return sorted(files, key=str.lower)

def get_tool_version(tool: str, version_arg: str = "--version") -> Optional[str]:
    exe = which(tool)
    if not exe:
        return None
    rc, out, err = run_cmd([exe, version_arg])
    text = out.strip() or err.strip()
    return text or None

# ---------- Ruff parsing ----------
def run_ruff(file_path: str) -> List[Issue]:
    exe = which("ruff")
    if not exe:
        return []
    # Ruff JSON output is a list of objects
    # ruff check --format json file.py
    rc, out, err = run_cmd([exe, "check", "--format", "json", file_path])
    if not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        # Some ruff versions may emit non-JSON extras; try to isolate the JSON array
        m = re.search(r"(\[.*\])", out, re.S)
        data = json.loads(m.group(1)) if m else []
    issues: List[Issue] = []
    for item in data:
        issues.append(
            Issue(
                tool="ruff",
                code=item.get("code"),
                message=item.get("message", ""),
                path=item.get("filename", file_path),
                line=(item.get("location") or {}).get("row"),
                column=(item.get("location") or {}).get("column"),
                severity=item.get("severity"),
                extra={
                    "fix": item.get("fix"),
                    "url": item.get("url"),
                    "end_location": item.get("end_location"),
                },
            )
        )
    return issues

# ---------- Pylint parsing ----------
def run_pylint(file_path: str) -> List[Issue]:
    exe = which("pylint")
    if not exe:
        return []
    # pylint --output-format=json file.py
    rc, out, err = run_cmd([exe, "--output-format=json", file_path])
    if not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    issues: List[Issue] = []
    for d in data:
        issues.append(
            Issue(
                tool="pylint",
                code=d.get("message-id"),
                message=d.get("message", ""),
                path=d.get("path", file_path),
                line=d.get("line"),
                column=d.get("column"),
                severity=d.get("type"),  # convention, refactor, warning, error, fatal
                extra={
                    "symbol": d.get("symbol"),
                    "module": d.get("module"),
                    "obj": d.get("obj"),
                },
            )
        )
    return issues

# ---------- Mypy parsing ----------
def run_mypy(file_path: str) -> List[Issue]:
    exe = which("mypy")
    if not exe:
        return []
    # mypy --error-format=json file.py
    rc, out, err = run_cmd([exe, "--hide-error-context", "--show-column-numbers", "--error-format=json", file_path])
    if not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    issues: List[Issue] = []
    for msg in data.get("messages", []):
        issues.append(
            Issue(
                tool="mypy",
                code=msg.get("code"),
                message=msg.get("message", ""),
                path=msg.get("path", file_path),
                line=msg.get("line"),
                column=msg.get("column"),
                severity=msg.get("severity"),
                extra={
                    "end_line": msg.get("endLine"),
                    "end_column": msg.get("endColumn"),
                },
            )
        )
    return issues

# ---------- Bandit parsing ----------
def run_bandit(file_path: str) -> List[Issue]:
    exe = which("bandit")
    if not exe:
        return []
    # bandit -f json -q -r file.py (bandit prefers recursive, but for a single file it works)
    rc, out, err = run_cmd([exe, "-f", "json", "-q", "-r", file_path])
    if not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    results = data.get("results", [])
    issues: List[Issue] = []
    for r in results:
        issues.append(
            Issue(
                tool="bandit",
                code=r.get("test_id"),
                message=r.get("issue_text", ""),
                path=r.get("filename", file_path),
                line=r.get("line_number"),
                column=None,
                severity=r.get("issue_severity"),
                extra={
                    "issue_confidence": r.get("issue_confidence"),
                    "more_info": r.get("more_info"),
                },
            )
        )
    return issues

def summarize(issues: List[Issue]) -> Dict[str, Any]:
    by_tool: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for i in issues:
        by_tool[i.tool] = by_tool.get(i.tool, 0) + 1
        if i.severity:
            key = str(i.severity).lower()
            by_severity[key] = by_severity.get(key, 0) + 1
    return {
        "total_issues": len(issues),
        "by_tool": by_tool,
        "by_severity": by_severity,
    }

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main() -> None:
    root = Path(ROOT_DIR).resolve()
    if not root.exists():
        print(f"ERROR: ROOT_DIR does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"Scanning for Python files under: {root}")
    files = gather_python_files(str(root))
    if not files:
        print("No Python files found.")
        save_json(MASTER_REPORT, asdict(MasterReport(
            root_dir=str(root),
            scanned_files=[],
            tool_versions={},
            totals={"files": 0, "issues": 0},
            generated_at=datetime.utcnow().isoformat() + "Z",
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            per_file=[],
        )))
        return

    # Detect tool versions (presence implies usage)
    tool_versions = {
        "ruff": get_tool_version("ruff") or "not found",
        "pylint": get_tool_version("pylint") or "not found",
        "mypy": get_tool_version("mypy") or "not found",
        "bandit": get_tool_version("bandit") or "not found",
    }

    print("Tools detected:")
    for t, v in tool_versions.items():
        print(f"  - {t}: {v}")

    per_file_reports: List[FileReport] = []
    total_issues = 0

    for idx, fp in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Linting: {fp}")
        file_issues: List[Issue] = []
        # Run all available tools
        file_issues.extend(run_ruff(fp))
        file_issues.extend(run_pylint(fp))
        file_issues.extend(run_mypy(fp))
        file_issues.extend(run_bandit(fp))

        rep = FileReport(
            path=fp,
            issues=file_issues,
            summary=summarize(file_issues),
        )
        per_file_reports.append(rep)
        total_issues += len(file_issues)

        # Write per-file report
        out_path = OUTPUT_DIR / (Path(fp).name + ".json")
        save_json(out_path, {
            "path": rep.path,
            "issues": [asdict(i) for i in rep.issues],
            "summary": rep.summary,
        })

    master = MasterReport(
        root_dir=str(root),
        scanned_files=files,
        tool_versions=tool_versions,
        totals={"files": len(files), "issues": total_issues},
        generated_at=datetime.utcnow().isoformat() + "Z",
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        per_file=per_file_reports,
    )

    save_json(MASTER_REPORT, {
        "root_dir": master.root_dir,
        "scanned_files": master.scanned_files,
        "tool_versions": master.tool_versions,
        "totals": master.totals,
        "generated_at": master.generated_at,
        "environment": master.environment,
        "per_file": [
            {
                "path": r.path,
                "summary": r.summary,
                # Avoid duplicating all issues again here (the per-file JSONs have them).
                # For convenience, include a small preview count:
                "issue_count": r.summary["total_issues"],
            }
            for r in master.per_file
        ],
    })

    print("\n✅ Done.")
    print(f"- Master report: {MASTER_REPORT.resolve()}")
    print(f"- Per-file reports: {OUTPUT_DIR.resolve()}")
    print(f"- Files checked ({len(files)}):")
    for p in files:
        print(f"  • {p}")

if __name__ == "__main__":
    main()
