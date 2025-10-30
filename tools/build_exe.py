#!/usr/bin/env python3
"""
Helper script to build a standalone pyrefine.exe using PyInstaller.

Requires PyInstaller to be installed inside your virtual environment:
    pip install pyinstaller

Usage:
    python PyRefine/tools/build_exe.py
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import PyInstaller.__main__
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyInstaller is required to build the executable. "
        "Install it with 'pip install pyinstaller'."
    ) from exc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pyrefine_entry = root / "tools" / "pyrefine.py"
    format_script = root / "tools" / "format.py"
    setup_script = root / "tools" / "setup_workspace.py"
    flake8_file = root / ".flake8"

    datas: list[str] = []

    def add_data(src: Path, dest: str) -> None:
        if src.exists():
            datas.extend(["--add-data", f"{src}{os.pathsep}{dest}"])

    add_data(format_script, "PyRefine/tools")
    add_data(setup_script, "PyRefine/tools")
    add_data(flake8_file, "PyRefine")

    PyInstaller.__main__.run(
        [
            "--name",
            "pyrefine",
            "--onefile",
            "--clean",
            *datas,
            str(pyrefine_entry),
        ]
    )


if __name__ == "__main__":
    main()
