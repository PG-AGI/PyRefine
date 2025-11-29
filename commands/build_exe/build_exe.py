#!/usr/bin/env python3
"""
Module Purpose:
    Builds the PyRefine standalone executables via PyInstaller with consistent
     data assets.

Key Components:
    - main: Configures PyInstaller arguments and triggers the build process.
    - add_data: Registers resource files so the bundled executable can locate
     templates.

Project Contribution:
    Enables shipping PyRefine as platform-specific binaries, ensuring users
     can run the automation toolkit without installing Python or managing
     source layouts manually.

new version
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
    root = Path(__file__).resolve().parents[2]
    pyrefine_entry = root / "cli" / "pyrefine.py"
    format_script = root / "commands" / "clean" / "format.py"
    flake8_file = root / "configs" / ".flake8"
    hidden_imports = [
        "string_fixer",
        "shared.common_vscode",
        "shared.gitignore_utils",
    ]

    datas: list[str] = []
    hidden_args: list[str] = []

    def add_data(src: Path, dest: str) -> None:
        if src.exists():
            datas.extend(["--add-data", f"{src}{os.pathsep}{dest}"])

    def add_hidden_import(module: str) -> None:
        hidden_args.extend(["--hidden-import", module])

    add_data(format_script, "PyRefine/commands/clean")
    add_data(flake8_file, "PyRefine/configs")
    for module in hidden_imports:
        add_hidden_import(module)

    PyInstaller.__main__.run(
        [
            "--name",
            "pyrefine",
            "--onefile",
            "--clean",
            *datas,
            *hidden_args,
            str(pyrefine_entry),
        ]
    )


if __name__ == "__main__":
    main()
