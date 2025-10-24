# Python Starter Template

This repository provides a lightweight Python-first starter kit with opinionated formatting and a single linting standard (Flake8). Clone it into an existing codebase or use it to scaffold a new project structure in seconds.

## 1. Bootstrap the workspace

```bash
python tools/bootstrap.py
```

- Choose **existing** if you dropped the template into an established project. The script offers to format everything immediately and lets you target additional files or folders on demand.
- Choose **new** to generate a minimal structure (`src/`, `tests/`, `configs/`, `scripts/`) with starter files.
- The bootstrapper installs `requirements.txt`, attempts to install recommended VS Code extensions, and uninstalls the conflicting ones (Pylint, Ruff, Pylance) when the `code` CLI is available.

## 2. VS Code configuration

The workspace already points VS Code to:

- Format on save with **Black** and organise imports with **Isort**.
- Enable linting through the **Flake8** extension only.
- Prefer the Jedi language server to avoid the Pylance linting overlap.

Recommended extensions are declared in `.vscode/extensions.json`, while conflicting ones are listed under `unwantedRecommendations` so VS Code suggests disabling them if present.

## 3. Formatting & linting on demand

Use the shared tooling whether you are in an editor or on the command line:

- `./format_code.sh --all` formats every recognised project directory and runs Flake8.
- `./format_code.sh path/to/file.py` or `./format_code.sh path/to/folder` narrows the scope.
- Append `--check` to perform a dry run (Black/Isort), or `--lint-only` to run Flake8 without touching files.

Under the hood, both the shell helper and the VS Code setup call `tools/formatting.py`, which locates the preferred interpreter (project `env/` or active virtualenv) before invoking Black, Isort, and Flake8 in sequence.

## 4. Customising the template

- Update `.flake8` if you need additional ignores or plugins.
- Adjust the default folders in `tools/bootstrap.py` to match your organisation's skeleton.
- Extend `requirements.txt` with runtime or tooling dependencies that should ship with the template.

Happy coding! Launch the bootstrap script whenever you clone this template to ensure every project starts with the same, predictable Python toolchain. 
