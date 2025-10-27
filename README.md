# Python Starter Template

This repository provides a lightweight Python-first starter kit with opinionated formatting and a single linting standard (Flake8). Drop the `PyRefine` folder into any project and use the automation to clean or scaffold the codebase in minutes.

## 1. Bootstrap the workspace

```bash
python tools/bootstrap.py
```

- First choose whether you want to **clean** the current repository or **create** a fresh project structure.
- The script detects the project root (the parent directory that contains the `PyRefine` folder). Confirm the path or supply a different absolute path.
- Cleaning runs the formatter across the project; creating scaffolds `src/`, `tests/`, `configs/`, and `scripts/`, installs dependencies (optional), then offers to format straight away.
- The bootstrapper installs `requirements.txt`, attempts to install the recommended VS Code extensions, and removes conflicting ones (Pylint, Ruff, Pylance) when the `code` CLI is available.

## 2. VS Code configuration

The workspace already points VS Code to:

- Format on save with **Black**, organised through **Isort**, using the shared automation.
- Enable linting through the **Flake8** extension only.
- Prefer the Jedi language server to avoid the Pylance linting overlap.

Recommended extensions are declared in `.vscode/extensions.json`, while conflicting ones are listed under `unwantedRecommendations` so VS Code suggests disabling them if present.

## 3. Formatting & linting on demand

Use the shared tooling whether you are in an editor or on the command line:

- `python tools/format.py all` (or `./format_code.sh`) processes the entire project using Black, Isort, Autoflake, Autopep8, and Flake8.
- `python tools/format.py /absolute/path/to/file.py` formats a single file.
- `python tools/format.py /absolute/path/to/directory` formats every Python file under the specified folder.
- The script expects absolute paths for file and directory targets and exits without action if you omit arguments.

On file save, VS Code invokes the same script via the Run on Save extension, ensuring editor and CLI behaviour remain perfectly aligned.

## 4. Customising the template

- Update `.flake8` if you need additional ignores or plugins.
- Adjust the default folders in `tools/bootstrap.py` to match your organisation's skeleton.
- Extend `requirements.txt` with runtime or tooling dependencies that should ship with the template.

Happy coding! Launch the bootstrap script whenever you clone this template to ensure every project starts with the same, predictable Python toolchain. 
