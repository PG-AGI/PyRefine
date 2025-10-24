# PyRefine Developer Notes

This repository contains a collection of Python utilities and sample services that are analysed with a heavy linting and formatting toolchain. The new VS Code workspace configuration helps keep all of the Python sources consistently structured.

## Environment setup

- Create/refresh the local virtual environment in `env` and install the tooling with `pip install -r requirements.txt`.
- Open the folder in VS Code; the workspace recommends the Python, Black, Ruff, Isort, and Run on Save extensions.
- When saving any `.py` file, VS Code invokes `tools/run_toolchain_on_save.py` so Black, Isort, Autoflake, Autopep8, and Unimport run automatically in sequence. The default formatter is Black and import fixes run via code actions on save.

## Manual formatting

- Run `./format_code.sh` to execute the same toolchain across the main backend and test script directories.
- Pass a specific file or folder to target only part of the project, e.g. `./format_code.sh test_py_scripts/main.py`.
- Append `--check` to preview formatting issues without modifying files (currently implemented for Black and Isort).
