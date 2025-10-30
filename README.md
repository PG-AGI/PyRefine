# PyRefine Step-by-Step Guide

PyRefine bundles formatting, linting, and VS Code configuration into a single workflow. Follow the steps below to drop it into any Python project and manage everything through one command.

---

## Step 1. Prepare your project repository

- **Existing project**: make sure the repository is under Git and you can run Python from the project root.
- **New project**: create a folder, run `git init`, and decide where your application code will live.

Throughout this guide we will call that folder the **project root**.

---

## Step 2. Add PyRefine to the project

From the project root, clone or copy the PyRefine folder:

```bash
git clone https://github.com/PG-AGI/PyRefine.git
```

Commit the `PyRefine/` directory if you want the tooling tracked with your codebase.

---

## Step 3. Create and activate a virtual environment

Set up an isolated interpreter so PyRefine's tools do not clash with your global site-packages:

```bash
python -m venv .venv
# Activate it
source .venv/bin/activate        # macOS or Linux
.\.venv\Scripts\activate         # Windows PowerShell
```

Install the required tooling packages:

```bash
pip install -r PyRefine/requirements.txt
```

---

## Step 4. Use the PyRefine CLI (single entry point)

All automation now lives in one command with three flags. Run them from the project root:

| Command | Purpose |
| --- | --- |
| `python PyRefine/tools/pyrefine.py --create` | Creates the standard Python scaffold (`src/`, `tests/`, `configs/`, `scripts/` + starter files). |
| `python PyRefine/tools/pyrefine.py --clean [path]` | Formats a file, directory, or the entire project (`.`). Directories are tidied (caches removed) before the Autoflake -> Isort -> Autopep8 -> Black -> Flake8 pipeline runs. |
| `python PyRefine/tools/pyrefine.py --setup` | Creates or merges `.vscode/settings.json` and `.vscode/extensions.json` so VS Code runs PyRefine's formatting on save. |

Running `--create` also drops a `.flake8` file into your project if one is not already present so that formatting rules are available immediately.

Use `--project-root /absolute/path` with any command to target a different repository.

---

## Step 5. Optional commands

- Recreate the VS Code workspace files at any time:

  ```bash
  python PyRefine/tools/setup_workspace.py --project-root /absolute/path --force
  ```

- Run the formatter directly when you already know the scope:

  ```bash
  python PyRefine/tools/format.py all
  python PyRefine/tools/format.py /absolute/path/to/file.py
  python PyRefine/tools/format.py /absolute/path/to/folder
  ```

---


## Step 6. Build a standalone executable (optional)

1. Activate your virtual environment and install PyInstaller: `pip install pyinstaller`.
2. Run `python PyRefine/tools/build_exe.py`.
3. The packaged binary (`pyrefine.exe`) will be created inside the `dist/` directory. Move or distribute it as needed.

## Reference: files included in PyRefine

| Item | Purpose |
| --- | --- |
| `tools/pyrefine.py` | Command-line utility providing --create, --clean, and --setup actions for project scaffolding, formatting, and VS Code integration. |
| `tools/bootstrap.py` | Legacy guided script that still supports scaffold generation and bulk formatting. |
| `tools/setup_workspace.py` | Writes or merges `.vscode/settings.json` and `.vscode/extensions.json` for a given project root. |
| `tools/format.py` | Runs the Autoflake -> Isort -> Autopep8 -> Black -> Flake8 pipeline. Used by the CLI and by VS Code on save. |
| `.flake8` | Shared lint configuration (79 character lines, cache directories ignored). |
| `requirements.txt` | Tooling dependencies to install inside your virtual environment. |
| `.vscode/` | Workspace defaults (format-on-save settings and extension recommendations). Recreated automatically if removed. |
| `tools/build_exe.py` | Convenience wrapper around PyInstaller to produce `pyrefine.exe`. |

---

## Next steps

- Commit the updated `.vscode/` folder and any cleaned scaffolding so teammates inherit the same setup.
- Add a CI job that runs `python PyRefine/tools/format.py all --lint-only` to keep pull requests consistent.
- Explore advanced tooling (SonarLint, Bandit, type checkers) by extending `requirements.txt` and the CLI if needed.

Run the CLI whenever you create or adopt a project to keep your Python repositories clean, consistent, and VS Code ready.
