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

Set up an isolated interpreter so PyRefine’s tools do not clash with your global site-packages:

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

## Step 4. Run the PyRefine CLI (single entry point)

Launch the interactive utility from the project root:

```bash
python PyRefine/tools/pyrefine_cli.py
```

You will be guided through three phases:

1. **Project structure setup**
   - *Create new template*: scaffold `src/`, `tests/`, `configs/`, and `scripts/` folders with starter files.
   - *Clean existing repository*: remove caches, build artifacts, and compiled files while keeping core folders.
   - *Skip*: leave the repo exactly as it is.
2. **VS Code on-save settings**
   - One-line summary: `Enable VS Code on-save rules?` (Yes/No).
   - Choose **Yes** to create or merge `.vscode/settings.json` and `.vscode/extensions.json`; choose **No** to leave existing settings untouched.
3. **Formatting options**
   - Decide whether to format a single file, process an entire folder recursively, or skip formatting for now.

Use `--project-root /absolute/path` to target another directory, and `--yes` to auto-accept default answers (handy for automation).

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

## Reference: files included in PyRefine

| Item | Purpose |
| --- | --- |
| `tools/pyrefine_cli.py` | Interactive CLI that orchestrates cleanup, VS Code setup, and formatting from one entry point. |
| `tools/bootstrap.py` | Legacy guided script that still supports scaffold generation and bulk formatting. |
| `tools/setup_workspace.py` | Writes or merges `.vscode/settings.json` and `.vscode/extensions.json` for a given project root. |
| `tools/format.py` | Runs the Autoflake -> Isort -> Autopep8 -> Black -> Flake8 pipeline. Used by the CLI and by VS Code on save. |
| `.flake8` | Shared lint configuration (79 character lines, cache directories ignored). |
| `requirements.txt` | Tooling dependencies to install inside your virtual environment. |
| `.vscode/` | Workspace defaults (format-on-save settings and extension recommendations). Recreated automatically if removed. |

---

## Next steps

- Commit the updated `.vscode/` folder and any cleaned scaffolding so teammates inherit the same setup.
- Add a CI job that runs `python PyRefine/tools/format.py all --lint-only` to keep pull requests consistent.
- Explore advanced tooling (SonarLint, Bandit, type checkers) by extending `requirements.txt` and the CLI if needed.

Run the CLI whenever you create or adopt a project to keep your Python repositories clean, consistent, and VS Code ready.
