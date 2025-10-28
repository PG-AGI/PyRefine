# PyRefine Starter Guide

Use this repository to drop a ready-to-go Python tooling stack into any project. Follow the steps below in order—each one builds on the previous and explains the prompts you will see.

---

## Step 1 — Prepare (or create) your project repository

- **Existing codebase**: ensure it is already a Git repository and that you can run Python from the project root.
- **New project**: create an empty directory, run `git init`, and decide where your application code will live.
- From now on, we will call this directory the **project root**.

---

## Step 2 — Bring in PyRefine

Inside the project root, clone or copy the PyRefine folder:

```bash
git clone https://github.com/PG-AGI/PyRefine.git
```

The structure should now look similar to:

```
your-project/
├─ PyRefine/
│  ├─ tools/
│  ├─ .flake8
│  └─ requirements.txt
└─ (your application files)
```

Commit the `PyRefine` folder if you want the tooling tracked with your project.

---

## Step 3 — Bootstrap the workspace

Run the bootstrap script from the PyRefine directory:

```bash
python PyRefine/tools/bootstrap.py
```

During execution you will see the following prompts:

| Prompt                                                                           | Purpose                                                                                                                      | Typical answer                |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `Do you want to clean an existing repository or create a new project structure?` | Choose **clean** to format existing code, or **create** to generate skeleton folders (`src`, `tests`, `configs`, `scripts`). | Based on your project state   |
| `Detected project root ... Use this directory?`                                  | Confirms the parent folder that will be formatted. You can enter a different absolute path if needed.                        | Usually `Y`                   |
| `Install Python dependencies from requirements.txt?`                             | Installs Black, Isort, Autoflake, Autopep8, and Flake8 into your current interpreter/venv.                                   | `Y` (recommended)             |
| `Generate the default project structure ... ?` (only in **create** mode)         | Creates the scaffold folders and placeholder files.                                                                          | `Y` if you are starting fresh |
| `Format the new project now?` / automatic formatting                             | Optionally runs the formatter across the project immediately.                                                                | `Y` when ready                |
| `Format another file or folder?`                                                 | Lets you target additional paths right away.                                                                                 | `N` to finish                 |

Behind the scenes the script also:

- Suggests/installs VS Code extensions (Black, Isort, Flake8, Run on Save).
- Removes conflicting ones (Pylint, Ruff, Pylance) if installed.
- Writes `.vscode/settings.json` and `.vscode/extensions.json` in the project root.
- Exports `PYREFINE_PROJECT_ROOT` when calling the formatter so the tools operate on your chosen directory.

You can re-run the script later with flags such as `--mode clean`, `--skip-format`, or `--skip-deps` if you want to automate specific parts.

---

## Step 4 — (Optional) Recreate workspace settings manually

If you ever delete or move the project and need the `.vscode` files again, run:

```bash
python PyRefine/tools/setup_workspace.py --project-root /abs/path/to/project --force
```

This writes the standard formatter/linter configuration into the specified project root. Use it when you open the parent folder in VS Code instead of the `PyRefine` subdirectory.

---

## Step 5 — Format and lint on demand

`tools/format.py` is the single entry point for automated cleanup:

| Command                                                | What it does                                                                                       |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `python PyRefine/tools/format.py all`                  | Runs Autoflake → Isort → Autopep8 → Black → Flake8 across the entire project (79-char line limit). |
| `python PyRefine/tools/format.py /abs/path/to/file.py` | Formats and lints one file.                                                                        |
| `python PyRefine/tools/format.py /abs/path/to/folder`  | Recursively processes all Python files under that folder.                                          |
| `python PyRefine/tools/format.py --lint-only ...`      | Skips the formatters and runs only Flake8.                                                         |

Notes:

- All paths must be absolute. The script exits without doing anything if you omit the target.
- VS Code’s Run-on-Save extension invokes the same script automatically for whichever file you save, so editor and CLI behaviour stay in sync.

---

## Reference — What each file is for

| File / folder                                       | Purpose                                                                                                                                                |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tools/bootstrap.py`                                | Guided setup: detects the project root, installs dependencies, manages VS Code extensions, creates scaffold folders, and optionally formats your code. |
| `tools/setup_workspace.py`                          | Rewrites `.vscode/settings.json` and `.vscode/extensions.json` for any project root (handy when you open the parent folder in VS Code).                |
| `tools/format.py`                                   | Core automation that runs Autoflake, Isort, Autopep8, Black, and Flake8. Used by both the CLI and VS Code on-save hook.                                |
| `.flake8`                                           | Repository-wide lint configuration (line length 79, ignore certain caches). Adjust if you need custom rules.                                           |
| `requirements.txt`                                  | Tooling dependencies to install into your environment.                                                                                                 |
| `.vscode/settings.json` & `.vscode/extensions.json` | Generated workspaces files that configure format-on-save and recommended extensions. Recreate with `setup_workspace.py` if deleted.                    |

---

## Next steps

- Commit the `.vscode` folder and any scaffold files you need so teammates inherit the same configuration.
- Add a CI job that runs `python PyRefine/tools/format.py all --lint-only` to fail builds when the code drifts.
- Explore additional tooling (e.g., SonarLint, Bandit) by editing `requirements.txt` and the scripts as required.

With these steps your project stays consistently formatted, linted, and ready for Python development on every machine. Happy coding!
