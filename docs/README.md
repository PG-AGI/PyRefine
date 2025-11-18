# PyRefine

## 1. How to use this project (Mac, Windows, Ubuntu)

- **macOS**
  ```bash
  chmod +x pyrefine-macos
  ./pyrefine-macos --setup      # other commands work with the same binary
  ```
- **Ubuntu**
  ```bash
  chmod +x pyrefine-linux
  ./pyrefine-linux --setup
  ```
- **Windows**
  ```powershell
  pyrefine.exe --setup
  ```
- Add `--project-root C:\path\to\project` (or `/path/to/project`) if the binary isn’t placed directly in the repo root.

## 2. Commands

- `pyrefine.exe` – defaults to `--clean .`, removing caches and formatting the current repo.
- `pyrefine --clean <path>` – formats the given file/folder/project using Autoflake → Isort → Autopep8 → Black → Flake8.
- `pyrefine --create` – generates the standard structure (src/tests/configs/scripts/utils/services, FastAPI app, Dockerfile, README, etc.).
- `pyrefine --setup` – writes VS Code settings, installs recommended extensions, and provisions `.venv` plus `.uv-env`.
- `pyrefine --test-coverage [path]` – runs pytest with coverage for the current project or a specified directory, saving reports under `pyrefine_artifacts/<project>/coverage`.
- `pyrefine --update [--manifest-url URL]` – downloads and applies the latest packaged release (Windows exe replaces itself automatically).
- `pyrefine --version` – prints the version of the binary you’re running.

## 3. Command Execution Flow

- **`--create`** – `cli/pyrefine.py` routes to `commands/create/scaffold_manager.ensure_scaffold`, which creates `src/tests/configs/scripts/utils/services`, drops template FastAPI/test files, writes `.gitignore`, `requirements.txt`, Dockerfile, and copies the shared `.flake8`.
- **`--clean [PATH]`** – `commands/clean/clean_manager.run_clean` figures out the absolute target, removes cache folders/files that aren’t in `.gitignore`, ensures `.flake8` exists, then runs `commands/clean/format.py` (Autoflake → Isort → Autopep8 → Black → Flake8) on the resolved Python files.
- **`--setup`** – `commands/setup/setup_manager.run_setup` writes/merges VS Code settings via `shared/common_vscode.py`, installs pip deps in `.venv`, ensures the `uv` CLI is present, and creates a `.uv-env` that mirrors `requirements.txt`.
- **`--test-coverage [path]`** – `commands/test_coverage/coverage_runner.run_for_projects` loads `.gitignore`, determines which project to test, runs `python -m coverage run -m pytest`, and saves HTML/XML/text reports to `pyrefine_artifacts/<project>/coverage`.
- **`--update`** – `commands/update/update_manager.handle_update` downloads `release/manifest.json`, compares versions, selects the OS-specific artifact (Windows/macOS/Linux), verifies the checksum, and swaps in the new binary (Windows uses a helper script; macOS/Linux replace in-place with a backup).
- **`pyrefine.exe` (no flag)** – By design, the CLI defaults to `--clean .`, so running the binary without flags simply formats the current repository using the flow above.

## 4. Future Work

- Optional Docker compose templates for multi-service repos.  
- Poetry support alongside pip/UV for dependency management.  
- Dashboard-style output summarizing formatting, coverage, and scaffold status.  
