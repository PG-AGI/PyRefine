# PyRefine – Working Notes (Developer Guide)

This guide explains every PyRefine command from two perspectives:

1. **Functional overview** – what the command does for the user.
2. **Code flow & dependencies** – which files/functions run under the hood.

## Runtime model & environment

- **Entry point**: `cli/pyrefine.py` parses CLI flags, enforces “one action per run,” and dispatches to matching handlers (`handle_clean`, `handle_create`, etc.).
- **Resource discovery**: `get_resource_root()` resolves bundled assets when running as a PyInstaller binary (`pyrefine.exe`) and falls back to the repo layout when running from source.
- **Execution modes**: helper scripts run via `runpy.run_path()` in frozen mode and via `subprocess` when running from source. `PYREFINE_PROJECT_ROOT` pins the formatter/cleaner context to the user project in both cases.
- **Key environment variables**:
  - `PYREFINE_PROJECT_ROOT` – forces formatter and clean workflows to treat a specific directory as root.
  - `PYREFINE_IGNORE_DIRS` – comma-separated directory names that the formatter must ignore in addition to built-in patterns.
  - `PYREFINE_UPDATE_URL` – alternate manifest for the updater.
  - `VIRTUAL_ENV` – preferred search root for formatter executables.

## Command reference

### Clean (`--clean`, default command)

#### Functional overview

- Removes caches/build artefacts defined in `CLUTTER_DIR_PATTERNS` and `CLUTTER_FILE_PATTERNS`.
- Ensures `.flake8` exists by copying the bundled template when missing.
- Runs the formatter pipeline in order: **string fixer → autoflake → isort → autopep8 → black → flake8**.
- Honors `.gitignore` (ignored paths are skipped and ignored directories are not traversed).

#### Code flow & dependencies

1. `cli/pyrefine.py::handle_clean()` resolves `project_root`, passes args to `commands.clean.clean_manager.run_clean()`.
2. `run_clean()` loads `.gitignore` via `shared.gitignore_utils.load_gitignore_spec()`, normalizes the target path, and calls `clean_target()`.
3. `clean_target()`  
   - Validates targets (existence, non-ignored).  
   - Calls `remove_clutter()` to delete cache folders/files.  
   - Invokes `ensure_flake8_config()` at the project root.  
   - Delegates to `run_formatter(target, project_root, format_script)`.
4. `run_formatter()` exports `PYREFINE_PROJECT_ROOT`, then either runs `commands/clean/format.py` via `runpy` (frozen) or spawns `sys.executable format.py <target>`.
5. `commands.clean.format.main()` collects targets and executes:  
   - `run_string_fixer()` (calls `string_fixer.fix_file(create_backup=False)` for every file and prints stats/errors).  
   - `run_autoflake()` (optional), `run_isort()` (required), `run_autopep8()` (optional), `run_black()` (required), `run_flake8()` (required).  
   - `find_executable()` searches `VIRTUAL_ENV`, project `.venv`/`env`, bundled envs, then the system `PATH`.

### Create (`--create`)

#### Functional overview

- Initializes the standardized project skeleton (src/tests/configs/scripts/utils/services).
- Generates starter files: FastAPI app, sample tests, `.env.example`, `requirements.txt`, `Dockerfile`, `.gitignore`, `.flake8`.
- Prints which directories/files were created so users can verify scaffolding.

#### Code flow & dependencies

1. `cli/pyrefine.py::handle_create()` resolves `project_root` and calls `commands.create.scaffold_manager.ensure_scaffold()`.
2. `ensure_scaffold()`  
   - `_ensure_directories()` creates folders listed in `PROJECT_DIRECTORIES`.  
   - `_ensure_template_files()` writes the content in `FILE_TEMPLATES`.  
   - `_write_file_if_missing()` handles `.gitignore`, `requirements.txt`, `Dockerfile`.  
   - `ensure_flake8()` copies the bundled `.flake8` template via the resolved `resource_root`.

### Setup (`--setup`)

#### Functional overview

- Configures VS Code (`settings.json`, `extensions.json`, run-on-save hook) to call the formatter embedded inside PyRefine.
- Creates or reuses `.venv`, upgrades pip, and installs `requirements.txt` when present.
- Ensures the `uv` CLI/environment exists (stored under `.uv-env`) for alternate workflows.

#### Code flow & dependencies

1. `cli/pyrefine.py::handle_setup()` passes `project_root` and `resource_root` to `commands.setup.setup_manager.run_setup()`.
2. `configure_vscode()`  
   - Reads existing `.vscode` files (if any).  
   - Uses `shared.common_vscode.build_settings_payload()` / `build_extensions_payload()` to compose desired state.  
   - Merges via `shared.common_vscode.merge_dict()` and `merge_run_on_save()` so user settings are preserved.  
   - Calls `ensure_pylance_extension()` and `ensure_flake8_extension()` (CLI installs when possible, otherwise prints advice).
3. `create_pip_environment()`  
   - Resolves a base interpreter via `_get_base_python()` (system `python` when frozen).  
   - Creates `.venv` if missing, upgrades pip, installs dependencies from `requirements.txt` when available.
4. `create_uv_environment()`  
   - Ensures the `uv` binary exists globally or inside `.venv`.  
   - Creates `.uv-env`, then installs requirements via `uv pip install --python <env_python> -r requirements.txt`.

### Test coverage (`--test-coverage`)

#### Functional overview

- Validates target directories and runs `pytest` with coverage, one project at a time.
- Stores artifacts under `pyrefine_artifacts/<project>/coverage` (XML/HTML reports, summary, `.coverage` copy).
- Provides consistent coverage outputs for CI and manual audits.

#### Code flow & dependencies

1. `cli/pyrefine.py::handle_test_coverage()`  
   - Resolves the execution root (handles frozen executables with `_execution_project_root()`).  
   - Loads `.gitignore`, ensures the target exists/is a directory/is not ignored.  
   - Builds the list of project paths and calls `commands.test_coverage.coverage_runner.run_for_projects()`.
2. `coverage_runner.run_for_projects()` iterates each project, printing status and calling `run_pytest_with_coverage()`.
3. `run_pytest_with_coverage()`  
   - Uses `_python_executable()` to select a real interpreter (system Python in frozen mode).  
   - Executes `coverage run -m pytest`, then `coverage xml`, `coverage html`, and `coverage report`.  
   - Writes outputs into `pyrefine_artifacts/<project>/coverage` and copies `.coverage` if present.

### Update (`--update`)

#### Functional overview

- Checks the manifest JSON for the latest release, compares versions, downloads the platform-specific binary, and swaps it in place.
- Only applies when running as a packaged executable (source runs exit with a clear message).
- Validates SHA256 checksums and performs safe replacement (Windows uses a helper `.cmd`, POSIX systems swap files with a `.bak` backup).

#### Code flow & dependencies

1. `cli/pyrefine.py::handle_update()` forwards flags to `commands.update.update_manager.handle_update()` along with the current version and default manifest URL.
2. `handle_update()`  
   - Resolves the manifest URL (`resolve_manifest_url()` considers `--manifest-url` and `PYREFINE_UPDATE_URL`).  
   - Downloads/parses JSON via `fetch_manifest()`.  
   - Compares versions using `is_newer_version()` / `parse_version()`.  
   - Ensures `sys.frozen` is set before continuing.  
   - Chooses the artifact entry (`select_artifact()` prioritizes platform-specific keys, then `default`).  
   - Calls `download_release_binary()` which streams the file and verifies checksums.
3. `apply_update_binary()`  
   - On Windows: stages the download, writes `pyrefine_update.cmd`, and schedules replacement via `cmd.exe` (`schedule_windows_replace()`).  
   - On POSIX: renames the current executable to `.bak`, promotes the staged `.new` file, and sets executable permissions.

### Build executable (`commands/build_exe/build_exe.py`)

#### Functional overview

- Developer-facing helper that packages PyRefine with PyInstaller.
- Copies required data files (formatter script, `.flake8`) into the bundle so runtime tooling works offline.

#### Code flow & dependencies

1. `build_exe.py` resolves repo root, entry script (`cli/pyrefine.py`), and data files (`commands/clean/format.py`, `configs/.flake8`).
2. `add_data()` registers each file with `PyInstaller.__main__.run()` under the `PyRefine/` prefix.
3. PyInstaller runs with `--onefile --clean`, producing the standalone executable in `dist/`.

## Clean pipeline deep-dive

| Stage | Purpose | Implementation |
| ----- | ------- | -------------- |
| Target discovery | Collect Python files while honoring `.gitignore`, `IGNORED_DIR_NAMES`, and `PYREFINE_IGNORE_DIRS`. | `commands.clean.format.collect_targets()` drives `iter_python_files()` which filters via `should_skip_dir()` / `should_include_file()`. |
| String fixer | Split long strings, docstrings, and comments before other tools reflow code. | `run_string_fixer()` imports `string_fixer.FixStats`, calls `fix_file(..., create_backup=False)` for each target, and prints summary/error lines. |
| Formatters | Apply traditional code formatters and linting. | `run_autoflake()` (optional), `run_isort()` (required), `run_autopep8()` (optional), `run_black()` (required), `run_flake8()` (required). |
| Tool resolution | Find executables consistently across dev and frozen environments. | `find_executable()` searches active `VIRTUAL_ENV`, project `.venv`/`env`, bundled envs, then system `PATH`. |

## Shared helpers & conventions

- `shared.gitignore_utils`  
  - `load_gitignore_spec()` reads project `.gitignore`; commands treat missing files as fatal (`GitignoreError`).  
  - `is_gitignored()` is used by clean, coverage, and other filesystem traversals.
- `shared.common_vscode`  
  - Provides payload builders for VS Code settings/extensions and merge helpers that avoid duplicate run-on-save entries.  
  - `pylance_installed()` / `notify_pylance_missing()` offer guidance when the extension is absent.
- Error signals: `GitignoreError`, `CoverageError`, `UpdateError` bubble up to the CLI and become user-friendly exit codes; formatter subprocesses propagate `subprocess.CalledProcessError` so exit statuses mirror the failing tool.

## Extending PyRefine

- Add new commands by wiring parser flags in `cli/pyrefine.py`, creating a manager in `commands/<area>/`, and keeping shared helpers in `shared/`.
- When inserting new formatter stages, update `commands/clean/format.py` so the string fixer still runs first, and adjust/extend tests under `tests/`.
- Prefer exporting `PYREFINE_PROJECT_ROOT` before running subprocess scripts so packaged binaries behave identically to source runs.

---

Update this document whenever a command’s functional behavior or code entrypoints change.
