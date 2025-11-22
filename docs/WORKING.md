# PyRefine — How it works (Developer guide)

This document explains the internal structure and the runtime flow of PyRefine.

## Top-level overview

- Entry point: `cli/pyrefine.py` — parses arguments and dispatches to a single command handler.
- Commands are implemented under `commands/` as managers and helper modules (for example, `commands/clean/clean_manager.py`).
- Shared utilities: `shared/` contains small helpers used by multiple commands (e.g., `.gitignore` logic and VS Code settings builders).
- Resource layout: `Resource root` contains the templates and executables that are bundled or copied into project scaffolds (see `cli/pyrefine.py` for the `RESOURCE_ROOT` computation).

## Execution modes and resource discovery

- When running from source: the `project_root` defaults to the current working directory (CLI param `--project-root` or `Path.cwd()`). The resource root is typically the repo path that holds `commands/`, `configs/`.
- When running as a frozen executable (PyInstaller): `sys.frozen` is used to detect the executable mode and the resource root is resolved from the bundle directory (via `_MEIPASS`).
- Scripts and helper tooling are executed either by importing and running (embedded mode) or by spawning subprocesses using the environment's Python and tools.

### Key environment variables

- `PYREFINE_PROJECT_ROOT` — When provided, overrides the project root for formatter and embedded tooling (for example `format.py` reads this env var to determine `PROJECT_ROOT`).
- `PYREFINE_IGNORE_DIRS` — Comma-separated directory names added to the format pipeline's dynamic ignore list.
- `PYREFINE_UPDATE_URL` — If set, acts as the default manifest URL; otherwise a hard-coded manifest is used.
- `VIRTUAL_ENV` — Used by the formatter and build scripts to prefer a virtual environment's executables.

### Resources and frozen (packaged) mode

- The resource root is discovered via `shared/common_vscode.get_resource_root()` or the CLI-level `get_resource_root()` in `cli/pyrefine.py`.
- When packaged by PyInstaller, a bundle directory is exposed via `sys._MEIPASS`; resource files (for example `format.py` and `.flake8`) are added to the single-file bundle using the `--add-data` option in `build_exe` and are placed under a `PyRefine/` prefix in the pack.
- Many modules have special cases when `sys.frozen` is set; these include using system Python (setup/build), using `runpy.run_path()` to execute embedded scripts, and the update manager only applying to packaged executables.

## Commands and their behavior (summary)

1. --create (scaffold)

   - Implemented by `commands/create/scaffold_manager.py`.
   - Ensures directories such as `src/`, `tests/`, `configs/`, `scripts/`, `utils/`, `services/` exist.
   - Writes template files (small FastAPI example, test suite, `requirements.txt`, `Dockerfile`, `.gitignore`, and `.flake8`) to bootstrap projects.

2. --clean (format/cleanup)

   - Implemented by `commands/clean/clean_manager.py`.
   - Loads `.gitignore` using `shared/gitignore_utils.py` and prunes cache/build artefacts (e.g., `__pycache__`, `.pytest_cache`) respecting `.gitignore`.
     - The clean step prunes both directory and file patterns. The default directory patterns include `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.tox`, `.ipynb_checkpoints`, `.eggs`, `*.egg-info`, `build`, and `dist`. File patterns include `*.pyc`, `*.pyo`, and `*.pyd`.
   - Ensures a `.flake8` config is present (either via bundled resources or generated defaults).
   - Runs the formatter pipeline using `commands/clean/format.py`, which accepts `all` or absolute paths to files/directories.

   - The clean workflow sets `PYREFINE_PROJECT_ROOT` in the formatter environment so the embedded formatting script can resolve project paths consistently. When packaged, the code uses `runpy.run_path()` to execute the embedded `format.py` directly; otherwise it spawns a subprocess invoking `/usr/bin/python` (or `sys.executable`) with the script.

   - Note: In non-frozen mode, the CLI launches `format.py` via a subprocess with `check=False` so pyrefine won't exit immediately on formatter failures; the formatter process controls its own exit status. In packaged mode, `format.py` is executed in-process via `runpy` and exceptions will bubble to the main PyRefine process.

3. Formatting pipeline — `commands/clean/format.py`

   - Discovers targets by traversing the filesystem and honoring `.gitignore` rules and a list of `IGNORED_DIR_NAMES` (for example `.venv`, `.vscode`, `__pycache__`).
   - Respects a dynamic ignore list via the `PYREFINE_IGNORE_DIRS` environment variable which is parsed to extend the static `IGNORED_DIR_NAMES`.
   - The pipeline is ordered to avoid churn and minimize conflicting changes: Autoflake → Isort → Autopep8 → Black → Flake8.
   - The script finds tool executables in possible virtual environments (e.g., `VIRTUAL_ENV`, `.venv`, or bundled `env`) or falls back to system PATH via `shutil.which`.

   - The formatter sequence treats some tools as optional and some as required: `autoflake` and `autopep8` are optional (the pipeline skips them when not found), while `isort`, `black`, and `flake8` are treated as required and will raise an error if missing.
   - `find_executable()` looks for candidate binaries in the active `VIRTUAL_ENV`, local project environment directories (for example `env`, `.venv`), and the PyRefine repo-level env folders; if none are found, it falls back to the global PATH via `shutil.which()`.
   - Targets are chunked to avoid huge command lines; the tool supports being run as embedded code (frozen) or via `subprocess`.

   ### Chunking, batching and resource control

   - For performance and to avoid overly long command-lines, targets are processed in batches configured by `FORMAT_BATCH_SIZE` which defaults to `200`.
   - The Black line-length is a constant (79) and is enforced by both isort arguments and Black settings.

   ### Flake8 config provisioning

   - The clean workflow will try to copy a `.flake8` template from the resource root into the project root if one doesn't exist; if the bundled template doesn't exist, a minimal default `.flake8` is synthesized with sensible defaults and an exclude list that mirrors the tool ignores.

4. --setup (VS Code + environments)

   - Implemented by `commands/setup/setup_manager.py`.
   - Writes/merges `.vscode/settings.json` and `extensions.json` by using payload helpers from `shared/common_vscode.py`.
   - Provisions a pip virtual environment in `.venv` (creates it and uses it to install `requirements.txt` if present).
   - Optionally provisions a UV environment for tools requiring the `uv` CLI (if found).
   - Attempts to install the Pylance extension using the available `code` CLI; prints advice if automatic install fails.

   ### Setup specifics

   - `create_pip_environment()` uses a base Python found via `_get_base_python()` which will use `sys.executable` when running as scripts but will fall back to a system `python`/`python3` when running frozen.
   - VS Code extension installation prefers the `code`/`code-insiders` CLI if available; otherwise `shared/common_vscode.pylance_installed()` tries to detect installed extensions from user directories as a fallback.

5. --test-coverage

   - Implemented in `commands/test_coverage/coverage_runner.py` (entrypoint: `coverage_runner.run_for_projects`).
   - Loads `.gitignore` and verifies the target is a project root not excluded by `.gitignore`.
   - Runs pytest with coverage flags and writes reports under `pyrefine_artifacts/<project>/coverage`.

   ### CLI-level validation for coverage runs

   - At the CLI layer, `handle_test_coverage()` first attempts to load the gitignore spec and then resolves the target path. If the default sentinel value is used, the root is chosen; otherwise an absolute path must be provided. The CLI ensures the target exists, is a directory, and is not ignored by the `.gitignore` rules before invoking the `coverage_runner`.

   ### Coverage runner specifics

   - Coverage uses a system Python interpreter when PyRefine is packaged (frozen); the runner searches for `python.exe`, `python`, or `python3` on the PATH, else it errors.
   - The discovery of projects is directory-based: child directories of the provided root are considered projects when they contain any known project markers such as `tests/`, `src/` or `requirements.txt`.
   - Coverage reports include XML, HTML, a `summary.txt` (stdout of `coverage report`), and a copy of `.coverage` if present in the project directory.

6. --update

   - Implemented by `commands/update/update_manager.py`.
   - Fetches a manifest URL (defaults to `DEFAULT_MANIFEST_URL` or `PYREFINE_UPDATE_URL` env var), determines if an update is available, and if so, applies new artifacts (e.g., replacing the exe on Windows or writing updated resources into the bundle when possible).

   ### Auto-update behavior and manifest format

   - The updater requires a JSON manifest with a top-level `version` string and an `artifacts` mapping. Example keys include `windows`, `linux`, `darwin`, or `default` entries. Each artifact should include `url` and `checksum` (e.g., "sha256:<hex>").
   - Only the packaged (frozen) PyRefine applies updates; attempting to run `--update` in source mode will fail with a clear error.
   - The updater downloads the artifact, validates checksum (sha256) and either schedules a safe replacement script on Windows or stores the updated binary with a `.updated` suffix on other platforms for manual replacement.
   - There are clear error messages for manifest parsing, network issues, checksum mismatch, and unsupported artifact formats: these raise `UpdateError` and return a non-zero exit.

7. build_exe (packaging)

   - Present in `commands/build_exe/build_exe.py`. This command is used to create a distribution artifact such as `pyrefine.exe` via PyInstaller; it may require building and tests to be run in the CI pipeline.

   ### Build and packaging notes

   - `build_exe` uses PyInstaller and registers required data files (for example `format.py` and `.flake8`) with the `--add-data` option so these resources remain available when the executable runs. They are added under `PyRefine/*` inside the single-file bundle.
   - Packaging uses `--onefile` and `--clean` to produce a standalone artifact suitable for distribution.

## Shared helpers and utilities

- `shared/gitignore_utils.py`: centralizes `.gitignore` parsing and provides `is_gitignored()` which is relied on by many commands.
- `shared/common_vscode.py`: builds VS Code `settings.json` and `extensions.json` payloads and provides merge helpers to keep user settings intact.

### Shared utility notes

- `shared/gitignore_utils.py` exposes `load_gitignore_spec()` which will error if a `.gitignore` is not found (raising `GitignoreError`) to ensure the rest of the toolchain respects project-specific excludes.
- `shared/common_vscode.py` contains helpers to build the VS Code payload and detect whether the Pylance extension is present; it also provides `merge_run_on_save()` to keep any existing user settings while ensuring a `run on save` command references the correct formatter script path.

- `merge_run_on_save()` intends to avoid adding duplicate commands; it inspects `emeraldwalk.runonsave.commands` and appends a single `cmd` matching the desired formatter reference only if it isn't already present.

## Developer notes and extension points

- Adding a new command: follow the existing manager convention: a top-level manager method (for example `run_clean`) is invoked by `cli/pyrefine.py` and should accept a `project_root` and arguments. Keep CLI flags in `cli/pyrefine.py` for consistent UX.
- If a new tool participates in the formatting pipeline, update `commands/clean/format.py` in the correct sequence, and ensure tests are added in `commands/test_coverage/` to exercise the changes.
- When running tests that need isolation, create temporary project directories and use `pyrefine`'s scaffold to create a reproducible layout.

### Conventions and error handling

- Most managers follow a pattern of validating inputs, loading a `.gitignore` spec, and raising domain-specific exceptions: `GitignoreError`, `CoverageError`, and `UpdateError`. These are caught by the CLI entry in `cli/pyrefine.py` or by the manager entrypoints and surface user-friendly messages with non-zero exit codes.
- The CLI parser enforces that only one action is specified at a time. If no action is provided, the default action is `--clean .`. The `--update` action has no default and requires `--update` to be explicitly passed.
- Formatting runs operate with the project root as `cwd` and are executed in reasonable sized batches to avoid filesystem or OS limits on command invocation size.

## Troubleshooting and common questions

- If a command fails because `python` is not found while frozen, ensure a system Python is installed and available in PATH.
- If `--clean` doesn't change files, confirm your virtual environment has the required tooling installed (autoflake, isort, black, flake8) or `requirements.txt` is created and installed by `--setup`.
- If `.gitignore` is missing, many commands will abort with an error instructing you to scaffold one: run `pyrefine --create`.

### Exit codes and errors to watch for

- `GitignoreError` (exit `1`) — produced by operations that require a `.gitignore` (format, coverage); the CLI prints an explanatory message directing users to `--create`.
- `CoverageError` (exit `1`) — when coverage runner cannot find a suitable Python interpreter or no projects are discovered.
- `UpdateError` (exit `1`) — encapsulates all manifest, download, checksum, and apply-time failures in the updater.
- `subprocess.CalledProcessError` — formatting and coverage commands may return the original command's exit code, which the CLI surfaces.

## Additional references

- CLI entry: `cli/pyrefine.py`
- Cleaner: `commands/clean/clean_manager.py` and `commands/clean/format.py`
- Scaffold: `commands/create/scaffold_manager.py`
- Setup: `commands/setup/setup_manager.py`
- Shared: `shared/gitignore_utils.py`, `shared/common_vscode.py`

---

This file aims to help maintainers and contributors understand the runtime flow and where to extend PyRefine behavior. Please open a PR to add more details specific to any command if you make changes that affect the runtime flow.
