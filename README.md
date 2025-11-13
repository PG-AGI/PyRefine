# PyRefine – Python Project Automation Toolkit

PyRefine bundles every repetitive task you need when adopting or creating a Python codebase—scaffolding, formatting, VS Code setup, environment creation (pip + UV), coverage, Docker, and a self-updating CLI/EXE.

---

## Feature Highlights

- **Standard project scaffold**: ensures `src/`, `tests/`, `configs/`, `scripts/`, `__init__.py` files, and a curated `.flake8`.
- **Docker-ready**: auto-generates a `Dockerfile` (python:3.11-slim base, pip installs, port exposure, configurable entrypoint) when missing.
- **VS Code integration**: merges `.vscode/settings.json` and `.vscode/extensions.json`, adds run-on-save formatting, and auto-installs Pylance via the VS Code CLI.
- **Dual environments**: provisions `.venv` (pip) and `.uv-env` (UV) with dependencies from `requirements.txt` and `uv.lock`.
- **Formatting pipeline**: `--clean` runs Autoflake → Isort → Autopep8 → Black → Flake8 and prunes cache folders.
- **Coverage automation**: `--test-coverage` executes pytest with coverage, storing reports per project under `pyrefine_artifacts/<project>/coverage/`.
- **Self-updating binary**: `pyrefine.exe --update` downloads new releases via the manifest system.
- **Cross-platform support**: works on Windows (exe or Python), macOS, and Ubuntu with identical commands.

---

## Quick Start (Existing Project)

1. **Clone or add PyRefine** to your project root:
   ```bash
   git clone https://github.com/PG-AGI/PyRefine.git
   ```
2. **Run the setup command** (see OS instructions below). This:
   - builds the project scaffold + Dockerfile,
   - merges VS Code settings/extensions (with Pylance auto-installed),
   - creates `.venv` + `.uv-env` and installs dependencies.
3. **Use the CLI** (`--clean`, `--create`, `--test-coverage`, etc.) to keep the project sanitized.

---

## Running PyRefine per Platform

**Windows**
- Python: `python PyRefine/tools/pyrefine.py --setup`
- EXE: `pyrefine.exe --setup` (use `--clean .`, `--test-coverage`, etc. the same way)

**macOS**
- Python: `python3 PyRefine/tools/pyrefine.py --setup`
- Binary: `chmod +x pyrefine-macos && ./pyrefine-macos --setup`

**Ubuntu / Linux**
- Python: `python3 PyRefine/tools/pyrefine.py --setup`
- Binary: `chmod +x pyrefine-linux && ./pyrefine-linux --setup`

Use the same flag set (`--clean`, `--create`, `--test-coverage`, `--project-root /path`) regardless of platform or binary/Python mode.

Use `--project-root /absolute/path` with any command when invoking PyRefine from outside the project.

---

## CLI Commands

| Command | Purpose |
| ------- | ------- |
| `--create` | Generate the standard scaffold (folders, `__init__.py`, `.flake8`). |
| `--clean [PATH]` | Format the entire project (`.` default), a folder, or a single `.py` file with the full formatter pipeline. |
| `--setup` | Runs the consolidated setup manager: scaffold/Dockerfile, VS Code config, Pylance install, `.venv` + `.uv-env` provisioning, dependency installs. |
| `--test-coverage [PATH]` | Run pytest+coverage either for the provided project path or every project under the root. Reports land in `pyrefine_artifacts/<project>/coverage/`. |
| `--update [--manifest-url URL]` | For `pyrefine.exe` users, download and apply the newest release via the manifest. |

*No flag* defaults to `--clean .`, ensuring the repo stays formatted.

---

## Environment Details

- **pip (`.venv`)**: Created with `python -m venv .venv`. PyRefine auto-upgrades pip and installs `requirements.txt`.
- **UV (`.uv-env`)**: Created with `uv venv .uv-env`. Dependencies sync from `uv.lock` (or fall back to `requirements.txt`). Install UV once (`pip install uv`) so PyRefine can use it automatically.
- **Switching**:
  ```bash
  # Pip
  source .venv/bin/activate          # macOS/Linux
  .\.venv\Scripts\activate           # Windows

  # UV
  uv run python src/main.py          # Preferred (no activation)
  source .uv-env/bin/activate        # Manual activation (macOS/Linux)
  .\.uv-env\Scripts\activate         # Windows
  ```

---

## Docker & Scaffold

- PyRefine drops a default `Dockerfile` when missing:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
  RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
  COPY requirements.txt .
  RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE ${PORT}
  CMD ["python", "src/main.py"]  # customize as needed
  ```
- Scaffold ensures the canonical layout (`src/`, `tests/`, `configs/`, `scripts/`, `.flake8`, `.vscode/`).
- Customize the entry point, ports, or dependencies as your service evolves.

---

## PyTest Coverage Workflow

1. Run coverage for all projects under the root:
   ```bash
   python PyRefine/tools/pyrefine.py --test-coverage
   ```
2. Or target a specific project:
   ```bash
   python PyRefine/tools/pyrefine.py --test-coverage path/to/project_B
   ```
3. Output structure:
   ```
   pyrefine_artifacts/
     project_A/
       coverage/
         coverage.xml
         summary.txt
         coverage_html_report/
         .coverage
   ```
4. Open `coverage_html_report/index.html` in a browser to see per-line details, or parse `coverage.xml` for CI dashboards.

---

## VS Code & Pylance Notes

- `.vscode/settings.json` sets format-on-save, Black, isort, Flake8, and run-on-save commands that call PyRefine’s formatter.
- `.vscode/extensions.json` recommends the Python, Black, isort, Flake8, Run On Save extensions, and now also ensures Pylance is installed automatically (falls back to a reminder if VS Code CLI is unavailable).
- Removing `.vscode/` is safe—rerun `--setup` to recreate it.

---

## Building the Standalone Executable (Optional)

```bash
pip install pyinstaller
python PyRefine/tools/build_exe.py
```

Artifacts land in `dist/`:
- `pyrefine-windows.exe`
- `pyrefine-linux`
- `pyrefine-macos`

CI builds (see `.github/workflows/build-binaries.yml`) rename and attach these per OS and publish a `manifest.json` for auto-update.

---

## Repository Reference

| File / Folder | Description |
| ------------- | ----------- |
| `tools/pyrefine.py` | Main CLI entrypoint. |
| `tools/setup_manager.py` | Implements `--setup` (scaffold, Dockerfile, VS Code, envs, Pylance). |
| `tools/coverage_runner.py` | Discovers projects and runs pytest+coverage. |
| `tools/format.py` | Formatting/linting pipeline used by `--clean` and VS Code. |
| `tools/bootstrap.py` | Legacy guided setup script (kept for reference). |
| `tools/setup_workspace.py` | Standalone workspace writer (superseded by `--setup`). |
| `tools/build_exe.py` | PyInstaller helper. |
| `.flake8`, `.vscode/` | Shared lint/IDE defaults. |
| `pyrefine_artifacts/` | Generated coverage and future analysis outputs (ignored by Git). |
| `requirements.txt`, `uv.lock` | Dependency manifests for pip and UV respectively. |

---

## Next Steps

- Commit the generated `.vscode/`, `Dockerfile`, and scaffolding so teammates inherit the standardized setup.
- Add CI jobs to run `python PyRefine/tools/pyrefine.py --clean . --project-root ... --lint-only` or `--test-coverage` to enforce quality gates.
- Customize the Dockerfile entrypoint / exposed ports to match your API (FastAPI, Flask, etc.).
- Publish releases using the provided GitHub Actions workflow and let users update via `pyrefine.exe --update`.

Run PyRefine whenever you onboard a repository to keep your Python projects clean, reproducible, and deployment-ready.
