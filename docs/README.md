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

## 3. Future Work

- Optional Docker compose templates for multi-service repos.  
- Poetry support alongside pip/UV for dependency management.  
- Dashboard-style output summarizing formatting, coverage, and scaffold status.  
