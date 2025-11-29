# PyRefine

## 1. Download the latest binary

The prebuilt executables live on the GitHub Releases page. Replace `v1.2` with the newest tag when needed.

| OS | Download command | First run |
| --- | --- | --- |
| Windows | `curl -L -o pyrefine.exe https://github.com/PG-AGI/PyRefine/releases/download/v1.2/pyrefine.exe` | `pyrefine.exe --version` |
| macOS | `curl -L -o pyrefine-macos https://github.com/PG-AGI/PyRefine/releases/download/v1.2/pyrefine-macos && chmod +x pyrefine-macos` | `./pyrefine-macos --version` |
| Ubuntu/Linux | `curl -L -o pyrefine-linux https://github.com/PG-AGI/PyRefine/releases/download/v1.2/pyrefine-linux && chmod +x pyrefine-linux` | `./pyrefine-linux --version` |

If the executable is stored outside the project root, pass `--project-root PATH_TO_REPO` on every command (e.g. `pyrefine.exe --project-root C:\code\MyRepo --setup`).

## 2. Standard workflows

### A. Bootstrap a brand-new project
1. Download the appropriate binary and place it in your empty project directory.
2. Run `pyrefine --create` (e.g. `pyrefine.exe --create`) to scaffold the baseline repository layout.
3. Run `pyrefine --setup` to configure VS Code settings, extensions, and recommended tooling.
4. Run `pyrefine --clean .` to apply the formatter stack and ensure a consistent starting state.
5. Commit the generated structure and begin development.

### B. Clean up or adopt an existing project
1. Drop the binary into the repository (or reference it via `--project-root`).
2. Run `pyrefine --setup` if you want editor settings and formatter tasks added to the project.
3. Run `pyrefine --clean .` (or target a subfolder) to apply the formatting/cleanup workflow.
4. Optionally use `pyrefine --test-coverage` or other commands below for validation.

## 3. Commands

| Command | Summary |
| ------- | ------- |
| `pyrefine.exe` | Runs clean workflow on current repo |
| `pyrefine --clean <path>` | Formats target using full formatter stack |
| `pyrefine --create` | Bootstraps standard project structure |
| `pyrefine --setup` | Configures VS Code and environments |
| `pyrefine --test-coverage [path]` | Generates pytest coverage reports per project |
| `pyrefine --update [--manifest-url URL]` | Downloads and installs newest binary release |

## 4. How PyRefine works (detailed)

If you want the complete working details for developers and maintainers, see [docs/WORKING.md](./WORKING.md) which describes the architecture, each command's behavior, and implementation notes.
