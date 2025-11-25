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
- Add `--project-root C:\path\to\project` (or `/path/to/project`) if the binary isn't placed directly in the repo root.

## 2. Commands

| Command | Summary |
| ------- | ------- |
| `pyrefine.exe` | Runs clean workflow on current repo |
| `pyrefine --clean <path>` | Formats target using full formatter stack |
| `pyrefine --create` | Bootstraps standard project structure |
| `pyrefine --setup` | Configures VS Code and environments |
| `pyrefine --test-coverage [path]` | Generates pytest coverage reports per project |
| `pyrefine --update [--manifest-url URL]` | Downloads and installs newest binary release |

## 3. How PyRefine works (detailed)

If you want the complete working details for developers and maintainers, see [docs/WORKING.md](./WORKING.md) which describes the architecture, each command's behavior, and implementation notes.
