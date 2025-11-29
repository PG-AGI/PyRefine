# PyRefine

## 1. Download the latest binary

The prebuilt executables live on the GitHub Releases page. Replace `v1.0` with the newest tag when needed.

**Windows**
```powershell
curl -L -o pyrefine.exe https://github.com/PG-AGI/PyRefine/releases/download/v1.0/pyrefine.exe
pyrefine.exe --version
```

**macOS**
```bash
curl -L -o pyrefine-macos https://github.com/PG-AGI/PyRefine/releases/download/v1.0/pyrefine-macos
chmod +x pyrefine-macos
./pyrefine-macos --version
```

**Ubuntu/Linux**
```bash
curl -L -o pyrefine-linux https://github.com/PG-AGI/PyRefine/releases/download/v1.0/pyrefine-linux
chmod +x pyrefine-linux
./pyrefine-linux --version
```

If the executable is stored outside the project root, pass `--project-root PATH_TO_REPO` on every command (e.g. `pyrefine.exe --project-root C:\code\MyRepo --setup`).

## 2. Standard workflows

### A. Bootstrap a brand-new project
1. Download the appropriate binary and place it in your empty project directory.
2. Run the initializers:
   ```bash
   pyrefine --create
   pyrefine --setup
   pyrefine --clean .
   ```
   (On Windows use `pyrefine.exe`.)
3. Commit the generated structure and begin development.

### B. Clean up or adopt an existing project
1. Drop the binary into the repository (or reference it via `--project-root`).
2. Apply the cleanup workflow:
   ```bash
   pyrefine --setup
   pyrefine --clean .
   pyrefine --test-coverage   # optional validation
   ```

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
