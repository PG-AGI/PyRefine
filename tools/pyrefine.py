#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

APP_VERSION = "1.1.0"
# Optional default manifest URL. Override via --manifest-url or the
# PYREFINE_UPDATE_URL environment variable.
DEFAULT_MANIFEST_URL = os.environ.get("PYREFINE_UPDATE_URL")
DOWNLOAD_BUFFER_SIZE = 64 * 1024
CHECKSUM_ALGORITHM = "sha256"
WINDOWS = os.name == "nt"


def get_resource_root() -> Path:
    """
    Locate the PyRefine resource directory for both source and bundled runs.
    """
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidate = Path(bundle_dir) / "PyRefine"
        if candidate.exists():
            return candidate
        return Path(bundle_dir)
    return Path(__file__).resolve().parents[1]


RESOURCE_ROOT = get_resource_root()
FORMAT_SCRIPT = RESOURCE_ROOT / "tools" / "format.py"
FLAKE8_TEMPLATE = RESOURCE_ROOT / ".flake8"

PYLANCE_EXTENSION_ID = "ms-python.vscode-pylance"


def pylance_installed() -> bool:
    """
    Check whether the VS Code Pylance extension is available.

    First tries the VS Code CLI (`code --list-extensions`), then falls back to
    scanning common extension directories.
    """
    for cmd in ("code", "code-insiders"):
        exe = shutil.which(cmd)
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, "--list-extensions"],
                capture_output=True,
                text=True,
                check=True,
            )
            if PYLANCE_EXTENSION_ID.lower() in result.stdout.lower():
                return True
            return False
        except subprocess.CalledProcessError:
            continue

    candidates: list[Path] = []
    home = Path.home()
    candidates.append(home / ".vscode" / "extensions")
    candidates.append(home / ".vscode-insiders" / "extensions")
    user_profile = Path(os.environ.get("USERPROFILE", home))
    candidates.append(user_profile / ".vscode" / "extensions")
    for base in candidates:
        if base.exists():
            for child in base.iterdir():
                if child.is_dir() and child.name.startswith(PYLANCE_EXTENSION_ID):
                    return True
    return False


def notify_pylance_missing() -> None:
    """
    Emit a console notice and, on Windows, a popup encouraging Pylance install.
    """
    message = (
        "The Pylance extension (ms-python.vscode-pylance) was not detected.\n"
        "Installing it is highly recommended for richer IntelliSense, smarter autocompletion, and an improved Python development experience.\n\n"
        "Marketplace link:\n"
        "https://marketplace.visualstudio.com/items?itemName="
        "ms-python.vscode-pylance"
    )
    print(f"NOTICE: {message}")
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
                None,
                message,
                "PyRefine Recommendation",
                0x00000040,
            )
        except Exception:
            pass


class UpdateError(RuntimeError):
    """Raised when the update routine fails."""


def parse_version(value: str) -> tuple[int, ...]:
    """
    Convert a semantic-version-like string into a tuple of integers.

    Non-integer segments stop the parsing (e.g. 1.2.0-rc1 -> (1, 2, 0)).
    """
    core = value.split("-", 1)[0].strip()
    parts: list[int] = []
    for segment in core.split("."):
        segment = segment.strip()
        if not segment:
            continue
        try:
            parts.append(int(segment))
        except ValueError:
            break
    return tuple(parts)


def is_newer_version(current: str, candidate: str) -> bool:
    """
    True when ``candidate`` represents a newer semantic version than ``current``.
    """
    return parse_version(candidate) > parse_version(current)


def resolve_manifest_url(manifest_override: str | None) -> str:
    """
    Decide which update manifest URL to use.
    """
    if manifest_override:
        return manifest_override
    if DEFAULT_MANIFEST_URL:
        return DEFAULT_MANIFEST_URL
    raise UpdateError(
        "No update manifest URL available. Provide --manifest-url or set "
        "PYREFINE_UPDATE_URL."
    )


def fetch_manifest(url: str) -> dict[str, object]:
    """
    Download and parse the remote manifest JSON document.
    """
    try:
        with urllib.request.urlopen(url) as response:  # nosec: B310
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise UpdateError(f"Unable to download manifest: {exc}") from exc

    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UpdateError("The update manifest is not valid JSON.") from exc

    if not isinstance(manifest, dict):
        raise UpdateError("The update manifest must be a JSON object.")
    return manifest


def select_artifact(manifest: dict[str, object]) -> tuple[str, str]:
    """
    Pick the appropriate download entry for the current platform.
    """
    downloads = manifest.get("artifacts")
    if not isinstance(downloads, dict):
        raise UpdateError("Manifest missing an 'artifacts' mapping.")

    preferred_keys: list[str] = []
    if WINDOWS:
        preferred_keys.extend(["windows", "win64", "win32"])
    else:
        preferred_keys.extend([sys.platform, "linux", "darwin"])
    preferred_keys.append("default")

    for key in preferred_keys:
        entry = downloads.get(key)
        if isinstance(entry, dict):
            url = entry.get("url")
            checksum = entry.get("checksum")
            if isinstance(url, str) and isinstance(checksum, str):
                return url, checksum
    raise UpdateError("Manifest does not define a compatible download entry.")


def _normalise_checksum(expected: str) -> tuple[str, str]:
    if ":" in expected:
        algorithm, value = expected.split(":", 1)
        return algorithm.strip().lower(), value.strip().lower()
    return CHECKSUM_ALGORITHM, expected.strip().lower()


def download_release_binary(url: str, checksum: str) -> Path:
    """
    Download the release artifact and verify its checksum.
    """
    hasher = hashlib.new(CHECKSUM_ALGORITHM)
    try:
        with urllib.request.urlopen(url) as response:  # nosec: B310
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                while True:
                    chunk = response.read(DOWNLOAD_BUFFER_SIZE)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    hasher.update(chunk)
                temp_path = Path(temp_file.name)
    except urllib.error.URLError as exc:
        raise UpdateError(f"Failed to download update artifact: {exc}") from exc

    expected_algo, expected_digest = _normalise_checksum(checksum)
    if expected_algo != CHECKSUM_ALGORITHM:
        raise UpdateError(
            f"Unsupported checksum algorithm '{expected_algo}'. "
            f"Expected {CHECKSUM_ALGORITHM}."
        )

    actual_digest = hasher.hexdigest().lower()
    if actual_digest != expected_digest:
        temp_path.unlink(missing_ok=True)
        raise UpdateError(
            "Checksum mismatch for downloaded artifact: "
            f"expected {expected_digest}, got {actual_digest}."
        )

    return temp_path


def schedule_windows_replace(target: Path, staged_binary: Path) -> None:
    """
    Spawn a detached helper script that replaces the running executable.
    """
    helper_dir = staged_binary.parent
    script_path = helper_dir / "pyrefine_update.cmd"
    script_contents = (
        "@echo off\r\n"
        "setlocal\r\n"
        f'set "TARGET={target}"\r\n'
        f'set "SOURCE={staged_binary}"\r\n'
        f'set "BACKUP={target.with_suffix(target.suffix + ".bak")}"\r\n'
        ":retry\r\n"
        'del "%TARGET%" >nul 2>&1\r\n'
        'if exist "%TARGET%" (\r\n'
        "    timeout /T 1 /NOBREAK >nul\r\n"
        "    goto retry\r\n"
        ")\r\n"
        'move /Y "%SOURCE%" "%TARGET%" >nul 2>&1\r\n'
        'if errorlevel 1 (\r\n'
        "    timeout /T 1 /NOBREAK >nul\r\n"
        "    goto retry\r\n"
        ")\r\n"
        'del "%BACKUP%" >nul 2>&1\r\n'
        'del "%~f0"\r\n'
    )
    script_path.write_text(script_contents, encoding="utf-8")

    creation_flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):  # pragma: no cover
        creation_flags = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(  # noqa: S603,S607
        ["cmd.exe", "/c", str(script_path)],
        creationflags=creation_flags,
        close_fds=False,
    )


def apply_update_binary(current_executable: Path, downloaded_path: Path) -> None:
    """
    Replace the current executable with the freshly downloaded one.
    """
    destination_dir = current_executable.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    if WINDOWS:
        staged_path = destination_dir / (current_executable.name + ".new")
        shutil.move(str(downloaded_path), staged_path)
        schedule_windows_replace(current_executable, staged_path)
        print("Update scheduled. The executable will be replaced shortly.")
        return

    replacement_path = destination_dir / (current_executable.name + ".updated")
    shutil.move(str(downloaded_path), replacement_path)
    print(
        "Downloaded updated binary to "
        f"{replacement_path}. Replace the current executable manually."
    )


def handle_update(args: argparse.Namespace) -> None:
    """
    Main entry point for the --update flag.
    """
    try:
        manifest_url = resolve_manifest_url(args.manifest_url)
        manifest = fetch_manifest(manifest_url)
        manifest_version = manifest.get("version")
        if not isinstance(manifest_version, str):
            raise UpdateError("Manifest missing a string 'version' field.")

        if not is_newer_version(APP_VERSION, manifest_version):
            print(f"PyRefine is up to date (version {APP_VERSION}).")
            return

        if not getattr(sys, "frozen", False):
            raise UpdateError(
                "The auto-update command only applies to the packaged executable. "
                "Re-run once you are using pyrefine.exe."
            )

        download_url, checksum = select_artifact(manifest)
        temp_binary = download_release_binary(download_url, checksum)
        try:
            current_executable = Path(sys.executable).resolve()
            apply_update_binary(current_executable, temp_binary)
        finally:
            if temp_binary.exists():
                temp_binary.unlink(missing_ok=True)

        notes = manifest.get("release_notes")
        if isinstance(notes, str) and notes.strip():
            print("\nRelease notes:\n")
            print(notes.strip())

        print(
            "Update applied. Please relaunch PyRefine after the helper finishes."
        )
    except UpdateError as exc:
        print(f"[pyrefine] Update failed: {exc}", file=sys.stderr)
        sys.exit(1)


# Directories we treat as part of the canonical scaffold
TEMPLATE_DIRECTORIES: tuple[str, ...] = ("src", "tests", "configs", "scripts")
TEMPLATE_FILES: tuple[tuple[str, str], ...] = (
    ("src/__init__.py", ""),
    ("tests/__init__.py", ""),
    ("configs/.gitkeep", ""),
    ("scripts/.gitkeep", ""),
)

# Items to prune when cleaning a repository or folder
CLUTTER_DIR_PATTERNS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".ipynb_checkpoints",
    ".eggs",
    "*.egg-info",
    "build",
    "dist",
)
CLUTTER_FILE_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PyRefine CLI for scaffolding, cleanup, and VS Code " "setup."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Project root folder (defaults to the current "
            "working directory)."
        ),
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help=(
            "Create the standard Python project scaffold "
            "(src/, tests/, configs/, scripts/)."
        ),
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const=".",
        metavar="PATH",
        help=(
            "Format a file, folder, or the entire project (default: current "
            "project)."
        ),
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create or merge .vscode settings and extension recommendations.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Check the update manifest and replace the current executable if "
            "a newer version is available."
        ),
    )
    parser.add_argument(
        "--manifest-url",
        help=(
            "Override the update manifest URL. Defaults to the value of the "
            "PYREFINE_UPDATE_URL environment variable (if set)."
        ),
    )
    args = parser.parse_args()

    actions = sum(
        [
            1 if args.create else 0,
            1 if args.clean is not None else 0,
            1 if args.setup else 0,
            1 if args.update else 0,
        ]
    )
    if actions > 1:
        parser.error(
            "Please choose only one action at a time "
            "(--create, --clean, --setup, or --update)."
        )
    if args.update:
        return args  # no default action when explicitly updating
    if actions == 0:
        args.clean = "."
    return args


def ensure_absolute(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def ensure_flake8_config(project_root: Path) -> None:
    """Ensure the project has a .flake8 configuration file."""
    if not FLAKE8_TEMPLATE.exists():
        return
    target = project_root / ".flake8"
    if not target.exists():
        shutil.copy2(FLAKE8_TEMPLATE, target)


def create_scaffold(project_root: Path) -> None:
    for directory in TEMPLATE_DIRECTORIES:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    for relative_path, contents in TEMPLATE_FILES:
        file_path = project_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(contents, encoding="utf-8")
    ensure_flake8_config(project_root)
    print(f"Created scaffold under {project_root}")


def remove_clutter(path: Path) -> None:
    removed_any = False
    for pattern in CLUTTER_DIR_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                removed_any = True
    for pattern in CLUTTER_FILE_PATTERNS:
        for item in path.glob(f"**/{pattern}"):
            if item.is_file():
                try:
                    item.unlink()
                    removed_any = True
                except OSError:
                    continue
    if removed_any:
        print("Removed cache/build artefacts before formatting.")


def run_formatter(target: str, project_root: Path) -> None:
    if not FORMAT_SCRIPT.exists():
        print("Formatter script missing from resources; skipping formatting.")
        return

    env = os.environ.copy()
    env["PYREFINE_PROJECT_ROOT"] = str(project_root)

    if hasattr(sys, "_MEIPASS"):
        previous_environ = os.environ.copy()
        previous_argv = sys.argv[:]
        try:
            os.environ.update(env)
            sys.argv = [str(FORMAT_SCRIPT), target]
            print(f"Running formatter via embedded script: {FORMAT_SCRIPT}")
            runpy.run_path(str(FORMAT_SCRIPT), run_name="__main__")
        finally:
            os.environ.clear()
            os.environ.update(previous_environ)
            sys.argv = previous_argv
    else:
        command = [sys.executable, str(FORMAT_SCRIPT), target]
        print(f"Running formatter: {' '.join(command)}")
        subprocess.run(command, check=False, env=env)


def clean_target(project_root: Path, target_arg: str) -> None:
    target_path = ensure_absolute(Path(target_arg), project_root)
    if target_arg == "." or target_path == project_root:
        remove_clutter(project_root)
        ensure_flake8_config(project_root)
        run_formatter("all", project_root)
        return

    if not target_path.exists():
        raise FileNotFoundError(f"Target '{target_path}' does not exist.")

    if target_path.is_file():
        if target_path.suffix != ".py":
            raise ValueError("Only Python files can be formatted directly.")
        run_formatter(str(target_path), project_root)
        return

    if target_path.is_dir():
        remove_clutter(target_path)
        ensure_flake8_config(project_root)
        run_formatter(str(target_path), project_root)
        return

    raise ValueError(f"Unsupported target: {target_path}")


def build_settings_payload(project_root: Path) -> dict[str, object]:
    command = f'python "{formatter_reference(project_root)}" "${{file}}"'
    return {
        "editor.formatOnSave": True,
        "[python]": {
            "editor.formatOnSave": True,
            "editor.defaultFormatter": "ms-python.black-formatter",
            "editor.codeActionsOnSave": {
                "source.organizeImports": True,
                "source.fixAll": True,
            },
        },
        "python.languageServer": "Jedi",
        "python.terminal.activateEnvironment": True,
        "python.formatting.provider": "black",
        "python.formatting.blackArgs": ["--line-length", "79"],
        "python.sortImports.args": [
            "--profile",
            "black",
            "--line-length",
            "79",
        ],
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.mypyEnabled": False,
        "python.linting.banditEnabled": False,
        "python.linting.ignorePatterns": [
            "env/**",
            "**/__pycache__/**",
        ],
        "emeraldwalk.runonsave": {
            "commands": [
                {
                    "match": "\\.py$",
                    "cmd": command,
                    "runIn": "terminal",
                }
            ],
        },
    }


def build_extensions_payload() -> dict[str, object]:
    return {
        "recommendations": [
            "ms-python.python",
            "ms-python.black-formatter",
            "ms-python.isort",
            "ms-python.flake8",
            "ms-python.vscode-pylance",
            "emeraldwalk.runonsave",
        ],
        "unwantedRecommendations": ["ms-python.pylint", "charliermarsh.ruff"],
    }


def formatter_reference(project_root: Path) -> str:
    try:
        relative = FORMAT_SCRIPT.relative_to(project_root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return FORMAT_SCRIPT.as_posix()


def merge_dict(
    base: dict[str, object], updates: dict[str, object]
) -> dict[str, object]:
    result = dict(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dict(result[key], value)
        elif (
            key in result
            and isinstance(result[key], list)
            and isinstance(value, list)
        ):
            combined = list(result[key])
            for item in value:
                if item not in combined:
                    combined.append(item)
            result[key] = combined
        else:
            result[key] = value
    return result


def merge_run_on_save(
    settings: dict[str, object], project_root: Path
) -> dict[str, object]:
    commands_obj = settings.get("emeraldwalk.runonsave")
    if not isinstance(commands_obj, dict):
        return settings
    commands = commands_obj.get("commands")
    if not isinstance(commands, list):
        return settings
    desired_cmd = f'python "{formatter_reference(project_root)}" "${{file}}"'
    if any(
        isinstance(entry, dict) and entry.get("cmd") == desired_cmd
        for entry in commands
    ):
        return settings
    commands.append(
        {
            "match": "\\.py$",
            "cmd": desired_cmd,
            "runIn": "terminal",
        }
    )
    return settings


def integrate_vscode(project_root: Path) -> None:
    settings_dir = project_root / ".vscode"
    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    extensions_path = settings_dir / "extensions.json"

    desired_settings = build_settings_payload(project_root)
    desired_extensions = build_extensions_payload()

    if settings_path.exists():
        try:
            existing_settings = json.loads(
                settings_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            existing_settings = {}
        merged_settings = merge_run_on_save(
            merge_dict(existing_settings, desired_settings),
            project_root,
        )
        settings_path.write_text(
            json.dumps(merged_settings, indent=4) + "\n", encoding="utf-8"
        )
        print(f"Merged settings into {settings_path}")
    else:
        settings_path.write_text(
            json.dumps(desired_settings, indent=4) + "\n", encoding="utf-8"
        )
        print(f"Created {settings_path}")

    if extensions_path.exists():
        try:
            existing_extensions = json.loads(
                extensions_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            existing_extensions = {}
        merged_extensions = merge_dict(existing_extensions, desired_extensions)
        extensions_path.write_text(
            json.dumps(merged_extensions, indent=4) + "\n", encoding="utf-8"
        )
        print(f"Merged extensions into {extensions_path}")
    else:
        extensions_path.write_text(
            json.dumps(desired_extensions, indent=4) + "\n", encoding="utf-8"
        )
        print(f"Created {extensions_path}")
    if not pylance_installed():
        print(
            "NOTICE: The Pylance extension (ms-python.vscode-pylance) was not "
            "detected. Installing it is highly recommended for richer "
            "IntelliSense, smarter autocompletion, and an improved Python development experience:\n"
            "https://marketplace.visualstudio.com/items?itemName="
            "ms-python.vscode-pylance"
        )


def handle_create(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    create_scaffold(project_root)
    print("Scaffold complete.")
    print(
        "Run 'python PyRefine/tools/pyrefine.py --setup' to configure VS Code."
    )


def handle_clean(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    try:
        target = args.clean if args.clean is not None else "."
        clean_target(project_root, target)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[pyrefine] {exc}", file=sys.stderr)
        sys.exit(1)


def handle_setup(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    integrate_vscode(project_root)


def main() -> None:
    args = parse_args()
    if args.create:
        handle_create(args)
    elif args.clean is not None:
        handle_clean(args)
    elif args.setup:
        handle_setup(args)
    elif args.update:
        handle_update(args)
    else:
        raise AssertionError(
            "Unreachable: at least one action must be specified."
        )


if __name__ == "__main__":
    main()
