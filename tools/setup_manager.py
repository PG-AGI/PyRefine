#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import common_vscode as cv
import scaffold_manager

PIP_ENV_DIRNAME = ".venv"
UV_ENV_DIRNAME = ".uv-env"


def _get_base_python() -> str:
    """Get the actual Python interpreter, even when running as a frozen executable."""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller executable - find system Python
        python_exe = shutil.which("python") or shutil.which("python3")
        if not python_exe:
            raise RuntimeError(
                "Cannot find Python interpreter. Please ensure Python is installed "
                "and available in your PATH."
            )
        return python_exe
    else:
        # Running as normal Python script
        return sys.executable


def _env_python(env_dir: Path) -> Path:
    script_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return env_dir / script_dir / executable


def _pip_command(env_dir: Path) -> list[str]:
    python_path = _env_python(env_dir)
    return [str(python_path), "-m", "pip"]


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def configure_vscode(project_root: Path, format_script: Path) -> None:
    settings_dir = project_root / ".vscode"
    settings_dir.mkdir(parents=True, exist_ok=True)

    settings_path = settings_dir / "settings.json"
    extensions_path = settings_dir / "extensions.json"

    desired_settings = cv.build_settings_payload(project_root, format_script)
    desired_extensions = cv.build_extensions_payload()

    if settings_path.exists():
        merged = cv.merge_run_on_save(
            cv.merge_dict(_read_json(settings_path), desired_settings),
            project_root,
            format_script,
        )
        _write_json(settings_path, merged)
        print(f"[setup] Merged settings into {settings_path}")
    else:
        _write_json(settings_path, desired_settings)
        print(f"[setup] Created {settings_path}")

    if extensions_path.exists():
        merged_ext = cv.merge_dict(
            _read_json(extensions_path), desired_extensions
        )
        _write_json(extensions_path, merged_ext)
        print(f"[setup] Merged extensions into {extensions_path}")
    else:
        _write_json(extensions_path, desired_extensions)
        print(f"[setup] Created {extensions_path}")

    ensure_pylance_extension()


def ensure_pylance_extension() -> None:
    if cv.pylance_installed():
        return

    if install_vscode_extension(cv.PYLANCE_EXTENSION_ID):
        print("[setup] Installed Pylance extension via VS Code CLI.")
    else:
        print(
            "[setup] Unable to install Pylance automatically. "
            "Please install 'ms-python.vscode-pylance' manually."
        )


def install_vscode_extension(extension_id: str) -> bool:
    for cmd in ("code", "code-insiders"):
        cli = shutil.which(cmd)
        if not cli:
            continue
        try:
            subprocess.run(
                [cli, "--install-extension", extension_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            continue
    return False


def create_pip_environment(project_root: Path) -> None:
    env_dir = project_root / PIP_ENV_DIRNAME
    if not env_dir.exists():
        print(f"[setup] Creating pip environment at {env_dir}")
        # Use the base Python interpreter, not the frozen executable
        base_python = _get_base_python()
        subprocess.run([base_python, "-m", "venv", str(env_dir)], check=True)
    else:
        print(f"[setup] Reusing existing pip environment at {env_dir}")

    pip_cmd = _pip_command(env_dir)
    subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)

    requirements = project_root / "requirements.txt"
    if requirements.exists():
        print(f"[setup] Installing pip dependencies from {requirements}")
        subprocess.run(
            pip_cmd + ["install", "-r", str(requirements)], check=True
        )
    else:
        print("[setup] requirements.txt not found; skipping pip installs.")


def create_uv_environment(project_root: Path) -> None:
    uv_exec = shutil.which("uv")
    if not uv_exec:
        print("[setup] 'uv' command not found. Skipping UV environment.")
        return

    env_dir = project_root / UV_ENV_DIRNAME
    if not env_dir.exists():
        print(f"[setup] Creating UV environment at {env_dir}")
        subprocess.run([uv_exec, "venv", str(env_dir)], check=True)
    else:
        print(f"[setup] Reusing existing UV environment at {env_dir}")

    env_python = _env_python(env_dir)
    requirements = project_root / "requirements.txt"

    if requirements.exists():
        print(f"[setup] Installing dependencies from {requirements}")
        subprocess.run(
            [
                uv_exec,
                "pip",
                "install",
                "--python",
                str(env_python),
                "-r",
                str(requirements),
            ],
            check=True,
        )
    else:
        print(
            "[setup] No requirements.txt found; skipping dependency installation."
        )


def run_setup(project_root: Path, resource_root: Path) -> None:
    format_script = resource_root / "tools" / "format.py"
    scaffold_manager.ensure_scaffold(project_root, resource_root)
    configure_vscode(project_root, format_script)
    create_pip_environment(project_root)
    create_uv_environment(project_root)
    print(
        "[setup] Completed scaffolding, VS Code configuration, and pip/UV "
        "environment setup."
    )
