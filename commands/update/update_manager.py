#!/usr/bin/env python3
from __future__ import annotations

"""
Module Purpose:
    Implements the self-update flow for packaged PyRefine binaries by
    consuming a manifest, downloading artifacts, and scheduling replacements.

Key Components:
    - handle_update: Entry point that validates versions, selects artifacts,
     and orchestrates work.
    - download_release_binary: Streams and verifies the new binary against
     checksums.
    - apply_update_binary / schedule_windows_replace: Swap executables safely
     on each platform.

Project Contribution:
    Keeps distributed PyRefine binaries up to date without manual downloads,
    ensuring users always run the latest automation features with minimal
     friction.

"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DOWNLOAD_BUFFER_SIZE = 64 * 1024
CHECKSUM_ALGORITHM = "sha256"
WINDOWS = os.name == "nt"
MACOS = sys.platform == "darwin"
LINUX = sys.platform.startswith("linux")


class UpdateError(RuntimeError):
    """Raised when the update routine fails."""


def handle_update(
    args: argparse.Namespace,
    current_version: str,
    default_manifest_url: str | None,
) -> None:
    """
    Main entry point for the --update flag.
    """
    try:
        manifest_url = resolve_manifest_url(
            args.manifest_url, default_manifest_url
        )
        manifest = fetch_manifest(manifest_url)
        manifest_version = manifest.get("version")
        if not isinstance(manifest_version, str):
            raise UpdateError("Manifest missing a string 'version' field.")

        if not is_newer_version(current_version, manifest_version):
            print(f"PyRefine is up to date (version {current_version}).")
            return

        if not getattr(sys, "frozen", False):
            raise UpdateError(
                "The auto-update command only applies to the packaged"
                " executable. Re-run once you are using pyrefine.exe."
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
            "Update applied. Please relaunch PyRefine after the"
            " helper finishes."
        )
    except UpdateError as exc:
        print(f"[pyrefine] Update failed: {exc}", file=sys.stderr)
        sys.exit(1)


def resolve_manifest_url(
    manifest_override: str | None, default_manifest_url: str | None
) -> str:
    if manifest_override:
        return manifest_override
    if default_manifest_url:
        return default_manifest_url
    raise UpdateError(
        "No update manifest URL available. Provide --manifest-url or set "
        "PYREFINE_UPDATE_URL."
    )


def parse_version(value: str) -> tuple[int, ...]:
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
    return parse_version(candidate) > parse_version(current)


def fetch_manifest(url: str) -> dict[str, object]:
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
    downloads = manifest.get("artifacts")
    if not isinstance(downloads, dict):
        raise UpdateError("Manifest missing an 'artifacts' mapping.")

    preferred_keys: list[str] = []
    if WINDOWS:
        preferred_keys.extend(["windows", "win64", "win32"])
    elif MACOS:
        preferred_keys.extend(["macos", "darwin"])
    elif LINUX:
        preferred_keys.extend(["linux", sys.platform])
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


def download_release_binary(url: str, checksum: str) -> Path:
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
        raise UpdateError(
            f"Failed to download update artifact: {exc}"
        ) from exc

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
        "if errorlevel 1 (\r\n"
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


def apply_update_binary(
    current_executable: Path, downloaded_path: Path
) -> None:
    destination_dir = current_executable.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    if WINDOWS:
        staged_path = destination_dir / (current_executable.name + ".new")
        shutil.move(str(downloaded_path), staged_path)
        schedule_windows_replace(current_executable, staged_path)
        print("Update scheduled. The executable will be replaced shortly.")
        return

    staged_path = destination_dir / (current_executable.name + ".new")
    shutil.move(str(downloaded_path), staged_path)
    staged_path.chmod(0o755)

    backup_path: Path | None = destination_dir / (current_executable.name + ".bak")
    try:
        if backup_path.exists():
            backup_path.unlink()
        current_executable.replace(backup_path)
    except OSError:
        backup_path = None

    staged_path.replace(current_executable)
    print("Update applied.")
    if backup_path is not None:
        print(f"A backup of the previous binary was saved to {backup_path}.")
    print("Relaunch PyRefine to use the updated executable.")


def _normalise_checksum(expected: str) -> tuple[str, str]:
    if ":" in expected:
        algorithm, value = expected.split(":", 1)
        return algorithm.strip().lower(), value.strip().lower()
    return CHECKSUM_ALGORITHM, expected.strip().lower()
