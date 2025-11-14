#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path
from textwrap import dedent

PROJECT_DIRECTORIES: tuple[str, ...] = (
    "src",
    "tests",
    "configs",
    "scripts",
    "utils",
    "services",
)

FILE_TEMPLATES: dict[str, str] = {
    "src/__init__.py": "",
    "src/main.py": dedent(
        """\
        import os

        from fastapi import FastAPI

        app = FastAPI(title="PyRefine Backend")


        @app.get("/health", tags=["health"])
        def health_check() -> dict[str, str]:
            return {"status": "ok"}


        if __name__ == "__main__":
            import uvicorn

            uvicorn.run(
                "src.main:app",
                host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")),
                reload=True,
            )
        """
    ),
    "tests/__init__.py": "",
    "tests/test_main.py": dedent(
        """\
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)


        def test_health_check() -> None:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        """
    ),
    "configs/__init__.py": "",
    "configs/settings.py": dedent(
        """\
        from pydantic import BaseSettings, Field


        class Settings(BaseSettings):
            app_name: str = Field("PyRefine Backend", env="APP_NAME")
            port: int = Field(8000, env="PORT")

            class Config:
                env_file = ".env"


        settings = Settings()
        """
    ),
    "scripts/.gitkeep": "",
    "utils/__init__.py": "",
    "utils/example.py": dedent(
        """\
        def slugify(value: str) -> str:
            \"\"\"Very small helper to normalize identifiers.\"\"\"
            return value.strip().lower().replace(" ", "-")
        """
    ),
    "services/__init__.py": "",
    "services/example_service.py": dedent(
        """\
        from utils.example import slugify


        def build_identifier(name: str) -> str:
            return slugify(name)
        """
    ),
    ".env.example": dedent(
        """\
        APP_NAME=PyRefine Backend
        PORT=8000
        """
    ),
    "README.md": dedent(
        """\
        # PyRefine Backend Template

        This project was bootstrapped by PyRefine. Update this README with
        service-specific documentation after you customize the scaffold.

        ## Getting Started
        1. Activate either `.venv` (pip) or `.uv-env` (UV).
        2. Run `python src/main.py` (development) or `uvicorn src.main:app`.
        3. Update `requirements.txt`, `Dockerfile`, and configs as needed.
        """
    ),
}

REQUIREMENTS_TEMPLATE = dedent(
    """\
    fastapi==0.111.0
    uvicorn[standard]==0.30.0
    pydantic==2.8.0
    python-dotenv==1.0.1
    pytest==8.2.2
    coverage==7.6.1
    httpx==0.27.0
    """
)

GITIGNORE_TEMPLATE = dedent(
    """\
    # Python
    __pycache__/
    *.py[cod]
    *.so
    *.egg-info/
    .python-version

    # Environments
    .venv/
    .uv-env/
    env/
    venv/

    # Tooling
    .mypy_cache/
    .pytest_cache/
    .ruff_cache/
    .ipynb_checkpoints/
    .DS_Store

    # PyRefine artifacts
    pyrefine_artifacts/

    # VS Code
    .vscode/

    # Logs
    *.log
    """
)

DOCKERFILE_TEMPLATE = dedent(
    """\
    # syntax=docker/dockerfile:1
    FROM python:3.11-slim

    WORKDIR /app

    ENV PYTHONDONTWRITEBYTECODE=1 \\
        PYTHONUNBUFFERED=1 \\
        PORT=8000

    RUN apt-get update \\
        && apt-get install -y --no-install-recommends build-essential \\
        && rm -rf /var/lib/apt/lists/*

    COPY requirements.txt .
    RUN pip install --upgrade pip \\
        && pip install --no-cache-dir -r requirements.txt

    COPY . .

    EXPOSE ${PORT}

    CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
    """
)


def ensure_scaffold(project_root: Path, resource_root: Path) -> None:
    created_dirs = _ensure_directories(project_root)
    created_files = _ensure_template_files(project_root)
    created_extra = []
    if _write_file_if_missing(project_root, ".gitignore", GITIGNORE_TEMPLATE):
        created_extra.append(".gitignore")
    if _write_file_if_missing(
        project_root, "requirements.txt", REQUIREMENTS_TEMPLATE
    ):
        created_extra.append("requirements.txt")
    if _write_file_if_missing(project_root, "Dockerfile", DOCKERFILE_TEMPLATE):
        created_extra.append("Dockerfile")
    if ensure_flake8(project_root, resource_root):
        created_extra.append(".flake8")
    if created_dirs:
        print(
            f"[scaffold] Ensured directories: {', '.join(sorted(created_dirs))}"
        )
    if created_files or created_extra:
        combined = created_files + created_extra
        print(f"[scaffold] Created files: {', '.join(sorted(combined))}")


def ensure_flake8(project_root: Path, resource_root: Path) -> bool:
    template = resource_root / ".flake8"
    destination = project_root / ".flake8"
    if destination.exists():
        return False
    if template.exists():
        shutil.copy2(template, destination)
        return True
    default = dedent(
        """\
        [flake8]
        max-line-length = 79
        exclude = .venv,.uv-env,build,dist,pyrefine_artifacts
        """
    )
    destination.write_text(default, encoding="utf-8")
    return True


def _ensure_directories(project_root: Path) -> list[str]:
    created: list[str] = []
    for directory in PROJECT_DIRECTORIES:
        path = project_root / directory
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created


def _ensure_template_files(project_root: Path) -> list[str]:
    created: list[str] = []
    for relative_path, contents in FILE_TEMPLATES.items():
        if _write_file_if_missing(project_root, relative_path, contents):
            created.append(relative_path)
    return created


def _write_file_if_missing(
    project_root: Path,
    relative_path: str,
    contents: str,
    create_empty: bool = False,
) -> bool:
    target = project_root / relative_path
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if not contents and not create_empty:
        target.touch()
    else:
        target.write_text(contents, encoding="utf-8")
    return True
