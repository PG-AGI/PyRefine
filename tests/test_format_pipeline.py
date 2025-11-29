from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from commands.clean import format as formatter


def test_run_string_fixer_modifies_long_strings(tmp_path: Path) -> None:
    sample = tmp_path / "long_strings.py"
    sample.write_text('message = "' + "A" * 100 + '"\n', encoding="utf-8")

    formatter.run_string_fixer([sample])

    content = sample.read_text(encoding="utf-8")
    assert content != 'message = "' + "A" * 100 + '"\n'
    assert "\\" in content or len(content.splitlines()) > 1
    backup_path = sample.with_suffix(sample.suffix + ".bak")
    assert not backup_path.exists()


def test_main_runs_string_fixer_before_other_tools(monkeypatch, tmp_path: Path):
    sample = tmp_path / "example.py"
    sample.write_text("value = 1\n", encoding="utf-8")

    call_order: list[str] = []

    monkeypatch.setattr(
        formatter,
        "parse_args",
        lambda: SimpleNamespace(target="all", lint_only=False),
    )
    monkeypatch.setattr(formatter, "gather_all_targets", lambda: [sample])
    monkeypatch.setattr(formatter, "deduplicate_paths", lambda paths: list(paths))
    monkeypatch.setattr(
        formatter,
        "run_string_fixer",
        lambda targets: call_order.append("string_fixer"),
    )
    monkeypatch.setattr(
        formatter,
        "run_autoflake",
        lambda targets: call_order.append("autoflake"),
    )
    monkeypatch.setattr(
        formatter,
        "run_isort",
        lambda targets: call_order.append("isort"),
    )
    monkeypatch.setattr(
        formatter,
        "run_autopep8",
        lambda targets: call_order.append("autopep8"),
    )
    monkeypatch.setattr(
        formatter,
        "run_black",
        lambda targets: call_order.append("black"),
    )
    monkeypatch.setattr(
        formatter,
        "run_flake8",
        lambda targets: call_order.append("flake8"),
    )

    formatter.main()

    assert call_order[0] == "string_fixer"
    assert call_order[1:] == ["autoflake", "isort", "autopep8", "black", "flake8"]
