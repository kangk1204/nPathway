"""Tests for the Seurat-to-h5ad conversion helper CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_convert_seurat_cli_check_only_writes_status_report(tmp_path) -> None:
    """Check-only mode should report readiness without needing a Seurat object."""
    status_json = tmp_path / "seurat_status.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.convert_seurat",
            "--check-only",
            "--status-json",
            str(status_json),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert status_json.exists()
    payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert "packages" in payload
    assert {item["name"] for item in payload["packages"]} == {"Seurat", "SeuratDisk"}
    assert "nPathway Seurat conversion readiness" in result.stdout
