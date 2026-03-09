"""Tests for public project metadata and script entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
GITIGNORE = ROOT / ".gitignore"
README = ROOT / "README.md"
QUICKSTART_GUIDE = ROOT / "docs" / "quickstart_input_guide.md"
USAGE_MODES_GUIDE = ROOT / "docs" / "discovery_vs_comparison_modes.md"
VALIDATE_INPUTS_SCRIPT = ROOT / "scripts" / "validate_npathway_inputs.py"
BULK_WORKFLOW_SCRIPT = ROOT / "scripts" / "run_batch_aware_bulk_workflow.py"
COMPARE_GSEA_SCRIPT = ROOT / "scripts" / "run_curated_vs_dynamic_gsea.py"
SCRNA_EASY_SCRIPT = ROOT / "scripts" / "run_scrna_easy.py"
SCRNA_PSEUDOBULK_SCRIPT = ROOT / "scripts" / "run_scrna_pseudobulk_dynamic_pathway.py"
SEURAT_CONVERT_SCRIPT = ROOT / "scripts" / "convert_seurat_to_h5ad.py"
TENX_CONVERT_SCRIPT = ROOT / "scripts" / "convert_10x_to_h5ad.py"
INSTALL_EASY_SCRIPT = ROOT / "scripts" / "install_npathway_easy.sh"
PUBLIC_SNAPSHOT_SCRIPT = ROOT / "scripts" / "prepare_public_github_snapshot.sh"
PUBLIC_REFERENCE_BUILD_SCRIPT = ROOT / "scripts" / "build_public_reference_stack.py"


def test_pyproject_declares_public_entrypoints() -> None:
    """Packaging metadata should expose the public CLI entrypoints."""
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "scbert = [\"performer-pytorch>=1.1\"]" in pyproject
    assert "all-models = [\"scgpt\", \"transformers>=4.30\", \"performer-pytorch>=1.1\"]" in pyproject
    assert 'npathway = "npathway.cli.main:main"' in pyproject
    assert 'npathway-validate-inputs = "npathway.cli.validate_inputs:main"' in pyproject
    assert 'npathway-demo = "npathway.cli.demo:main"' in pyproject
    assert 'npathway-bulk-workflow = "npathway.cli.bulk_workflow:main"' in pyproject
    assert 'npathway-compare-gsea = "npathway.cli.compare_gsea:main"' in pyproject
    assert 'npathway-scrna-easy = "npathway.cli.scrna_easy:main"' in pyproject
    assert 'npathway-convert-seurat = "npathway.cli.convert_seurat:main"' in pyproject
    assert 'npathway-convert-10x = "npathway.cli.convert_10x:main"' in pyproject


def test_readme_is_github_friendly_and_public_only() -> None:
    """README should guide first-time users without linking private manuscript assets."""
    readme = README.read_text(encoding="utf-8")

    assert "Option 1. One-click install" in readme
    assert "bash scripts/install_npathway_easy.sh" in readme
    assert "auto-prefers `python3.11`, `python3.12`, or `python3.10`" in readme
    assert "npathway quickstart" in readme
    assert "npathway demo bulk" in readme
    assert "npathway demo scrna" in readme
    assert "results/demo_bulk_case_vs_control/index.html" in readme
    assert "results/demo_scrna_case_vs_control/index.html" in readme
    assert "interactive plots and interactive tables" in readme
    assert "docs/assets/npathway_workflow_overview.png" in readme
    assert "docs/assets/npathway_dashboard_preview.png" in readme
    assert "npathway validate bulk" in readme
    assert "npathway validate scrna" in readme
    assert "npathway run bulk" in readme
    assert "npathway run scrna" in readme
    assert "npathway compare --help" in readme
    assert "npathway convert seurat --check-only" in readme
    assert "npathway convert 10x --check-only" in readme
    assert "npathway-demo bulk" in readme
    assert "npathway-demo scrna" in readme
    assert "npathway-validate-inputs" in readme
    assert "npathway-bulk-workflow" in readme
    assert "npathway-scrna-easy" in readme
    assert "npathway-compare-gsea" in readme
    assert "npathway-convert-seurat" in readme
    assert "npathway-convert-10x" in readme
    assert "npathway references build --output-dir data/reference/public" in readme
    assert "reactome,wikipathways,pathwaycommons" in readme
    assert "Reference Layers" in readme
    assert "How nPathway Differs From Standard GSEA" in readme
    assert "Private manuscript materials, submission figures, and large generated results" in readme
    assert "prepare_public_github_snapshot.sh --dry-run" in readme

    assert "data/templates/" in readme
    assert "docs/quickstart_input_guide.md" in readme
    assert "docs/discovery_vs_comparison_modes.md" in readme
    assert "run_scrna_pseudobulk_dynamic_pathway.py" in readme
    assert "validate_npathway_inputs.py" in readme
    assert "samples_by_genes" in readme
    assert "must map to **exactly one** group label" in readme
    assert "Hard minimum: **2 samples per group**" in readme
    assert "Hard minimum: **2 retained pseudobulk samples per group**" in readme
    assert "bulk_demo_case_vs_ctrl" in readme
    assert "scrna_demo_case_vs_ctrl" in readme
    assert "--html-out" in readme

    assert "run_gse203206_case_study.py" not in readme
    assert "run_gse147528_ec_case_study.py" not in readme
    assert "run_gse214921_mdd_case_study.py" not in readme
    assert "build_mdd_deseq2_validation_figure.py" not in readme
    assert "methods_submission_figures" not in readme
    assert "methods_submission_metadata" not in readme
    assert "fig8_mental_neural_audit.pdf" not in readme


def test_public_docs_and_templates_exist() -> None:
    """Repository should ship public templates and beginner docs."""
    assert GITIGNORE.exists()
    assert QUICKSTART_GUIDE.exists()
    assert USAGE_MODES_GUIDE.exists()
    assert (ROOT / "docs" / "assets" / "npathway_workflow_overview.png").exists()
    assert (ROOT / "docs" / "assets" / "npathway_dashboard_preview.png").exists()
    assert (ROOT / "data" / "templates" / "README.md").exists()
    assert (ROOT / "data" / "templates" / "bulk_matrix_template.csv").exists()
    assert (ROOT / "data" / "templates" / "bulk_metadata_template.csv").exists()
    assert (ROOT / "data" / "templates" / "scrna_obs_template.csv").exists()
    assert (ROOT / "data" / "bulk_demo_case_vs_ctrl" / "README.md").exists()
    assert (ROOT / "data" / "bulk_demo_case_vs_ctrl" / "bulk_matrix_case_ctrl_demo.csv").exists()
    assert (ROOT / "data" / "bulk_demo_case_vs_ctrl" / "bulk_metadata_case_ctrl_demo.csv").exists()
    assert (ROOT / "data" / "bulk_demo_case_vs_ctrl" / "bulk_reference_demo.gmt").exists()
    assert (ROOT / "data" / "scrna_demo_case_vs_ctrl" / "README.md").exists()
    assert (ROOT / "data" / "scrna_demo_case_vs_ctrl" / "demo_scrna_case_ctrl.h5ad").exists()
    assert (ROOT / "data" / "scrna_demo_case_vs_ctrl" / "demo_scrna_obs_preview.csv").exists()
    assert (ROOT / "data" / "scrna_demo_case_vs_ctrl" / "scrna_reference_demo.gmt").exists()


def test_gitignore_blocks_private_and_generated_assets() -> None:
    """Public GitHub snapshot should ignore manuscript and generated outputs."""
    gitignore = GITIGNORE.read_text(encoding="utf-8")

    assert "results/" in gitignore
    assert "data/external/" in gitignore
    assert "paper/" in gitignore
    assert "final_delivery_*/" in gitignore
    assert "docs/methods_submission_*.md" in gitignore
    assert "docs/methods_cover_letter_draft_*.md" in gitignore
    assert "scripts/run_gse203206_case_study.py" in gitignore
    assert "scripts/run_gse147528_ec_case_study.py" in gitignore
    assert "scripts/run_gse214921_mdd_case_study.py" in gitignore
    assert "scripts/build_de_ranking_sensitivity_figure.py" in gitignore
    assert "scripts/build_bulk_common_de_engine_figure.py" in gitignore
    assert "scripts/build_mdd_deseq2_validation_figure.py" in gitignore
    assert "scripts/build_submission_stats_package.py" in gitignore
    assert "scripts/validate_manuscript_consistency.py" in gitignore
    assert "tests/test_de_ranking_sensitivity_figure.py" in gitignore
    assert "tests/test_submission_stats_package.py" in gitignore
    assert PUBLIC_SNAPSHOT_SCRIPT.exists()


def test_user_docs_describe_discovery_and_comparison_modes() -> None:
    """Beginner docs should explicitly branch users into discovery vs comparison paths."""
    quickstart = QUICKSTART_GUIDE.read_text(encoding="utf-8")
    modes = USAGE_MODES_GUIDE.read_text(encoding="utf-8")

    assert "Discovery mode" in quickstart
    assert "Comparison mode" in quickstart
    assert "npathway-compare-gsea" in quickstart
    assert "npathway-bulk-workflow" in quickstart
    assert "npathway-scrna-easy" in quickstart
    assert "npathway-convert-seurat" in quickstart
    assert "npathway-convert-10x" in quickstart
    assert "install_npathway_easy.sh" in quickstart

    assert "Mode 1. Discovery Mode" in modes
    assert "Mode 2. Comparison Mode" in modes
    assert "pathway-specialized layer" in modes
    assert "full ranked gene table" in modes


def test_public_cli_help_scripts_work_without_editable_install() -> None:
    """Public scripts should be runnable without relying on editable install."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for script in [
        SCRNA_PSEUDOBULK_SCRIPT,
        VALIDATE_INPUTS_SCRIPT,
        BULK_WORKFLOW_SCRIPT,
        COMPARE_GSEA_SCRIPT,
        SCRNA_EASY_SCRIPT,
        SEURAT_CONVERT_SCRIPT,
        TENX_CONVERT_SCRIPT,
        PUBLIC_REFERENCE_BUILD_SCRIPT,
    ]:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    result = subprocess.run(
        ["bash", str(INSTALL_EASY_SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "auto-pick python3.11/3.12/3.10" in result.stdout

    result = subprocess.run(
        ["bash", str(INSTALL_EASY_SCRIPT), "--dry-run", "--no-smoke-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "python" in result.stdout.lower()
    assert ".venv" in result.stdout

    result = subprocess.run(
        ["bash", str(INSTALL_EASY_SCRIPT), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "npathway --help" in result.stdout
    assert "npathway quickstart" in result.stdout
    assert "npathway-demo --help" in result.stdout
    assert "npathway-bulk-workflow --help" in result.stdout
    assert "npathway-scrna-easy --help" in result.stdout
    assert "npathway-compare-gsea --help" in result.stdout

    result = subprocess.run(
        [sys.executable, "-m", "npathway.cli.main", "quickstart"],
        cwd=ROOT,
        env={**env, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "npathway demo bulk" in result.stdout
    assert "npathway run bulk --help" in result.stdout
    assert "results/demo_bulk_case_vs_control/index.html" in result.stdout

    result = subprocess.run(
        [sys.executable, "-m", "npathway.cli.main", "references", "--help"],
        cwd=ROOT,
        env={**env, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "npathway references build" in result.stdout

    result = subprocess.run(
        ["bash", str(PUBLIC_SNAPSHOT_SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Untrack private manuscript/code-generation assets" in result.stdout
