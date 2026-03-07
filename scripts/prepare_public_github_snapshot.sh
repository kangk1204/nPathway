#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
ROOT_OVERRIDE=""

print_help() {
  cat <<'USAGE'
Usage: bash scripts/prepare_public_github_snapshot.sh [options]

Untrack private manuscript/code-generation assets that should not be pushed to GitHub.
This script only removes files from the git index with `git rm --cached`; it does not
remove local working files.

Options:
  --root PATH   Run against a specific git working tree
  --dry-run     Print what would be untracked without changing the index
  -h, --help    Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help >&2
      exit 1
      ;;
  esac
done

if [[ -n "$ROOT_OVERRIDE" ]]; then
  ROOT_DIR="$ROOT_OVERRIDE"
else
  ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi

if [[ -z "$ROOT_DIR" || ! -d "$ROOT_DIR/.git" ]]; then
  echo "ERROR: no git repository found. Run this inside your git repo or pass --root PATH." >&2
  exit 1
fi

cd "$ROOT_DIR"

patterns=(
  "results"
  "paper"
  "data/external"
  "final_delivery_*"
  ".latex-env"
  ".r-lib"
  ".claude"
  "codex-parallel.log"
  "docs/alzheimers_case_study_plan_*.md"
  "docs/methods_submission_*.md"
  "docs/methods_special_issue_submission_plan_*.md"
  "docs/methods_cover_letter_draft_*.md"
  "docs/phase4_manuscript_outline.md"
  "docs/dynamic_pathway_product_blueprint_*.md"
  "docs/agent*_*.md"
  "docs/agent_supervisor_execution_summary_*.md"
  "docs/pathway_agent_framework_*.md"
  "scripts/run_publication_suite.sh"
  "scripts/validate_manuscript_consistency.py"
  "scripts/build_submission_stats_package.py"
  "scripts/package_publication_artifacts.py"
  "scripts/patch_manuscript_claim_blocks.py"
  "scripts/generate_paper_draft.py"
  "scripts/check_manuscript_consistency.py"
  "scripts/run_gse203206_case_study.py"
  "scripts/run_gse147528_ec_case_study.py"
  "scripts/run_gse203206_common_de_comparison.py"
  "scripts/run_gse214921_mdd_case_study.py"
  "scripts/build_de_ranking_sensitivity_figure.py"
  "scripts/build_bulk_common_de_engine_figure.py"
  "scripts/build_mdd_deseq2_validation_figure.py"
  "tests/test_de_ranking_sensitivity_figure.py"
  "tests/test_bulk_common_de_engine_figure.py"
  "tests/test_mdd_deseq2_validation_figure.py"
  "tests/test_manuscript_consistency.py"
  "tests/test_manuscript_consistency_checker.py"
  "tests/test_submission_stats_package.py"
)

tracked=()
for pattern in "${patterns[@]}"; do
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    tracked+=("$path")
  done < <(git ls-files -- "$pattern")
done

if [[ ${#tracked[@]} -eq 0 ]]; then
  echo "No tracked private assets matched the public-snapshot rules."
  exit 0
fi

# Unique + stable order without requiring Bash 4 mapfile.
unique_tracked=()
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  unique_tracked+=("$path")
done < <(printf '%s\n' "${tracked[@]}" | awk '!seen[$0]++')
tracked=("${unique_tracked[@]}")

echo "Git root: $ROOT_DIR"
echo "Tracked private assets to untrack:"
printf ' - %s\n' "${tracked[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

git rm -r --cached -- "${tracked[@]}"

echo
echo "Done. Local files are still on disk."
echo "Next: git status && git add .gitignore README.md tests/test_project_scripts.py scripts/prepare_public_github_snapshot.sh"
