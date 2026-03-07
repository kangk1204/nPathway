#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${ROOT_DIR}/.venv"
EXTRAS=""
DRY_RUN=0
SMOKE_CHECK=1

print_help() {
  cat <<'EOF'
Usage: bash scripts/install_npathway_easy.sh [options]

Beginner-friendly one-step installer for nPathway.

Options:
  --python PATH       Python executable to use (default: auto-pick python3.11/3.12/3.10, then python3)
  --venv-dir PATH     Virtual environment directory (default: .venv in repo root)
  --extras LIST       Optional extras, e.g. report or scbert,report
  --no-smoke-check    Skip post-install command checks
  --dry-run           Print the commands without executing them
  -h, --help          Show this help message
EOF
}

pick_python_bin() {
  local candidate
  for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ %q' "$1"
    shift
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    --no-smoke-check)
      SMOKE_CHECK=0
      shift
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

if [[ -z "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN="$(pick_python_bin)"; then
    echo "No supported Python executable found. Install Python 3.10, 3.11, or 3.12, or pass --python PATH." >&2
    exit 1
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

INSTALL_TARGET='-e .'
if [[ -n "$EXTRAS" ]]; then
  INSTALL_TARGET="-e .[$EXTRAS]"
fi

printf 'nPathway easy installer\n'
printf '%s\n' "- repo: ${ROOT_DIR}"
printf '%s\n' "- python: ${PYTHON_BIN}"
printf '%s\n' "- venv: ${VENV_DIR}"
printf '%s\n' "- extras: ${EXTRAS:-none}"

run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
run_cmd "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
if [[ -n "$EXTRAS" ]]; then
  run_cmd "$VENV_DIR/bin/pip" install -e ".[${EXTRAS}]"
else
  run_cmd "$VENV_DIR/bin/pip" install -e .
fi

if [[ "$SMOKE_CHECK" -eq 1 ]]; then
  run_cmd "$VENV_DIR/bin/npathway-validate-inputs" --help
  run_cmd "$VENV_DIR/bin/npathway-demo" --help
  run_cmd "$VENV_DIR/bin/npathway-bulk-workflow" --help
  run_cmd "$VENV_DIR/bin/npathway-scrna-easy" --help
  run_cmd "$VENV_DIR/bin/npathway-compare-gsea" --help
  run_cmd "$VENV_DIR/bin/npathway-convert-seurat" --help
  run_cmd "$VENV_DIR/bin/npathway-convert-10x" --help
fi

printf '\nNext steps\n'
printf '%s\n' "1. source ${VENV_DIR}/bin/activate"
printf '%s\n' "2. npathway-demo bulk --output-dir results/demo_bulk_case_vs_control"
printf '%s\n' "3. npathway-demo scrna --output-dir results/demo_scrna_case_vs_control"
printf '%s\n' "4. npathway-scrna-easy --wizard-only --adata data/my_scrna.h5ad --condition-col condition --case case --control control"

if command -v Rscript >/dev/null 2>&1; then
  if Rscript -e "pkgs <- c('limma','edgeR'); quit(status = ifelse(all(sapply(pkgs, requireNamespace, quietly=TRUE)), 0, 1))" >/dev/null 2>&1; then
    if Rscript -e "quit(status = ifelse(requireNamespace('sva', quietly=TRUE), 0, 1))" >/dev/null 2>&1; then
      printf '%s\n' "- R batch-aware backend: available (guarded SVA auto support detected)"
    else
      printf '%s\n' "- R batch-aware backend: available (guarded SVA auto will skip because the R package 'sva' is missing)"
    fi
  else
    printf '%s\n' "- R batch-aware backend: R found, but limma/edgeR are missing"
  fi
else
  printf '%s\n' "- R batch-aware backend: Rscript not found; scrna-easy will fall back to the simpler backend"
fi
