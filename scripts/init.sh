#!/usr/bin/env bash
set -Eeuo pipefail

# Quick setup script (uv-first, requirements required)
# - Create venv with `uv venv .venv`
# - Install deps from ahc-tester/requirements.txt (required)
# - Run setup: `uv run ahc-tester/setup.py <objective> <tl> [-i]`
#   - objective (required): max|min|maximize|minimize
#   - tl (required): time limit in seconds (float)
#   - -i/--interactive (optional): interactive problem flag

here_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here_dir}/.." && pwd)"
cd "${repo_root}"

req_file="ahc-tester/requirements.txt"

info() { printf "[info] %s\n" "$*"; }
err()  { printf "[error] %s\n" "$*" >&2; }

usage() {
  cat <<USAGE
Usage: bash scripts/init.sh -o <objective> -t <seconds> [--interactive]

Options:
  -o, --objective   max|min|maximize|minimize (required)
  -t, --tl          time limit in seconds (float, required)
  -i, --interactive mark problem as interactive (optional)
  -h, --help       show this help
USAGE
}

objective=""
tl=""
interactive=false

lc() { printf %s "$1" | tr '[:upper:]' '[:lower:]'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--objective)
      [[ $# -ge 2 ]] || { err "Missing value for $1"; usage; exit 2; }
      objective="$(lc "$2")"; shift 2 ;;
    -t|--tl)
      [[ $# -ge 2 ]] || { err "Missing value for $1"; usage; exit 2; }
      tl="$2"; shift 2 ;;
    -i|--interactive)
      interactive=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    *)
      err "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# Validate objective
case "$(lc "${objective:-}")" in
  max|min|maximize|minimize) : ;;
  "") err "-o/--objective is required"; usage; exit 2 ;;
  *) err "Invalid objective: $objective"; usage; exit 2 ;;
esac

# Validate tl is positive number
if [[ -z "$tl" ]]; then
  err "-t/--tl is required"; usage; exit 2
fi
if ! awk "BEGIN{exit(!($tl+0>0))}" </dev/null; then
  err "tl must be a positive number (seconds), got: $tl"; exit 2
fi

trap 'err "setup failed (line $LINENO)"' ERR

if ! command -v uv >/dev/null 2>&1; then
  err "'uv' is required but not found. Install from https://github.com/astral-sh/uv"
  exit 1
fi

if [[ ! -f "${req_file}" ]]; then
  err "Requirements file not found: ${req_file}"
  exit 1
fi

info "Creating virtual environment (.venv) with uv"
uv venv .venv

info "Installing dependencies from ${req_file}"
uv pip install -r "${req_file}"

cmd=(uv run ahc-tester/setup.py "$objective" "$tl")
if $interactive; then cmd+=("-i"); fi
info "Running setup: ${cmd[*]}"
"${cmd[@]}"

