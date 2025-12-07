#!/usr/bin/env bash
set -Eeuo pipefail

# Quick setup script (uv-managed, pyproject/uv.lock ベース)
# - 依存解決は `uv sync` にまとめて任せる（.venv の事前作成は不要）
# - Run setup: `uv run ahc-tester/setup.py <objective> [-i]`
#   - objective (required): max|min|maximize|minimize
#   - -i/--interactive (optional): interactive problem flag

here_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here_dir}/.." && pwd)"
cd "${repo_root}"

info() { printf "[info] %s\n" "$*"; }
err()  { printf "[error] %s\n" "$*" >&2; }

cleanup_zone_identifier_files() {
  mapfile -t zid_files < <(find "${repo_root}" -name '*:Zone.Identifier' -type f -print)
  if [[ ${#zid_files[@]} -eq 0 ]]; then
    return
  fi
  info "Removing ${#zid_files[@]} Zone.Identifier files"
  rm -f -- "${zid_files[@]}"
}

usage() {
  cat <<USAGE
Usage: bash scripts/init.sh -o <objective> [--interactive]

Options:
  -o, --objective   max|min|maximize|minimize (required)
  -i, --interactive mark problem as interactive (optional)
  -h, --help       show this help
USAGE
}

objective=""
interactive=false

lc() { printf %s "$1" | tr '[:upper:]' '[:lower:]'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--objective)
      [[ $# -ge 2 ]] || { err "Missing value for $1"; usage; exit 2; }
      objective="$(lc "$2")"; shift 2 ;;
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

trap 'err "setup failed (line $LINENO)"' ERR

cleanup_zone_identifier_files

if ! command -v uv >/dev/null 2>&1; then
  err "'uv' is required but not found. Install from https://github.com/astral-sh/uv"
  exit 1
fi

sync_cmd=(uv sync)
if [[ -f "${repo_root}/uv.lock" ]]; then
  info "Syncing dependencies from uv.lock (--frozen)"
  sync_cmd+=(--frozen)
else
  info "Resolving and installing dependencies with uv sync"
fi
"${sync_cmd[@]}"

cmd=(uv run ahc-tester/setup.py "$objective")
if $interactive; then cmd+=("-i"); fi
info "Running setup: ${cmd[*]}"
"${cmd[@]}"
