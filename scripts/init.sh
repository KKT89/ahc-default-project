#!/usr/bin/env bash
set -Eeuo pipefail

here_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here_dir}/.." && pwd)"
cd "${repo_root}"

info() { printf "[info] %s\n" "$*"; }
err()  { printf "[error] %s\n" "$*" >&2; }

usage() {
  cat <<USAGE
Usage: bash scripts/init.sh -o <objective> [--interactive]

Options:
  -o, --objective   max|min|maximize|minimize (required)
  -i, --interactive mark problem as interactive (optional)
  -h, --help        show this help
USAGE
}

lc() { printf %s "$1" | tr '[:upper:]' '[:lower:]'; }

objective=""
interactive=false

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

case "${objective}" in
  max|min|maximize|minimize) : ;;
  "") err "-o/--objective is required"; usage; exit 2 ;;
  *)  err "Invalid objective: ${objective}"; usage; exit 2 ;;
esac

trap 'err "setup failed (line $LINENO)"' ERR

# WSL 環境でコピーされた Zone.Identifier ファイルを削除
find "${repo_root}" -name '*:Zone.Identifier' -type f -delete 2>/dev/null || true

if ! command -v uv >/dev/null 2>&1; then
  err "'uv' is required but not found. Install from https://github.com/astral-sh/uv"
  exit 1
fi

if [[ -f "${repo_root}/uv.lock" ]]; then
  info "Syncing dependencies from uv.lock (--frozen)"
  uv sync --frozen
else
  info "Resolving and installing dependencies with uv sync"
  uv sync
fi

cmd=(uv run ahc-tester/setup.py "${objective}")
$interactive && cmd+=("-i")
info "Running setup: ${cmd[*]}"
"${cmd[@]}"
