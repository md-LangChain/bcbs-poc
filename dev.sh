#!/usr/bin/env bash
#
# Start the whole local stack: the LangGraph agents and the review UI.
#
#   ./dev.sh
#
# Ctrl+C stops both.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LANGGRAPH_PORT="${LANGGRAPH_PORT:-2024}"
UI_PORT="${UI_PORT:-3000}"
LOG_DIR="$ROOT/.dev-logs"
mkdir -p "$LOG_DIR"

# Job control, so each server below becomes its own process group and can be
# killed as a whole. `uv run` and `pnpm run` each spawn a child that a plain
# kill on the wrapper would leave running.
set -m

say()  { printf '\033[36m•\033[0m %s\n' "$1"; }
warn() { printf '\033[33m!\033[0m %s\n' "$1"; }
die()  { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- preflight

[ -f .env ] || die "No .env at the repo root. Copy .env.example to .env and add your keys."

# Read the two required keys without printing their values.
missing=()
for key in LANGSMITH_API_KEY OPENAI_API_KEY; do
  value="$(grep -E "^${key}=" .env | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)"
  [ -n "$value" ] || missing+=("$key")
done
if [ ${#missing[@]} -gt 0 ]; then
  die "Missing in .env: ${missing[*]}"
fi

command -v uv   >/dev/null || die "uv is not installed. See https://docs.astral.sh/uv/"
command -v node >/dev/null || die "node is not installed."

PKG=pnpm
command -v pnpm >/dev/null || PKG=npm

for port in "$LANGGRAPH_PORT" "$UI_PORT"; do
  if lsof -ti:"$port" >/dev/null 2>&1; then
    die "Port $port is already in use. Stop that process, or set LANGGRAPH_PORT / UI_PORT."
  fi
done

if [ ! -d web/node_modules ]; then
  say "Installing UI dependencies ($PKG)…"
  (cd web && $PKG install)
fi

# ------------------------------------------------------------------ cleanup

PIDS=()
cleanup() {
  trap - INT TERM EXIT
  printf '\n'
  say "Stopping…"

  # Kill each server's whole process group (negative pid), not just the
  # wrapper process we forked.
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done

  # Give them a moment, then take the ports back from anything that survived.
  sleep 2
  for port in "$LANGGRAPH_PORT" "$UI_PORT"; do
    lsof -ti:"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  done

  wait 2>/dev/null || true
  say "Stopped."
}
trap cleanup INT TERM EXIT

wait_for() { # url, label, attempts
  local url="$1" label="$2" tries="${3:-60}"
  for _ in $(seq "$tries"); do
    if curl -sf -o /dev/null --max-time 2 "$url"; then return 0; fi
    sleep 1
  done
  warn "$label did not answer at $url — see $LOG_DIR"
  return 1
}

# ------------------------------------------------------------------- agents
#
# --n-jobs-per-worker matters: the dev server defaults to ONE concurrent job,
# and intake calls back into this same server to reach eligibility. With one
# worker a review waits on a worker it is itself holding, and hangs.
#
# --no-reload keeps the file watcher off web/node_modules.

say "Starting LangGraph agents on :$LANGGRAPH_PORT …"
uv run langgraph dev \
  --no-browser --no-reload \
  --port "$LANGGRAPH_PORT" \
  --n-jobs-per-worker 10 \
  > "$LOG_DIR/langgraph.log" 2>&1 &
PIDS+=($!)

wait_for "http://127.0.0.1:$LANGGRAPH_PORT/ok" "LangGraph" 90 \
  || die "LangGraph failed to start. Last lines:
$(tail -20 "$LOG_DIR/langgraph.log")"
say "Agents ready — intake, validation, eligibility, a2a-validation"

if grep -q "Context Hub prompt unavailable" "$LOG_DIR/langgraph.log" 2>/dev/null; then
  warn "Context Hub prompt not found; validation is using its built-in fallback prompt."
fi

# ----------------------------------------------------------------------- UI

say "Starting UI on :$UI_PORT …"
(cd web && PORT="$UI_PORT" LANGGRAPH_URL="http://127.0.0.1:$LANGGRAPH_PORT" $PKG run dev) \
  > "$LOG_DIR/ui.log" 2>&1 &
PIDS+=($!)

wait_for "http://127.0.0.1:$UI_PORT/" "UI" 60 \
  || die "UI failed to start. Last lines:
$(tail -20 "$LOG_DIR/ui.log")"

printf '\n'
printf '  \033[1mReview UI\033[0m   http://localhost:%s\n' "$UI_PORT"
printf '  LangGraph   http://127.0.0.1:%s\n' "$LANGGRAPH_PORT"
printf '  Logs        %s\n' "$LOG_DIR"
printf '\n  Ctrl+C to stop both.\n\n'

wait
