#!/usr/bin/env bash
# Backend stall analysis.
# Uses available events from perf list; falls back gracefully.
#
# Usage: ./perf_backend.sh [demo] [config] [prompt]

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== Backend Stall Analysis ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

PERF_LIST=$(perf list 2>/dev/null || true)
EVENTS="cycles,instructions"

[[ "$PERF_LIST" == *"stalled-cycles-backend"* ]] && EVENTS="$EVENTS,stalled-cycles-backend"
[[ "$PERF_LIST" == *"stalled-cycles-frontend"* ]] && EVENTS="$EVENTS,stalled-cycles-frontend"
[[ "$PERF_LIST" == *"cache-misses"* ]] && EVENTS="$EVENTS,cache-misses"
[[ "$PERF_LIST" == *"cache-references"* ]] && EVENTS="$EVENTS,cache-references"

echo "  events: $EVENTS"
echo ""

# Run perf stat with CSV output for parsing
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

perf stat -x, -e "$EVENTS" -o "$TMPFILE" -- "$DEMO" "$CONFIG" "$PROMPT"

# Display raw counter values
echo ""
echo "--- Raw Counters ---"
awk -F, 'NF >= 3 { printf "  %-30s %s\n", $3, $1 }' "$TMPFILE"

# Parse CSV: format is "value,,event-name,..."
extract_val() {
    local event="$1"
    awk -F, -v ev="$event" '
        $3 == ev || $3 ~ ("/" ev "/$") {
            if ($1 ~ /^[0-9]/) { gsub(/ /, "", $1); val += $1 }
        }
        END { if (val > 0) printf "%d", val; else print "" }
    ' "$TMPFILE"
}

compute_ratio() {
    local num="$1" den="$2"
    if [ -n "$num" ] && [ -n "$den" ] && [ "$den" != "0" ]; then
        awk "BEGIN { printf \"%.2f%%\", ($num / $den) * 100 }"
    else
        echo "N/A"
    fi
}

compute_ipc() {
    local ins="$1" cyc="$2"
    if [ -n "$ins" ] && [ -n "$cyc" ] && [ "$cyc" != "0" ]; then
        awk "BEGIN { printf \"%.3f\", $ins / $cyc }"
    else
        echo "N/A"
    fi
}

CYCLES=$(extract_val "cycles")
INSTRUCTIONS=$(extract_val "instructions")
STALLED_BACKEND=$(extract_val "stalled-cycles-backend")
STALLED_FRONTEND=$(extract_val "stalled-cycles-frontend")
CACHE_MISSES=$(extract_val "cache-misses")
CACHE_REFS=$(extract_val "cache-references")

echo ""
echo "--- Computed Ratios ---"
echo "  IPC:                $(compute_ipc "$INSTRUCTIONS" "$CYCLES")"
echo "  Backend stall%:     $(compute_ratio "$STALLED_BACKEND" "$CYCLES")"
echo "  Frontend stall%:    $(compute_ratio "$STALLED_FRONTEND" "$CYCLES")"
echo "  Cache miss rate:    $(compute_ratio "$CACHE_MISSES" "$CACHE_REFS")"

echo ""
echo "--- What to watch ---"
echo "  IPC = instructions / cycles  (higher is better)"
echo "  stalled-cycles-backend / cycles = backend stall ratio"
echo "  stalled-cycles-frontend / cycles = frontend stall ratio"
echo "  cache-misses / cache-references = cache miss rate"
echo ""
echo "  If interleaved has higher IPC and lower stall%, the two models"
echo "  overlap their memory stalls."
