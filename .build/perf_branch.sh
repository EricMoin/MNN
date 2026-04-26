#!/usr/bin/env bash
# Branch prediction efficiency.
# High miss rate → unpredictable branches wasting pipeline slots.
#
# Usage: ./perf_branch.sh [demo] [config] [prompt]

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== Branch Prediction ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

PERF_LIST=$(perf list 2>/dev/null || true)
EVENTS=""

[[ "$PERF_LIST" == *"branch-misses"* ]] && EVENTS="branch-misses"
if [[ "$PERF_LIST" == *"branches"* ]]; then
    [ -n "$EVENTS" ] && EVENTS="$EVENTS,branches" || EVENTS="branches"
fi
[[ "$PERF_LIST" == *"branch-load-misses"* ]] && EVENTS="$EVENTS,branch-load-misses"
[[ "$PERF_LIST" == *"branch-loads"* ]] && EVENTS="$EVENTS,branch-loads"

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

BRANCH_MISSES=$(extract_val "branch-misses")
BRANCHES=$(extract_val "branches")
BRANCH_LOAD_MISSES=$(extract_val "branch-load-misses")
BRANCH_LOADS=$(extract_val "branch-loads")

echo ""
echo "--- Computed Ratios ---"
echo "  Branch miss rate:      $(compute_ratio "$BRANCH_MISSES" "$BRANCHES")"
echo "  Branch load miss rate: $(compute_ratio "$BRANCH_LOAD_MISSES" "$BRANCH_LOADS")"

echo ""
echo "--- What to watch ---"
echo "  miss rate = branch-misses / branches"
echo "  >5% → too many unpredictable branches (decoder loop, sampler)"
echo "  Interleaved mode adds branchy while-loop; if miss rate unchanged,"
echo "  the extra branches are well-predicted (stop conditions rarely fire)."
