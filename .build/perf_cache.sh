#!/usr/bin/env bash
# Cache hierarchy breakdown: L1 hit/miss, LLC hit/miss.
# Low LLC miss rate → data fits in cache; high → DRAM bottleneck.
#
# Usage: ./perf_cache.sh [demo] [config] [prompt]
# Default: ./llm_demo ~/Project/models/qwen2.5/config-inter.json prompt.txt

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== Cache Hierarchy ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

PERF_LIST=$(perf list 2>/dev/null || true)
EVENTS=""

[[ "$PERF_LIST" == *"l1-dcache-loads"* ]] && EVENTS="l1-dcache-loads"
[[ "$PERF_LIST" == *"l1-dcache-stores"* ]] && EVENTS="$EVENTS,l1-dcache-stores"
[[ "$PERF_LIST" == *"l1-dcache-load-misses"* ]] && EVENTS="$EVENTS,l1-dcache-load-misses"
[[ "$PERF_LIST" == *"l1-icache-loads"* ]] && EVENTS="$EVENTS,l1-icache-loads"
[[ "$PERF_LIST" == *"l1-icache-load-misses"* ]] && EVENTS="$EVENTS,l1-icache-load-misses"
[[ "$PERF_LIST" == *"llc-load-misses"* ]] && EVENTS="$EVENTS,llc-load-misses"
[[ "$PERF_LIST" == *"llc-loads"* ]] && EVENTS="$EVENTS,llc-loads"
[[ "$PERF_LIST" == *"llc-store-misses"* ]] && EVENTS="$EVENTS,llc-store-misses"
[[ "$PERF_LIST" == *"llc-stores"* ]] && EVENTS="$EVENTS,llc-stores"

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

L1D_LOADS=$(extract_val "l1-dcache-loads")
L1D_LOAD_MISSES=$(extract_val "l1-dcache-load-misses")
LLC_LOAD_MISSES=$(extract_val "llc-load-misses")
LLC_LOADS=$(extract_val "llc-loads")
LLC_STORE_MISSES=$(extract_val "llc-store-misses")
LLC_STORES=$(extract_val "llc-stores")
L1I_LOADS=$(extract_val "l1-icache-loads")
L1I_LOAD_MISSES=$(extract_val "l1-icache-load-misses")

echo ""
echo "--- Computed Ratios ---"
echo "  L1D load miss rate:  $(compute_ratio "$L1D_LOAD_MISSES" "$L1D_LOADS")"
echo "  LLC load miss rate:  $(compute_ratio "$LLC_LOAD_MISSES" "$LLC_LOADS")"
echo "  LLC store miss rate: $(compute_ratio "$LLC_STORE_MISSES" "$LLC_STORES")"
echo "  L1I load miss rate:  $(compute_ratio "$L1I_LOAD_MISSES" "$L1I_LOADS")"

echo ""
echo "--- What to watch ---"
echo "  LLC load miss rate = llc-load-misses / llc-loads"
echo "  LLC miss high → data spills to DRAM → memory-bound"
echo "  Compare serial vs interleaved: if interleaved LLC miss rate is"
echo "  lower, the two models' hot data overlaps in cache."
echo "  l1-dcache-loads can be used as a proxy for total L1 traffic."
