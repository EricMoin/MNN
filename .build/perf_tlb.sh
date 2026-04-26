#!/usr/bin/env bash
# TLB (Translation Lookaside Buffer) pressure analysis.
# Critical for LLM inference: large KV caches, weight matrices,
# and activation buffers compete for TLB entries.
#
# Usage: ./perf_tlb.sh [demo] [config] [prompt]

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== TLB Pressure Analysis ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

PERF_LIST=$(perf list 2>/dev/null || true)
EVENTS=""

[[ "$PERF_LIST" == *"dtlb-loads"* ]] && EVENTS="$EVENTS,dtlb-loads"
[[ "$PERF_LIST" == *"dtlb-load-misses"* ]] && EVENTS="$EVENTS,dtlb-load-misses"
[[ "$PERF_LIST" == *"dtlb-stores"* ]] && EVENTS="$EVENTS,dtlb-stores"
[[ "$PERF_LIST" == *"dtlb-store-misses"* ]] && EVENTS="$EVENTS,dtlb-store-misses"
[[ "$PERF_LIST" == *"itlb-load-misses"* ]] && EVENTS="$EVENTS,itlb-load-misses"
[[ "$PERF_LIST" == *"dtlb_load_misses.walk_completed"* ]] && EVENTS="$EVENTS,dtlb_load_misses.walk_completed"
[[ "$PERF_LIST" == *"dtlb_store_misses.walk_completed"* ]] && EVENTS="$EVENTS,dtlb_store_misses.walk_completed"

EVENTS="${EVENTS#,}"

if [ -z "$EVENTS" ]; then
    echo "ERROR: No TLB events available on this platform."
    exit 1
fi

echo "  events: $EVENTS"
echo ""

TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

perf stat -x, -e "$EVENTS" -o "$TMPFILE" -- "$DEMO" "$CONFIG" "$PROMPT"

echo ""

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

echo "=== TLB Ratio Summary ==="

VAL_LOADS=$(extract_val "dtlb-loads")
VAL_LOAD_MISSES=$(extract_val "dtlb-load-misses")
VAL_STORES=$(extract_val "dtlb-stores")
VAL_STORE_MISSES=$(extract_val "dtlb-store-misses")
VAL_ITLB_MISSES=$(extract_val "itlb-load-misses")
VAL_PAGE_WALKS_LOAD=$(extract_val "dtlb_load_misses.walk_completed")
VAL_PAGE_WALKS_STORE=$(extract_val "dtlb_store_misses.walk_completed")

echo "  dTLB load miss rate:      $(compute_ratio "$VAL_LOAD_MISSES" "$VAL_LOADS")"
echo "  dTLB store miss rate:     $(compute_ratio "$VAL_STORE_MISSES" "$VAL_STORES")"
echo "  Page walk completions (load):  ${VAL_PAGE_WALKS_LOAD:-N/A}"
echo "  Page walk completions (store): ${VAL_PAGE_WALKS_STORE:-N/A}"
echo "  iTLB load misses:         ${VAL_ITLB_MISSES:-N/A}"
echo ""

echo "--- What to watch ---"
echo "  dTLB miss rate >1% → significant address translation overhead"
echo "  High page walk count → large working set doesn't fit in TLB"
echo "  For LLM: KV cache, weight matrices, and activation buffers"
echo "    compete for TLB entries"
echo "  Consider huge pages (2MB/1GB) if dTLB miss rate is high"
echo "  iTLB misses usually low for LLM (hot loops are compact)"
echo "  Compare serial vs interleaved: if interleaved has higher"
echo "    TLB miss rate, the two models' working sets conflict"
