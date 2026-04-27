#!/usr/bin/env bash
# Memory bandwidth estimation.
# LLM inference (especially decode phase) is heavily memory-bandwidth
# bound. This script measures DRAM traffic using uncore IMC events
# when available, falling back to core-level DRAM indicators.
#
# Usage: ./perf_mem_bw.sh [demo] [config] [prompt]
# NOTE: uncore IMC events require system-wide collection and may need root.

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== Memory Bandwidth Estimation ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

PERF_LIST=$(perf list 2>/dev/null || true)

HAS_UNCORE_READ=false
HAS_UNCORE_WRITE=false
[[ "$PERF_LIST" == *"unc_m_cas_count.rd"* ]] && HAS_UNCORE_READ=true
[[ "$PERF_LIST" == *"unc_m_cas_count.wr"* ]] && HAS_UNCORE_WRITE=true

EVENTS="cycles,task-clock"
$HAS_UNCORE_READ && EVENTS="$EVENTS,unc_m_cas_count.rd"
$HAS_UNCORE_WRITE && EVENTS="$EVENTS,unc_m_cas_count.wr"

[[ "$PERF_LIST" == *"ocr.demand_data_rd.dram"* ]] && EVENTS="$EVENTS,ocr.demand_data_rd.dram"
[[ "$PERF_LIST" == *"offcore_requests.demand_data_rd"* ]] && EVENTS="$EVENTS,offcore_requests.demand_data_rd"
[[ "$PERF_LIST" == *"offcore_requests.demand_rfo"* ]] && EVENTS="$EVENTS,offcore_requests.demand_rfo"
[[ "$PERF_LIST" == *"mem_load_l3_miss_retired.local_dram"* ]] && EVENTS="$EVENTS,mem_load_l3_miss_retired.local_dram"

echo "  events: $EVENTS"
echo ""

TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

# uncore IMC events require system-wide (-a) mode
if $HAS_UNCORE_READ || $HAS_UNCORE_WRITE; then
    echo "  [uncore IMC events detected, using system-wide (-a) mode]"
    echo ""
    perf stat -x, -a -e "$EVENTS" -o "$TMPFILE" -- "$DEMO" "$CONFIG" "$PROMPT"
else
    echo "  [no uncore IMC events, using core-level DRAM indicators only]"
    echo ""
    perf stat -x, -e "$EVENTS" -o "$TMPFILE" -- "$DEMO" "$CONFIG" "$PROMPT"
fi

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

extract_val_float() {
    local event="$1"
    awk -F, -v ev="$event" '
        $3 == ev || $3 ~ ("/" ev "/$") {
            if ($1 ~ /^[0-9]/) { gsub(/ /, "", $1); val += $1 }
        }
        END { if (val > 0) print val; else print "" }
    ' "$TMPFILE"
}

echo "=== Bandwidth Summary ==="

TASK_CLOCK=$(extract_val_float "task-clock")
VAL_CAS_RD=$(extract_val "unc_m_cas_count.rd")
VAL_CAS_WR=$(extract_val "unc_m_cas_count.wr")
VAL_OCR_DRAM=$(extract_val "ocr.demand_data_rd.dram")
VAL_OFFCORE_RD=$(extract_val "offcore_requests.demand_data_rd")
VAL_OFFCORE_RFO=$(extract_val "offcore_requests.demand_rfo")
VAL_L3_MISS_DRAM=$(extract_val "mem_load_l3_miss_retired.local_dram")

if [ -n "$VAL_CAS_RD" ] || [ -n "$VAL_CAS_WR" ]; then
    # Each CAS command transfers 64 bytes
    # BW (GB/s) = count * 64 / (task-clock-ms / 1000) / 1e9
    if [ -n "$VAL_CAS_RD" ] && [ -n "$TASK_CLOCK" ] && [ "$TASK_CLOCK" != "0" ]; then
        BW_READ=$(awk "BEGIN { printf \"%.2f\", ($VAL_CAS_RD * 64) / ($TASK_CLOCK / 1000) / 1e9 }")
        echo "  CAS read count:     $VAL_CAS_RD"
        echo "  BW_read (GB/s):     $BW_READ"
    else
        echo "  CAS read count:     N/A"
    fi

    if [ -n "$VAL_CAS_WR" ] && [ -n "$TASK_CLOCK" ] && [ "$TASK_CLOCK" != "0" ]; then
        BW_WRITE=$(awk "BEGIN { printf \"%.2f\", ($VAL_CAS_WR * 64) / ($TASK_CLOCK / 1000) / 1e9 }")
        echo "  CAS write count:    $VAL_CAS_WR"
        echo "  BW_write (GB/s):    $BW_WRITE"
    else
        echo "  CAS write count:    N/A"
    fi

    if [ -n "${BW_READ:-}" ] && [ -n "${BW_WRITE:-}" ]; then
        BW_TOTAL=$(awk "BEGIN { printf \"%.2f\", ${BW_READ} + ${BW_WRITE} }")
        echo "  BW_total (GB/s):    $BW_TOTAL"
    fi
fi

echo ""
echo "--- Core-level DRAM indicators (fallback / cross-check) ---"
echo "  ocr.demand_data_rd.dram:              ${VAL_OCR_DRAM:-N/A}"
echo "  offcore_requests.demand_data_rd:      ${VAL_OFFCORE_RD:-N/A}"
echo "  offcore_requests.demand_rfo:          ${VAL_OFFCORE_RFO:-N/A}"
echo "  mem_load_l3_miss_retired.local_dram:  ${VAL_L3_MISS_DRAM:-N/A}"
echo ""

echo "--- What to watch ---"
echo "  Total DRAM BW close to platform max → memory-bandwidth saturated"
echo "  For this i9-14900HX: DDR5 dual-channel ≈ ~80-90 GB/s theoretical max"
echo "  LLM decode is heavily BW-bound: each token reads all weight parameters"
echo "  High BW utilization during decode → well-optimized memory access pattern"
echo "  Low BW utilization → possible compute bottleneck or poor data locality"
echo "  Compare serial vs interleaved: if interleaved achieves higher"
echo "    total BW, it exploits spare DRAM capacity from concurrent models"
