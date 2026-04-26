#!/usr/bin/env bash
# Comprehensive performance analysis — runs all perf metrics in one shot.
# Combines: backend stalls, branch prediction, cache hierarchy, TLB, memory BW, SIMD.
#
# Usage: ./perf_all.sh [demo] [config] [prompt]

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "================================================================"
echo " MNN Performance Analysis — All-in-One"
echo "================================================================"
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

# ---------------------------------------------------------------------------
# Build event list from what's available
# Cache perf list output once to avoid SIGPIPE issues with pipefail
# ---------------------------------------------------------------------------
PERF_LIST=$(perf list 2>/dev/null || true)

EVENTS="cycles,instructions,task-clock,duration_time"

# Backend stalls
[[ "$PERF_LIST" == *"stalled-cycles-backend"* ]] && EVENTS="$EVENTS,stalled-cycles-backend"
[[ "$PERF_LIST" == *"stalled-cycles-frontend"* ]] && EVENTS="$EVENTS,stalled-cycles-frontend"

# Branch
[[ "$PERF_LIST" == *"branches"* ]] && EVENTS="$EVENTS,branches"
[[ "$PERF_LIST" == *"branch-misses"* ]] && EVENTS="$EVENTS,branch-misses"

# Cache L1
[[ "$PERF_LIST" == *"l1-dcache-loads"* ]] && EVENTS="$EVENTS,l1-dcache-loads"
[[ "$PERF_LIST" == *"l1-dcache-load-misses"* ]] && EVENTS="$EVENTS,l1-dcache-load-misses"
[[ "$PERF_LIST" == *"l1-dcache-stores"* ]] && EVENTS="$EVENTS,l1-dcache-stores"
[[ "$PERF_LIST" == *"l1-icache-loads"* ]] && EVENTS="$EVENTS,l1-icache-loads"
[[ "$PERF_LIST" == *"l1-icache-load-misses"* ]] && EVENTS="$EVENTS,l1-icache-load-misses"

# Cache LLC
[[ "$PERF_LIST" == *"llc-loads"* ]] && EVENTS="$EVENTS,llc-loads"
[[ "$PERF_LIST" == *"llc-load-misses"* ]] && EVENTS="$EVENTS,llc-load-misses"
[[ "$PERF_LIST" == *"llc-stores"* ]] && EVENTS="$EVENTS,llc-stores"
[[ "$PERF_LIST" == *"llc-store-misses"* ]] && EVENTS="$EVENTS,llc-store-misses"

# TLB
[[ "$PERF_LIST" == *"dtlb-loads"* ]] && EVENTS="$EVENTS,dtlb-loads"
[[ "$PERF_LIST" == *"dtlb-load-misses"* ]] && EVENTS="$EVENTS,dtlb-load-misses"
[[ "$PERF_LIST" == *"dtlb-stores"* ]] && EVENTS="$EVENTS,dtlb-stores"
[[ "$PERF_LIST" == *"dtlb-store-misses"* ]] && EVENTS="$EVENTS,dtlb-store-misses"
[[ "$PERF_LIST" == *"itlb-load-misses"* ]] && EVENTS="$EVENTS,itlb-load-misses"

# SIMD
[[ "$PERF_LIST" == *"fp_arith_inst_retired.scalar"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.scalar"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.vector"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.vector"

# Memory
[[ "$PERF_LIST" == *"cache-misses"* ]] && EVENTS="$EVENTS,cache-misses"
[[ "$PERF_LIST" == *"cache-references"* ]] && EVENTS="$EVENTS,cache-references"

echo "  events: $EVENTS"
echo ""

# ---------------------------------------------------------------------------
# Run perf stat — single invocation, all events
# ---------------------------------------------------------------------------
TMPFILE=$(mktemp)
RAW_FILE=$(mktemp --tmpdir mnn_perf_all_XXXXXX.csv)
trap "rm -f $TMPFILE" EXIT

echo "Running perf stat (this may take a while)..."
echo ""

perf stat -x, -e "$EVENTS" -o "$TMPFILE" -- "$DEMO" "$CONFIG" "$PROMPT"

# Save raw output for reference
cp "$TMPFILE" "$RAW_FILE"

# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------
extract_val() {
    local event="$1" file="${2:-$TMPFILE}"
    # Match: exact name (e.g., duration_time) or /event/ suffix (e.g., cpu_atom/cycles/)
    awk -F, -v ev="$event" '
        $3 == ev || $3 ~ ("/" ev "/$") {
            if ($1 ~ /^[0-9]/) { gsub(/ /, "", $1); val += $1 }
        }
        END { if (val > 0) printf "%d", val; else print "" }
    ' "$file"
}

extract_val_float() {
    local event="$1" file="${2:-$TMPFILE}"
    awk -F, -v ev="$event" '
        $3 == ev || $3 ~ ("/" ev "/$") {
            if ($1 ~ /^[0-9]/) { gsub(/ /, "", $1); val += $1 }
        }
        END { if (val > 0) print val; else print "" }
    ' "$file"
}

compute_ratio() {
    local num="$1" den="$2"
    if [ -n "$num" ] && [ -n "$den" ] && [ "$den" != "0" ]; then
        awk "BEGIN { printf \"%.2f\", ($num / $den) * 100 }"
    else
        echo ""
    fi
}

compute_ratio_raw() {
    local num="$1" den="$2"
    if [ -n "$num" ] && [ -n "$den" ] && [ "$den" != "0" ]; then
        awk "BEGIN { printf \"%.4f\", ($num / $den) }"
    else
        echo ""
    fi
}

safe_val() {
    local val="$1" default="${2:-N/A}"
    if [ -n "$val" ]; then
        echo "$val"
    else
        echo "$default"
    fi
}

# ---------------------------------------------------------------------------
# Extract counters
# ---------------------------------------------------------------------------
VAL_CYCLES=$(extract_val "cycles")
VAL_INSTRUCTIONS=$(extract_val "instructions")
VAL_WALL_NS=$(extract_val "duration_time")
VAL_TASK_CLOCK=$(extract_val_float "task-clock")

VAL_STALL_BACKEND=$(extract_val "stalled-cycles-backend")
VAL_STALL_FRONTEND=$(extract_val "stalled-cycles-frontend")
VAL_BRANCHES=$(extract_val "branches")
VAL_BRANCH_MISSES=$(extract_val "branch-misses")

VAL_L1D_LOADS=$(extract_val "l1-dcache-loads")
VAL_L1D_LOAD_MISSES=$(extract_val "l1-dcache-load-misses")
VAL_L1D_STORES=$(extract_val "l1-dcache-stores")
VAL_L1I_LOADS=$(extract_val "l1-icache-loads")
VAL_L1I_LOAD_MISSES=$(extract_val "l1-icache-load-misses")

VAL_LLC_LOADS=$(extract_val "llc-loads")
VAL_LLC_LOAD_MISSES=$(extract_val "llc-load-misses")
VAL_LLC_STORES=$(extract_val "llc-stores")
VAL_LLC_STORE_MISSES=$(extract_val "llc-store-misses")

VAL_DTLB_LOADS=$(extract_val "dtlb-loads")
VAL_DTLB_LOAD_MISSES=$(extract_val "dtlb-load-misses")
VAL_DTLB_STORES=$(extract_val "dtlb-stores")
VAL_DTLB_STORE_MISSES=$(extract_val "dtlb-store-misses")
VAL_ITLB_LOAD_MISSES=$(extract_val "itlb-load-misses")

VAL_FP_SCALAR=$(extract_val "fp_arith_inst_retired.scalar")
VAL_FP_VECTOR=$(extract_val "fp_arith_inst_retired.vector")

VAL_CACHE_MISSES=$(extract_val "cache-misses")
VAL_CACHE_REFS=$(extract_val "cache-references")

# ---------------------------------------------------------------------------
# Computed metrics
# ---------------------------------------------------------------------------
IPC=$(compute_ratio_raw "$VAL_INSTRUCTIONS" "$VAL_CYCLES")
IPC_FMT=$(awk "BEGIN { printf \"%.2f\", $IPC }" 2>/dev/null || echo "N/A")

# Wall time in seconds
WALL_SEC=""
if [ -n "$VAL_WALL_NS" ]; then
    WALL_SEC=$(awk "BEGIN { printf \"%.3f\", $VAL_WALL_NS / 1000000000 }" 2>/dev/null)
fi

# Pipeline ratios
BACKEND_STALL_PCT=$(compute_ratio "$VAL_STALL_BACKEND" "$VAL_CYCLES")
FRONTEND_STALL_PCT=$(compute_ratio "$VAL_STALL_FRONTEND" "$VAL_CYCLES")
BRANCH_MISS_RATE=$(compute_ratio "$VAL_BRANCH_MISSES" "$VAL_BRANCHES")

# Cache ratios
L1D_MISS_RATE=$(compute_ratio "$VAL_L1D_LOAD_MISSES" "$VAL_L1D_LOADS")
L1I_MISS_RATE=$(compute_ratio "$VAL_L1I_LOAD_MISSES" "$VAL_L1I_LOADS")
LLC_LOAD_MISS_RATE=$(compute_ratio "$VAL_LLC_LOAD_MISSES" "$VAL_LLC_LOADS")
LLC_STORE_MISS_RATE=$(compute_ratio "$VAL_LLC_STORE_MISSES" "$VAL_LLC_STORES")

# TLB ratios
DTLB_LOAD_MISS_RATE=$(compute_ratio "$VAL_DTLB_LOAD_MISSES" "$VAL_DTLB_LOADS")
DTLB_STORE_MISS_RATE=$(compute_ratio "$VAL_DTLB_STORE_MISSES" "$VAL_DTLB_STORES")

# SIMD ratio
FP_VECTOR_RATIO=""
if [ -n "$VAL_FP_SCALAR" ] && [ -n "$VAL_FP_VECTOR" ]; then
    FP_TOTAL=$(awk "BEGIN { print $VAL_FP_SCALAR + $VAL_FP_VECTOR }" 2>/dev/null)
    if [ -n "$FP_TOTAL" ] && [ "$FP_TOTAL" != "0" ]; then
        FP_VECTOR_RATIO=$(awk "BEGIN { printf \"%.1f\", ($VAL_FP_VECTOR / $FP_TOTAL) * 100 }" 2>/dev/null)
    fi
fi

# ---------------------------------------------------------------------------
# Bottleneck heuristic
# ---------------------------------------------------------------------------
BOTTLENECK=""
BOTTLENECK_DETAIL=""
if [ -n "$BACKEND_STALL_PCT" ]; then
    BACKEND_PCT=$(awk "BEGIN { printf \"%.0f\", $BACKEND_STALL_PCT }" 2>/dev/null)
    if [ "$BACKEND_PCT" -gt 50 ] 2>/dev/null; then
        BOTTLENECK="Backend Bound (memory-bound LLM inference)"
        BOTTLENECK_DETAIL="backend stall = ${BACKEND_STALL_PCT}%"
    fi
fi
if [ -z "$BOTTLENECK" ] && [ -n "$FRONTEND_STALL_PCT" ]; then
    FRONTEND_PCT=$(awk "BEGIN { printf \"%.0f\", $FRONTEND_STALL_PCT }" 2>/dev/null)
    if [ "$FRONTEND_PCT" -gt 30 ] 2>/dev/null; then
        BOTTLENECK="Frontend Bound (instruction fetch issues)"
        BOTTLENECK_DETAIL="frontend stall = ${FRONTEND_STALL_PCT}%"
    fi
fi
if [ -z "$BOTTLENECK" ] && [ -n "$BRANCH_MISS_RATE" ]; then
    BRMISS_PCT=$(awk "BEGIN { printf \"%.1f\", $BRANCH_MISS_RATE }" 2>/dev/null)
    if [ "${BRMISS_PCT%.*}" -gt 5 ] 2>/dev/null; then
        BOTTLENECK="Bad Speculation (branch misprediction)"
        BOTTLENECK_DETAIL="branch miss rate = ${BRANCH_MISS_RATE}%"
    fi
fi
if [ -z "$BOTTLENECK" ] && [ -n "$IPC" ] && [ "$IPC" != "N/A" ]; then
    IPC_VAL=$(awk "BEGIN { printf \"%.0f\", $IPC }" 2>/dev/null)
    if [ "$IPC_VAL" -gt 2 ] 2>/dev/null; then
        BOTTLENECK="Well-optimized (compute-bound)"
        BOTTLENECK_DETAIL="IPC = ${IPC_FMT}"
    fi
fi
if [ -z "$BOTTLENECK" ]; then
    BOTTLENECK="Mixed (no single dominant bottleneck)"
    BOTTLENECK_DETAIL=""
fi

# ---------------------------------------------------------------------------
# Format helpers for display
# ---------------------------------------------------------------------------
fmt_pct() {
    local val="$1" default="${2:-N/A}"
    if [ -n "$val" ]; then
        echo "${val}%"
    else
        echo "$default"
    fi
}

fmt_int() {
    local val="$1" default="${2:-N/A}"
    if [ -n "$val" ]; then
        printf "%'d" "$val" 2>/dev/null || echo "$val"
    else
        echo "$default"
    fi
}

# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------
echo ""
echo "+===========================================================+"
echo "|              MNN Performance Summary                      |"
echo "+===========================================================+"
echo "| Runtime                                                  |"
echo "|   Wall time:          $(safe_val "$WALL_SEC") s                           |"
echo "|   Instructions:       $(fmt_int "$VAL_INSTRUCTIONS")                      |"
echo "|   Cycles:             $(fmt_int "$VAL_CYCLES")                            |"
echo "|   IPC:                ${IPC_FMT}                                |"
echo "+-----------------------------------------------------------+"
echo "| Pipeline                                                 |"
echo "|   Frontend stall:     $(fmt_pct "$FRONTEND_STALL_PCT")                              |"
echo "|   Backend stall:      $(fmt_pct "$BACKEND_STALL_PCT")                              |"
echo "|   Branch miss rate:   $(fmt_pct "$BRANCH_MISS_RATE")                              |"
echo "+-----------------------------------------------------------+"
echo "| Cache Hierarchy                                          |"
echo "|   L1D miss rate:      $(fmt_pct "$L1D_MISS_RATE")                              |"
echo "|   L1I miss rate:      $(fmt_pct "$L1I_MISS_RATE")                              |"
echo "|   LLC load miss rate: $(fmt_pct "$LLC_LOAD_MISS_RATE")                              |"
echo "|   LLC store miss rate:$(fmt_pct "$LLC_STORE_MISS_RATE")                              |"
echo "+-----------------------------------------------------------+"
echo "| TLB                                                      |"
echo "|   dTLB load miss rate:  $(fmt_pct "$DTLB_LOAD_MISS_RATE")                              |"
echo "|   dTLB store miss rate: $(fmt_pct "$DTLB_STORE_MISS_RATE")                              |"
echo "+-----------------------------------------------------------+"
echo "| Vectorization                                            |"
echo "|   FP vector ratio:    $(fmt_pct "$FP_VECTOR_RATIO")                              |"
echo "+-----------------------------------------------------------+"
echo "| Bottleneck:  ${BOTTLENECK} |"
echo "+-----------------------------------------------------------+"
echo ""
echo "Raw perf data saved to: $RAW_FILE"
echo ""
echo "--- Quick Reference ---"
echo "  IPC = instructions / cycles (higher is better, >2 is good for CPU)"
echo "  Backend stall = stalled-cycles-backend / cycles (memory/compute bound)"
echo "  Frontend stall = stalled-cycles-frontend / cycles (fetch/decode issues)"
echo "  Miss rate = misses / accesses (lower is better)"
echo "  FP vector ratio = vector instructions / total FP instructions"
echo ""
echo "For deeper dives:"
echo "  ./.build/perf_topdown.sh    — Top-down microarchitecture breakdown"
echo "  ./.build/perf_cache.sh      — Cache hierarchy detail"
echo "  ./.build/perf_tlb.sh        — TLB pressure analysis"
echo "  ./.build/perf_branch.sh     — Branch prediction efficiency"
echo "  ./.build/perf_backend.sh    — Backend stall analysis"
