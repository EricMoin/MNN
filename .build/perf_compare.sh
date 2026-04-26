#!/usr/bin/env bash
# A/B performance comparison.
# Runs two configurations and shows side-by-side metrics with deltas.
#
# Usage: ./perf_compare.sh <config_A> <config_B> [demo] [prompt]
# Example: ./perf_compare.sh config-serial.json config-interleaved.json ./llm_demo prompt.txt

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_A> <config_B> [demo] [prompt]"
    echo ""
    echo "  config_A  — first model config (required)"
    echo "  config_B  — second model config (required)"
    echo "  demo      — path to llm_demo binary (default: ./llm_demo)"
    echo "  prompt    — path to prompt file (default: prompt.txt)"
    echo ""
    echo "Example:"
    echo "  $0 config-serial.json config-interleaved.json ./llm_demo prompt.txt"
    exit 1
fi

CONFIG_A="$1"
CONFIG_B="$2"
DEMO="${3:-./llm_demo}"
PROMPT="${4:-prompt.txt}"

if [ ! -f "$CONFIG_A" ]; then
    echo "ERROR: config_A not found: $CONFIG_A"
    exit 1
fi
if [ ! -f "$CONFIG_B" ]; then
    echo "ERROR: config_B not found: $CONFIG_B"
    exit 1
fi

echo "================================================================"
echo " MNN A/B Performance Comparison"
echo "================================================================"
echo "  Config A:  $CONFIG_A"
echo "  Config B:  $CONFIG_B"
echo "  demo:      $DEMO"
echo "  prompt:    $PROMPT"
echo ""

# ---------------------------------------------------------------------------
# Build event list (cache perf list to avoid SIGPIPE with pipefail)
# ---------------------------------------------------------------------------
PERF_LIST=$(perf list 2>/dev/null || true)

EVENTS="cycles,instructions,task-clock,duration_time"

[[ "$PERF_LIST" == *"stalled-cycles-backend"* ]] && EVENTS="$EVENTS,stalled-cycles-backend"
[[ "$PERF_LIST" == *"stalled-cycles-frontend"* ]] && EVENTS="$EVENTS,stalled-cycles-frontend"
[[ "$PERF_LIST" == *"branches"* ]] && EVENTS="$EVENTS,branches"
[[ "$PERF_LIST" == *"branch-misses"* ]] && EVENTS="$EVENTS,branch-misses"
[[ "$PERF_LIST" == *"l1-dcache-loads"* ]] && EVENTS="$EVENTS,l1-dcache-loads"
[[ "$PERF_LIST" == *"l1-dcache-load-misses"* ]] && EVENTS="$EVENTS,l1-dcache-load-misses"
[[ "$PERF_LIST" == *"l1-dcache-stores"* ]] && EVENTS="$EVENTS,l1-dcache-stores"
[[ "$PERF_LIST" == *"l1-icache-loads"* ]] && EVENTS="$EVENTS,l1-icache-loads"
[[ "$PERF_LIST" == *"l1-icache-load-misses"* ]] && EVENTS="$EVENTS,l1-icache-load-misses"
[[ "$PERF_LIST" == *"llc-loads"* ]] && EVENTS="$EVENTS,llc-loads"
[[ "$PERF_LIST" == *"llc-load-misses"* ]] && EVENTS="$EVENTS,llc-load-misses"
[[ "$PERF_LIST" == *"llc-stores"* ]] && EVENTS="$EVENTS,llc-stores"
[[ "$PERF_LIST" == *"llc-store-misses"* ]] && EVENTS="$EVENTS,llc-store-misses"
[[ "$PERF_LIST" == *"dtlb-loads"* ]] && EVENTS="$EVENTS,dtlb-loads"
[[ "$PERF_LIST" == *"dtlb-load-misses"* ]] && EVENTS="$EVENTS,dtlb-load-misses"
[[ "$PERF_LIST" == *"dtlb-stores"* ]] && EVENTS="$EVENTS,dtlb-stores"
[[ "$PERF_LIST" == *"dtlb-store-misses"* ]] && EVENTS="$EVENTS,dtlb-store-misses"
[[ "$PERF_LIST" == *"itlb-load-misses"* ]] && EVENTS="$EVENTS,itlb-load-misses"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.scalar"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.scalar"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.vector"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.vector"
[[ "$PERF_LIST" == *"cache-misses"* ]] && EVENTS="$EVENTS,cache-misses"
[[ "$PERF_LIST" == *"cache-references"* ]] && EVENTS="$EVENTS,cache-references"

echo "  events: $EVENTS"
echo ""

# ---------------------------------------------------------------------------
# Run config A
# ---------------------------------------------------------------------------
TMPFILE_A=$(mktemp)
trap "rm -f $TMPFILE_A $TMPFILE_B" EXIT

echo "=== Running Config A: $CONFIG_A ==="
perf stat -x, -e "$EVENTS" -o "$TMPFILE_A" -- "$DEMO" "$CONFIG_A" "$PROMPT"
echo ""

# ---------------------------------------------------------------------------
# Run config B
# ---------------------------------------------------------------------------
TMPFILE_B=$(mktemp)
trap "rm -f $TMPFILE_A $TMPFILE_B" EXIT

echo "=== Running Config B: $CONFIG_B ==="
perf stat -x, -e "$EVENTS" -o "$TMPFILE_B" -- "$DEMO" "$CONFIG_B" "$PROMPT"
echo ""

# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------
extract_val() {
    local event="$1" file="$2"
    awk -F, -v ev="$event" '
        $3 == ev || $3 ~ ("/" ev "/$") {
            if ($1 ~ /^[0-9]/) { gsub(/ /, "", $1); val += $1 }
        }
        END { if (val > 0) printf "%d", val; else print "" }
    ' "$file"
}

extract_val_float() {
    local event="$1" file="$2"
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

compute_fp_ratio() {
    local scalar="$1" vector="$2"
    if [ -n "$scalar" ] && [ -n "$vector" ]; then
        local total
        total=$(awk "BEGIN { print $scalar + $vector }" 2>/dev/null)
        if [ -n "$total" ] && [ "$total" != "0" ]; then
            awk "BEGIN { printf \"%.1f\", ($vector / $total) * 100 }"
            return
        fi
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Extract metrics for config A
# ---------------------------------------------------------------------------
get_metrics() {
    local file="$1"
    local cycles instructions wall_ns
    local stall_backend stall_frontend branches branch_misses
    local l1d_loads l1d_load_misses l1i_loads l1i_load_misses
    local llc_loads llc_load_misses llc_store_misses
    local dtlb_loads dtlb_load_misses dtlb_stores dtlb_store_misses
    local fp_scalar fp_vector

    cycles=$(extract_val "cycles" "$file")
    instructions=$(extract_val "instructions" "$file")
    wall_ns=$(extract_val "duration_time" "$file")
    stall_backend=$(extract_val "stalled-cycles-backend" "$file")
    stall_frontend=$(extract_val "stalled-cycles-frontend" "$file")
    branches=$(extract_val "branches" "$file")
    branch_misses=$(extract_val "branch-misses" "$file")
    l1d_loads=$(extract_val "l1-dcache-loads" "$file")
    l1d_load_misses=$(extract_val "l1-dcache-load-misses" "$file")
    l1i_loads=$(extract_val "l1-icache-loads" "$file")
    l1i_load_misses=$(extract_val "l1-icache-load-misses" "$file")
    llc_loads=$(extract_val "llc-loads" "$file")
    llc_load_misses=$(extract_val "llc-load-misses" "$file")
    llc_store_misses=$(extract_val "llc-store-misses" "$file")
    dtlb_loads=$(extract_val "dtlb-loads" "$file")
    dtlb_load_misses=$(extract_val "dtlb-load-misses" "$file")
    dtlb_stores=$(extract_val "dtlb-stores" "$file")
    dtlb_store_misses=$(extract_val "dtlb-store-misses" "$file")
    fp_scalar=$(extract_val "fp_arith_inst_retired.scalar" "$file")
    fp_vector=$(extract_val "fp_arith_inst_retired.vector" "$file")

    local ipc wall_sec backend_pct frontend_pct branch_miss_pct
    local l1d_miss_pct l1i_miss_pct llc_load_miss_pct llc_store_miss_pct
    local dtlb_load_miss_pct dtlb_store_miss_pct fp_vector_pct

    ipc=$(compute_ratio_raw "$instructions" "$cycles")
    wall_sec=""
    if [ -n "$wall_ns" ]; then
        wall_sec=$(awk "BEGIN { printf \"%.3f\", $wall_ns / 1000000000 }" 2>/dev/null)
    fi
    backend_pct=$(compute_ratio "$stall_backend" "$cycles")
    frontend_pct=$(compute_ratio "$stall_frontend" "$cycles")
    branch_miss_pct=$(compute_ratio "$branch_misses" "$branches")
    l1d_miss_pct=$(compute_ratio "$l1d_load_misses" "$l1d_loads")
    l1i_miss_pct=$(compute_ratio "$l1i_load_misses" "$l1i_loads")
    llc_load_miss_pct=$(compute_ratio "$llc_load_misses" "$llc_loads")
    llc_store_miss_pct=$(compute_ratio "$llc_store_misses" "$(extract_val "llc-stores" "$file")")
    dtlb_load_miss_pct=$(compute_ratio "$dtlb_load_misses" "$dtlb_loads")
    dtlb_store_miss_pct=$(compute_ratio "$dtlb_store_misses" "$dtlb_stores")
    fp_vector_pct=$(compute_fp_ratio "$fp_scalar" "$fp_vector")

    printf '%s\n' "$ipc" "$wall_sec" "$backend_pct" "$frontend_pct" "$branch_miss_pct" \
        "$l1d_miss_pct" "$l1i_miss_pct" "$llc_load_miss_pct" "$llc_store_miss_pct" \
        "$dtlb_load_miss_pct" "$dtlb_store_miss_pct" "$fp_vector_pct"
}

echo "Computing metrics..."
mapfile -t METRICS_A < <(get_metrics "$TMPFILE_A")
mapfile -t METRICS_B < <(get_metrics "$TMPFILE_B")

IPC_A="${METRICS_A[0]}"
WALL_A="${METRICS_A[1]}"
BACKEND_A="${METRICS_A[2]}"
FRONTEND_A="${METRICS_A[3]}"
BRANCH_A="${METRICS_A[4]}"
L1D_A="${METRICS_A[5]}"
L1I_A="${METRICS_A[6]}"
LLC_LOAD_A="${METRICS_A[7]}"
LLC_STORE_A="${METRICS_A[8]}"
DTLB_LOAD_A="${METRICS_A[9]}"
DTLB_STORE_A="${METRICS_A[10]}"
FP_A="${METRICS_A[11]}"

IPC_B="${METRICS_B[0]}"
WALL_B="${METRICS_B[1]}"
BACKEND_B="${METRICS_B[2]}"
FRONTEND_B="${METRICS_B[3]}"
BRANCH_B="${METRICS_B[4]}"
L1D_B="${METRICS_B[5]}"
L1I_B="${METRICS_B[6]}"
LLC_LOAD_B="${METRICS_B[7]}"
LLC_STORE_B="${METRICS_B[8]}"
DTLB_LOAD_B="${METRICS_B[9]}"
DTLB_STORE_B="${METRICS_B[10]}"
FP_B="${METRICS_B[11]}"

# ---------------------------------------------------------------------------
# Delta computation helpers
# ---------------------------------------------------------------------------
delta_pct_change() {
    local a="$1" b="$2"
    if [ -n "$a" ] && [ -n "$b" ] && [ "$a" != "0" ]; then
        awk "BEGIN { d = (($b - $a) / $a) * 100; if (d >= 0) printf \"+%.1f%%\", d; else printf \"%.1f%%\", d }"
    else
        echo "N/A"
    fi
}

delta_pp() {
    local a="$1" b="$2"
    if [ -n "$a" ] && [ -n "$b" ]; then
        awk "BEGIN { d = $b - $a; if (d >= 0) printf \"+%.2fpp\", d; else printf \"%.2fpp\", d }"
    else
        echo "N/A"
    fi
}

delta_wall() {
    local a="$1" b="$2"
    if [ -n "$a" ] && [ -n "$b" ] && [ "$a" != "0" ]; then
        awk "BEGIN { d = (($b - $a) / $a) * 100; if (d >= 0) printf \"+%.1f%%\", d; else printf \"%.1f%%\", d }"
    else
        echo "N/A"
    fi
}

fmt_val() {
    local val="$1"
    if [ -n "$val" ]; then
        echo "$val"
    else
        echo "N/A"
    fi
}

fmt_pct_val() {
    local val="$1"
    if [ -n "$val" ]; then
        echo "${val}%"
    else
        echo "N/A"
    fi
}

fmt_ipc() {
    local val="$1"
    if [ -n "$val" ]; then
        awk "BEGIN { printf \"%.2f\", $val }"
    else
        echo "N/A"
    fi
}

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
echo ""
echo "+================================================+===========+===========+===========+"
echo "| Metric                         |  Config A  |  Config B  |  Delta    |"
echo "+================================================+===========+===========+===========+"
printf "| %-30s | %9s | %9s | %9s |\n" "IPC" "$(fmt_ipc "$IPC_A")" "$(fmt_ipc "$IPC_B")" "$(delta_pct_change "$IPC_A" "$IPC_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "Backend stall %%" "$(fmt_pct_val "$BACKEND_A")" "$(fmt_pct_val "$BACKEND_B")" "$(delta_pp "$BACKEND_A" "$BACKEND_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "Frontend stall %%" "$(fmt_pct_val "$FRONTEND_A")" "$(fmt_pct_val "$FRONTEND_B")" "$(delta_pp "$FRONTEND_A" "$FRONTEND_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "Branch miss rate" "$(fmt_pct_val "$BRANCH_A")" "$(fmt_pct_val "$BRANCH_B")" "$(delta_pp "$BRANCH_A" "$BRANCH_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "L1D miss rate" "$(fmt_pct_val "$L1D_A")" "$(fmt_pct_val "$L1D_B")" "$(delta_pp "$L1D_A" "$L1D_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "L1I miss rate" "$(fmt_pct_val "$L1I_A")" "$(fmt_pct_val "$L1I_B")" "$(delta_pp "$L1I_A" "$L1I_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "LLC load miss rate" "$(fmt_pct_val "$LLC_LOAD_A")" "$(fmt_pct_val "$LLC_LOAD_B")" "$(delta_pp "$LLC_LOAD_A" "$LLC_LOAD_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "LLC store miss rate" "$(fmt_pct_val "$LLC_STORE_A")" "$(fmt_pct_val "$LLC_STORE_B")" "$(delta_pp "$LLC_STORE_A" "$LLC_STORE_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "dTLB load miss rate" "$(fmt_pct_val "$DTLB_LOAD_A")" "$(fmt_pct_val "$DTLB_LOAD_B")" "$(delta_pp "$DTLB_LOAD_A" "$DTLB_LOAD_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "dTLB store miss rate" "$(fmt_pct_val "$DTLB_STORE_A")" "$(fmt_pct_val "$DTLB_STORE_B")" "$(delta_pp "$DTLB_STORE_A" "$DTLB_STORE_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "FP vector ratio" "$(fmt_pct_val "$FP_A")" "$(fmt_pct_val "$FP_B")" "$(delta_pp "$FP_A" "$FP_B")"
printf "| %-30s | %9s | %9s | %9s |\n" "Wall time (s)" "$(fmt_val "$WALL_A")" "$(fmt_val "$WALL_B")" "$(delta_wall "$WALL_A" "$WALL_B")"
echo "+================================================+===========+===========+===========+"

# ---------------------------------------------------------------------------
# Winner determination
# ---------------------------------------------------------------------------
echo ""
echo "--- Winner Analysis ---"

WINS_A=0
WINS_B=0

compare_metric() {
    local name="$1" a="$2" b="$3" lower_better="$4"
    if [ -z "$a" ] || [ -z "$b" ]; then
        return
    fi
    local cmp
    cmp=$(awk "BEGIN { if ($a < $b) print \"lt\"; else if ($a > $b) print \"gt\"; else print \"eq\" }")
    if [ "$lower_better" = "true" ]; then
        if [ "$cmp" = "lt" ]; then
            WINS_A=$((WINS_A + 1))
        elif [ "$cmp" = "gt" ]; then
            WINS_B=$((WINS_B + 1))
        fi
    else
        if [ "$cmp" = "gt" ]; then
            WINS_A=$((WINS_A + 1))
        elif [ "$cmp" = "lt" ]; then
            WINS_B=$((WINS_B + 1))
        fi
    fi
}

compare_metric "IPC" "$IPC_A" "$IPC_B" "false"
compare_metric "Wall time" "$WALL_A" "$WALL_B" "true"
compare_metric "Backend stall" "$BACKEND_A" "$BACKEND_B" "true"
compare_metric "Branch miss" "$BRANCH_A" "$BRANCH_B" "true"
compare_metric "L1D miss" "$L1D_A" "$L1D_B" "true"
compare_metric "LLC load miss" "$LLC_LOAD_A" "$LLC_LOAD_B" "true"
compare_metric "dTLB load miss" "$DTLB_LOAD_A" "$DTLB_LOAD_B" "true"

echo "  Config A wins on $WINS_A key metrics, Config B wins on $WINS_B key metrics."

# Determine winner
WINNER=""
WINNER_NAME=""
if [ "$WINS_B" -gt "$WINS_A" ] 2>/dev/null; then
    WINNER="B"
    WINNER_NAME="$CONFIG_B"
elif [ "$WINS_A" -gt "$WINS_B" ] 2>/dev/null; then
    WINNER="A"
    WINNER_NAME="$CONFIG_A"
fi

# Build winner summary bullets — deltas show winner's advantage
BULLETS=""
if [ "$WINNER" = "B" ]; then
    if [ -n "$WALL_A" ] && [ -n "$WALL_B" ]; then
        WALL_DELTA=$(awk "BEGIN { d = (($WALL_A - $WALL_B) / $WALL_A) * 100; printf \"%.1f\", d }")
        if [ -n "$WALL_DELTA" ]; then
            BULLETS="$BULLETS  - ${WALL_DELTA}% faster wall time"
            BULLETS="$BULLETS"$'\n'
        fi
    fi
    if [ -n "$IPC_A" ] && [ -n "$IPC_B" ]; then
        IPC_DELTA=$(awk "BEGIN { d = (($IPC_B - $IPC_A) / $IPC_A) * 100; printf \"%.1f\", d }")
        if [ -n "$IPC_DELTA" ]; then
            BULLETS="$BULLETS  - ${IPC_DELTA}% higher IPC"
            BULLETS="$BULLETS"$'\n'
        fi
    fi
    if [ -n "$BACKEND_A" ] && [ -n "$BACKEND_B" ]; then
        BACKEND_DELTA=$(awk "BEGIN { printf \"%.1f\", $BACKEND_A - $BACKEND_B }")
        if [ -n "$BACKEND_DELTA" ]; then
            BULLETS="$BULLETS  - ${BACKEND_DELTA}pp lower backend stall ratio"
            BULLETS="$BULLETS"$'\n'
        fi
    fi
elif [ "$WINNER" = "A" ]; then
    if [ -n "$WALL_A" ] && [ -n "$WALL_B" ]; then
        WALL_DELTA=$(awk "BEGIN { d = (($WALL_B - $WALL_A) / $WALL_B) * 100; printf \"%.1f\", d }")
        if [ -n "$WALL_DELTA" ]; then
            BULLETS="$BULLETS  - ${WALL_DELTA}% faster wall time"
            BULLETS="$BULLETS"$'\n'
        fi
    fi
    if [ -n "$IPC_A" ] && [ -n "$IPC_B" ]; then
        IPC_DELTA=$(awk "BEGIN { d = (($IPC_A - $IPC_B) / $IPC_B) * 100; printf \"%.1f\", d }")
        if [ -n "$IPC_DELTA" ]; then
            BULLETS="$BULLETS  - ${IPC_DELTA}% higher IPC"
            BULLETS="$BULLETS"$'\n'
        fi
    fi
    if [ -n "$BACKEND_A" ] && [ -n "$BACKEND_B" ]; then
        BACKEND_DELTA=$(awk "BEGIN { printf \"%.1f\", $BACKEND_B - $BACKEND_A }")
        if [ -n "$BACKEND_DELTA" ]; then
            BULLETS="$BULLETS  - ${BACKEND_DELTA}pp lower backend stall ratio"
            BULLETS="$BULLETS"$'\n'
        fi
    fi
fi

if [ -n "$WINNER" ]; then
    echo ""
    echo "Winner: Config $WINNER ($WINNER_NAME)"
    echo "$BULLETS"
else
    echo ""
    echo "Result: Tie — both configurations perform similarly"
fi

echo ""
echo "--- What to watch ---"
echo "  IPC: higher is better (>2 is good for CPU-bound workloads)"
echo "  Stall %%: lower is better (backend = memory/compute bound, frontend = fetch bound)"
echo "  Miss rates: lower is better across all cache/TLB levels"
echo "  FP vector ratio: higher is better (SIMD utilization)"
echo "  Wall time: lower is better (the ultimate measure)"
