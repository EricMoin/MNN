#!/usr/bin/env bash
# SIMD / vectorization utilization metrics.
# Measures FP instruction breakdown: scalar vs vector, SSE vs AVX split,
# FLOPS estimate, and SSE↔AVX transition penalties.
#
# Usage: ./perf_simd.sh [demo] [config] [prompt]
# Default: ./llm_demo ~/Project/models/qwen2.5/config-inter.json prompt.txt

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== SIMD / Vectorization Analysis ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

# ---------------------------------------------------------------------------
# Build event list from what's available
# ---------------------------------------------------------------------------
PERF_LIST=$(perf list 2>/dev/null || true)

EVENTS="instructions"
HAS_VECTOR_DIRECT=0

[[ "$PERF_LIST" == *"fp_arith_inst_retired.scalar"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.scalar"
if [[ "$PERF_LIST" == *"fp_arith_inst_retired.vector"* ]]; then
    EVENTS="$EVENTS,fp_arith_inst_retired.vector"
    HAS_VECTOR_DIRECT=1
fi
[[ "$PERF_LIST" == *"fp_arith_inst_retired.128b_packed_single"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.128b_packed_single"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.128b_packed_double"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.128b_packed_double"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.256b_packed_single"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.256b_packed_single"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.256b_packed_double"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.256b_packed_double"
[[ "$PERF_LIST" == *"fp_arith_inst_retired.4_flops"* ]] && EVENTS="$EVENTS,fp_arith_inst_retired.4_flops"
[[ "$PERF_LIST" == *"assists.sse_avx_mix"* ]] && EVENTS="$EVENTS,assists.sse_avx_mix"

echo "  events: $EVENTS"
echo "  Note: Intel PMU has ~4-8 general-purpose counters. With many events,"
echo "        perf may multiplex — estimates are approximate but useful."
echo ""

# ---------------------------------------------------------------------------
# Run perf stat, capturing to a temp file (perf stat outputs to stderr,
# -x, gives machine-parseable CSV)
# ---------------------------------------------------------------------------
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

perf stat -x, -e "$EVENTS" -o "$TMPFILE" -- "$DEMO" "$CONFIG" "$PROMPT"

# ---------------------------------------------------------------------------
# Extract counter value for a given event from the CSV output
# ---------------------------------------------------------------------------
extract_val() {
    local event="$1"
    awk -F, -v ev="$event" '
        ($3 == ev || $3 ~ ("/" ev "/$")) && $1 ~ /^[0-9]/ {
            gsub(/ /, "", $1); val += $1
        }
        END { if (val > 0) printf "%d", val; else print "" }
    ' "$TMPFILE"
}

# ---------------------------------------------------------------------------
# Compute a ratio percentage safely
# ---------------------------------------------------------------------------
compute_ratio() {
    local num="$1" den="$2"
    if [ -n "$num" ] && [ -n "$den" ] && [ "$den" != "0" ] && [ "$den" != "0.0" ]; then
        awk "BEGIN { printf \"%.2f%%\", ($num / $den) * 100 }"
    else
        echo "N/A"
    fi
}

# ---------------------------------------------------------------------------
# Extract all counters
# ---------------------------------------------------------------------------
SCALAR=$(extract_val "fp_arith_inst_retired.scalar")
VECTOR=$(extract_val "fp_arith_inst_retired.vector")
P128_SINGLE=$(extract_val "fp_arith_inst_retired.128b_packed_single")
P128_DOUBLE=$(extract_val "fp_arith_inst_retired.128b_packed_double")
P256_SINGLE=$(extract_val "fp_arith_inst_retired.256b_packed_single")
P256_DOUBLE=$(extract_val "fp_arith_inst_retired.256b_packed_double")
F4_FLOPS=$(extract_val "fp_arith_inst_retired.4_flops")
SSE_AVX_MIX=$(extract_val "assists.sse_avx_mix")
TOTAL_INSNS=$(extract_val "instructions")

# If vector counter not directly available, compute from packed sub-events
if [ "$HAS_VECTOR_DIRECT" = "0" ]; then
    VECTOR_COMPUTED="0"
    for v in "$P128_SINGLE" "$P128_DOUBLE" "$P256_SINGLE" "$P256_DOUBLE"; do
        if [ -n "$v" ] && [ "$v" != "0" ] 2>/dev/null; then
            VECTOR_COMPUTED=$(awk "BEGIN { printf \"%.0f\", $VECTOR_COMPUTED + $v }")
        fi
    done
    VECTOR="$VECTOR_COMPUTED"
fi

# ---------------------------------------------------------------------------
# Computed metrics
# ---------------------------------------------------------------------------
VECTOR_RATIO=$(compute_ratio "$VECTOR" "$(awk "BEGIN { print ($SCALAR + 0) + ($VECTOR + 0) }")")
TOTAL_128B=$(awk "BEGIN { print (${P128_SINGLE:-0}) + (${P128_DOUBLE:-0}) }")
TOTAL_256B=$(awk "BEGIN { print (${P256_SINGLE:-0}) + (${P256_DOUBLE:-0}) }")
SSE_RATIO=$(compute_ratio "$TOTAL_128B" "$(awk "BEGIN { print ($TOTAL_128B + 0) + ($TOTAL_256B + 0) }")")
AVX_RATIO=$(compute_ratio "$TOTAL_256B" "$(awk "BEGIN { print ($TOTAL_128B + 0) + ($TOTAL_256B + 0) }")")

# FLOPS estimate: scalar=1, 128b_single=4, 128b_double=2, 256b_single=8, 256b_double=4
FLOPS_EST=$(awk "BEGIN { printf \"%.0f\", \
    (${SCALAR:-0})*1 + \
    (${P128_SINGLE:-0})*4 + (${P128_DOUBLE:-0})*2 + \
    (${P256_SINGLE:-0})*8 + (${P256_DOUBLE:-0})*4 }")

SSE_AVX_RATE="N/A"
if [ -n "$SSE_AVX_MIX" ] && [ -n "$TOTAL_INSNS" ] && [ "$TOTAL_INSNS" != "0" ] 2>/dev/null; then
    SSE_AVX_RATE=$(awk "BEGIN { printf \"%.4f%%\", (${SSE_AVX_MIX} / ${TOTAL_INSNS}) * 100 }")
fi

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
echo ""
echo "=== Results ==="
echo ""

printf "  %-50s %s\n" "Total instructions" "${TOTAL_INSNS:-N/A}"
echo ""
echo "  -- FP Instruction Breakdown --"
printf "  %-50s %s\n" "fp_arith_inst_retired.scalar (unvectorized)" "${SCALAR:-N/A}"
printf "  %-50s %s\n" "fp_arith_inst_retired.vector (any width)" "${VECTOR:-N/A}"
printf "  %-50s %s\n" "fp_arith_inst_retired.128b_packed_single (SSE)" "${P128_SINGLE:-N/A}"
printf "  %-50s %s\n" "fp_arith_inst_retired.128b_packed_double (SSE)" "${P128_DOUBLE:-N/A}"
printf "  %-50s %s\n" "fp_arith_inst_retired.256b_packed_single (AVX)" "${P256_SINGLE:-N/A}"
printf "  %-50s %s\n" "fp_arith_inst_retired.256b_packed_double (AVX)" "${P256_DOUBLE:-N/A}"
printf "  %-50s %s\n" "fp_arith_inst_retired.4_flops" "${F4_FLOPS:-N/A}"
printf "  %-50s %s\n" "assists.sse_avx_mix (transition penalties)" "${SSE_AVX_MIX:-N/A}"
echo ""
echo "  -- Computed Metrics --"
printf "  %-50s %s\n" "Vector ratio (vector/(scalar+vector))" "$VECTOR_RATIO"
printf "  %-50s %s\n" "SSE share (128b / (128b+256b))" "$SSE_RATIO"
printf "  %-50s %s\n" "AVX share (256b / (128b+256b))" "$AVX_RATIO"
printf "  %-50s %s\n" "Estimated FLOPS" "$FLOPS_EST"
printf "  %-50s %s\n" "SSE↔AVX assist rate" "$SSE_AVX_RATE"

echo ""
echo "--- What to watch ---"
echo "  Vector ratio >90%   → excellent vectorization"
echo "  Vector ratio <50%   → significant scalar fallback, optimization target"
echo "  SSE↔AVX assists >0  → mixing SSE and AVX code paths causes expensive"
echo "                         state transitions (vzeroupper penalties). Should be zero."
echo "  256-bit dominant    → AVX2 being effectively used"
echo "  128-bit dominant    → may be falling back to SSE paths"
echo "  For LLM: GEMV/GEMM kernels should be heavily vectorized;"
echo "           the decode/sampling phase may be less so."
echo "  If interleaved preserves vector ratio, the scheduling overhead"
echo "  does not degrade SIMD utilization."
