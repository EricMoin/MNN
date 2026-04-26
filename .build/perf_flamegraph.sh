#!/usr/bin/env bash
# CPU flamegraph via perf record + FlameGraph tools.
# Generates an interactive SVG flamegraph for hotspot analysis.
#
# Usage: ./perf_flamegraph.sh [demo] [config] [prompt] [output_dir]
# Default: ./llm_demo ~/Project/models/qwen2.5/config-inter.json prompt.txt .
#
# Env vars:
#   PERF_FREQ=99       sampling frequency (Hz)
#   PERF_DURATION=     if set, sample for N seconds instead of tracing demo to completion

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"
OUTPUT_DIR="${4:-.}"

FREQ="${PERF_FREQ:-99}"

echo "=== CPU Flamegraph ==="
echo "  demo:       $DEMO"
echo "  config:     $CONFIG"
echo "  prompt:     $PROMPT"
echo "  output dir: $OUTPUT_DIR"
echo "  frequency:  ${FREQ} Hz"
echo ""

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Check perf availability
# ---------------------------------------------------------------------------
if ! command -v perf &>/dev/null; then
    echo "ERROR: 'perf' not found. Install linux-tools or linux-perf."
    exit 1
fi

# ---------------------------------------------------------------------------
# Check FlameGraph tools
# ---------------------------------------------------------------------------
STACKCOLLAPSE=""
FLAMEGRAPH=""

for d in "$HOME/FlameGraph" "/opt/FlameGraph" "/usr/local/FlameGraph"; do
    if [ -x "$d/stackcollapse-perf.pl" ] && [ -x "$d/flamegraph.pl" ]; then
        STACKCOLLAPSE="$d/stackcollapse-perf.pl"
        FLAMEGRAPH="$d/flamegraph.pl"
        break
    fi
done

if [ -z "$STACKCOLLAPSE" ]; then
    if command -v stackcollapse-perf.pl &>/dev/null && command -v flamegraph.pl &>/dev/null; then
        STACKCOLLAPSE="stackcollapse-perf.pl"
        FLAMEGRAPH="flamegraph.pl"
    fi
fi

if [ -z "$STACKCOLLAPSE" ]; then
    echo "ERROR: FlameGraph tools not found."
    echo "  Install them:"
    echo "    git clone --depth=1 https://github.com/brendangregg/FlameGraph.git ~/FlameGraph"
    echo "  Or set PATH to include the directory containing stackcollapse-perf.pl and flamegraph.pl."
    exit 1
fi

echo "  stackcollapse: $STACKCOLLAPSE"
echo "  flamegraph:    $FLAMEGRAPH"
echo ""

# ---------------------------------------------------------------------------
# Run perf record
# ---------------------------------------------------------------------------
PERF_DATA="$OUTPUT_DIR/perf.data"

echo "--- Recording ($PERF_DATA) ---"
echo ""

if [ -n "${PERF_DURATION:-}" ]; then
    echo "  Sampling for ${PERF_DURATION}s in background, running demo..."
    perf record -g --call-graph dwarf -F "$FREQ" -o "$PERF_DATA" -- "$DEMO" "$CONFIG" "$PROMPT" &
    PERF_PID=$!
    sleep "$PERF_DURATION"
    kill "$PERF_PID" 2>/dev/null || true
    wait "$PERF_PID" 2>/dev/null || true
else
    perf record -g --call-graph dwarf -F "$FREQ" -o "$PERF_DATA" -- "$DEMO" "$CONFIG" "$PROMPT"
fi

echo ""
echo "  Recording complete: $PERF_DATA ($(du -h "$PERF_DATA" 2>/dev/null | awk '{print $1}' || echo "?") )"

# ---------------------------------------------------------------------------
# Generate flamegraph SVG
# ---------------------------------------------------------------------------
FLAMEGRAPH_SVG="$OUTPUT_DIR/flamegraph.svg"
echo ""
echo "--- Generating flamegraph ($FLAMEGRAPH_SVG) ---"

perf script -i "$PERF_DATA" | "$STACKCOLLAPSE" | "$FLAMEGRAPH" > "$FLAMEGRAPH_SVG"

echo "  Flamegraph: $FLAMEGRAPH_SVG ($(du -h "$FLAMEGRAPH_SVG" 2>/dev/null | awk '{print $1}' || echo "?"))"

# ---------------------------------------------------------------------------
# Generate reverse (icicle) graph
# ---------------------------------------------------------------------------
REVERSE_SVG="$OUTPUT_DIR/flamegraph_reverse.svg"
echo "--- Generating reverse/icicle graph ($REVERSE_SVG) ---"

perf script -i "$PERF_DATA" | "$STACKCOLLAPSE" | "$FLAMEGRAPH" --reverse > "$REVERSE_SVG"

echo "  Reverse:   $REVERSE_SVG ($(du -h "$REVERSE_SVG" 2>/dev/null | awk '{print $1}' || echo "?"))"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Done ==="
echo ""
echo "  Open in browser:"
echo "    xdg-open $FLAMEGRAPH_SVG"
echo "    xdg-open $REVERSE_SVG"
echo ""
echo "  Keep perf.data for re-analysis:"
echo "    perf script -i $PERF_DATA | stackcollapse-perf.pl | flamegraph.pl --title='Custom' > custom.svg"
echo ""
echo "--- What to watch ---"
echo "  Wide plateaus at top of flamegraph → functions consuming the most CPU"
echo "  For MNN LLM: expect to see GEMM/GEMV kernels, attention computation,"
echo "               memory copy, and framework dispatch overhead"
echo "  Deep call stacks with narrow flames → overhead in framework dispatch"
echo "  Compare flamegraphs before/after optimization to verify hotspot changes"
echo "  The reverse (icicle) graph shows the hottest call paths from root down"
echo "  If interleaved shows wider dispatch/merge overhead but same kernel"
echo "  width, the scheduling cost is in orchestration, not in compute."
