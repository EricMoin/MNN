#!/usr/bin/env bash
# Top-down Microarchitecture Analysis (Intel Skylake+).
# Breaks down pipeline slots into: Retiring / Bad Speculation / Frontend Bound / Backend Bound.
#
# Usage: ./perf_topdown.sh [demo] [config] [prompt]

set -euo pipefail

DEMO="${1:-./llm_demo}"
CONFIG="${2:-$HOME/Project/models/qwen2.5/config-inter.json}"
PROMPT="${3:-prompt.txt}"

echo "=== Top-down Analysis ==="
echo "  demo:   $DEMO"
echo "  config: $CONFIG"
echo "  prompt: $PROMPT"
echo ""

perf stat --topdown -a -- "$DEMO" "$CONFIG" "$PROMPT"

echo ""
echo "--- What to watch ---"
echo "  Retiring        — slots with useful work (higher = better)"
echo "  Bad Speculation — wasted on mispredicted branches"
echo "  Frontend Bound  — stalled waiting for instructions to fetch/decode"
echo "  Backend Bound   — stalled waiting for data (memory or execution units)"
echo ""
echo "  For LLM inference, Backend Bound is the main bottleneck."
echo "  If interleaved has lower Backend Bound than serial, the two models"
echo "  overlap their memory stalls — a genuine scheduling win."
