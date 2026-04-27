#!/usr/bin/env bash
# Wrapper: run perf scripts with sudo when perf_event_paranoid > 0.
# Usage: ./.build/sudo_perf.sh <script> [args...]
# Example: ./.build/sudo_perf.sh ./.build/perf_cache.sh

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <perf_script> [args...]"
    echo "  e.g. $0 ./.build/perf_cache.sh"
    exit 1
fi

SCRIPT="$1"
shift

PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 2)

if [ "$PARANOID" -gt 0 ] 2>/dev/null; then
    echo "perf_event_paranoid=$PARANOID, using sudo..."
    exec sudo "$SCRIPT" "$@"
else
    exec "$SCRIPT" "$@"
fi
