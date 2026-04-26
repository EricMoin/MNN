#!/usr/bin/env python3
"""
parse_perf_log.py - Parse [PERF] key=value log output from MNN omni.cpp instrumentation.

Usage:
    python3 parse_perf_log.py perf_log.txt [--summary|--csv|--json]
    cat perf_log.txt | python3 parse_perf_log.py

Output formats:
    --summary (default): Human-readable statistics with per-stage breakdown
    --csv: Comma-separated values for spreadsheet import
    --json: Machine-readable JSON with aggregated statistics
"""

import re
import sys
import json
import argparse
import statistics
from collections import defaultdict


def parse_line(line):
    m = re.match(r'\[PERF\]\s+(.+)', line)
    if not m:
        return None
    content = m.group(1)
    record = {}
    for kv in content.split():
        if '=' in kv:
            k, v = kv.split('=', 1)
            try:
                if '.' in v:
                    record[k] = float(v)
                else:
                    record[k] = int(v)
            except ValueError:
                record[k] = v
    return record


def parse_log(lines):
    records = defaultdict(list)
    all_records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = parse_line(line)
        if rec is None:
            continue
        all_records.append(rec)
        stage = rec.get('stage', 'unknown')
        phase = rec.get('phase', 'unknown')
        records[f"{stage}:{phase}"].append(rec)
    return records, all_records


def compute_stats(values):
    if not values:
        return {}
    return {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values) if len(values) >= 2 else values[0],
        'stddev': statistics.stdev(values) if len(values) >= 2 else 0.0,
        'sum': sum(values),
    }


def print_summary(records, all_records):
    print("=" * 70)
    print("MNN Omni Audio Pipeline Performance Report")
    print("=" * 70)

    stage_metrics = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        stage = rec.get('stage', 'unknown')
        for key in ['dit_time_ms', 'vocoder_time_ms', 'total_time_ms',
                     'postprocess_time_ms', 'flush_time_ms']:
            if key in rec:
                stage_metrics[stage][key].append(rec[key])

    for stage in sorted(stage_metrics.keys()):
        metrics = stage_metrics[stage]
        print(f"\n--- Stage: {stage} ---")
        for metric_name in sorted(metrics.keys()):
            values = metrics[metric_name]
            stats = compute_stats(values)
            if stats:
                print(f"  {metric_name}:")
                print(f"    count={stats['count']} mean={stats['mean']:.3f} "
                      f"median={stats['median']:.3f} stddev={stats['stddev']:.3f} "
                      f"min={stats['min']:.3f} max={stats['max']:.3f}")

    queue_depths = defaultdict(list)
    for rec in all_records:
        for qkey in ['token_queue_depth', 'mel_queue_depth']:
            if qkey in rec:
                queue_depths[qkey].append(rec[qkey])

    if queue_depths:
        print(f"\n--- Queue Depths ---")
        for qkey in sorted(queue_depths.keys()):
            vals = queue_depths[qkey]
            stats = compute_stats(vals)
            print(f"  {qkey}: mean={stats['mean']:.1f} max={stats['max']} min={stats['min']}")

    chunk_timeline = defaultdict(dict)
    for rec in all_records:
        cid = rec.get('chunk_id')
        if cid is None:
            continue
        for key, val in rec.items():
            if key.endswith('_time_ms'):
                chunk_timeline[cid][key] = val

    if chunk_timeline:
        print(f"\n--- Per-Chunk Timeline (top 10) ---")
        for cid in sorted(chunk_timeline.keys())[:10]:
            print(f"  chunk_id={cid}: {chunk_timeline[cid]}")
        if len(chunk_timeline) > 10:
            print(f"  ... and {len(chunk_timeline) - 10} more chunks")

    for rec in all_records:
        if rec.get('phase', '').startswith('clone_'):
            print(f"\n--- Memory: {rec.get('phase')} ---")
            print(f"  rss_kb_before={rec.get('rss_kb_before', 'N/A')}")
            print(f"  rss_kb_after={rec.get('rss_kb_after', 'N/A')}")
            print(f"  delta_kb={rec.get('delta_kb', 'N/A')}")

    for rec in all_records:
        if rec.get('stage') == 'summary':
            print(f"\n--- Final Summary ---")
            for k, v in rec.items():
                if k != 'stage':
                    print(f"  {k}={v}")

    print("\n" + "=" * 70)


def print_csv(records, all_records):
    if not all_records:
        return
    all_keys = sorted(set().union(*(rec.keys() for rec in all_records)))
    print(','.join(all_keys))
    for rec in all_records:
        print(','.join(str(rec.get(k, '')) for k in all_keys))


def print_json(records, all_records):
    output = {
        'total_records': len(all_records),
        'raw_records': all_records,
        'stages': {},
    }
    stage_metrics = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        stage = rec.get('stage', 'unknown')
        for key in ['dit_time_ms', 'vocoder_time_ms', 'total_time_ms',
                     'postprocess_time_ms', 'flush_time_ms']:
            if key in rec:
                stage_metrics[stage][key].append(rec[key])

    for stage in sorted(stage_metrics.keys()):
        output['stages'][stage] = {}
        for metric_name in sorted(stage_metrics[stage].keys()):
            values = stage_metrics[stage][metric_name]
            stats = compute_stats(values)
            if stats:
                output['stages'][stage][metric_name] = stats

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Parse MNN [PERF] log output')
    parser.add_argument('input', nargs='?', help='Input file (or stdin if omitted)')
    parser.add_argument('--summary', action='store_true', default=True,
                        help='Human-readable summary (default)')
    parser.add_argument('--csv', action='store_true', help='CSV output')
    parser.add_argument('--json', action='store_true', help='JSON output')
    args = parser.parse_args()

    if args.csv:
        output_mode = 'csv'
    elif args.json:
        output_mode = 'json'
    else:
        output_mode = 'summary'

    if args.input:
        with open(args.input, 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    records, all_records = parse_log(lines)

    if not all_records:
        print("No [PERF] records found in input.", file=sys.stderr)
        sys.exit(1)

    if output_mode == 'csv':
        print_csv(records, all_records)
    elif output_mode == 'json':
        print_json(records, all_records)
    else:
        print_summary(records, all_records)


if __name__ == '__main__':
    main()
