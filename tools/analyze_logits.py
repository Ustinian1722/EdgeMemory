#!/usr/bin/env python3
"""Compare truncated GenieX logits while preserving the limits of top-k evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.hexagonflow import BundleError, _workspace_path


def compare(cpu: dict, npu: dict, cpu_config: dict, npu_config: dict) -> dict:
    for key in ('seed', 'prompt_tokens', 'model_sha256'):
        if key not in cpu_config or cpu_config.get(key) != npu_config.get(key):
            raise ValueError('input conditions do not match: ' + key)
    for key in ('n_prompt', 'vocab_size', 'top_n'):
        if cpu.get(key) != npu.get(key):
            raise ValueError('logits metadata mismatch: ' + key)
    if cpu.get('device') != 'cpu' or npu.get('device') != 'npu':
        raise ValueError('expected CPU and NPU result files')
    a, b = cpu['rows'][0], npu['rows'][0]
    if not a or not b:
        raise ValueError('empty logits rows')
    left = {int(token): float(value) for token, value in a}
    right = {int(token): float(value) for token, value in b}
    if len(left) != len(a) or len(right) != len(b):
        raise ValueError('duplicate token IDs')
    shared = sorted(left.keys() & right.keys())
    delta = [abs(left[token] - right[token]) for token in shared]
    return {'seed': cpu_config['seed'], 'model_sha256': cpu_config['model_sha256'],
            'prompt_tokens': cpu_config['prompt_tokens'],
            'cpu_top1_token': int(a[0][0]), 'npu_top1_token': int(b[0][0]),
            'top1_match': int(a[0][0]) == int(b[0][0]),
            'top_k': cpu['top_n'], 'shared_token_count': len(shared),
            'shared_mean_absolute_logit_difference': sum(delta) / len(delta) if delta else None,
            'shared_max_absolute_logit_difference': max(delta) if delta else None,
            'full_vocab_comparison': False,
            'limitation': 'Differences only cover tokens appearing in both truncated top-k outputs.'}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('cpu_run', type=Path)
    parser.add_argument('npu_run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        cpu = _workspace_path(args.cpu_run)
        npu = _workspace_path(args.npu_run)
        output = _workspace_path(args.output)
        read = lambda p: json.loads(p.read_text(encoding='utf-8'))
        result = compare(read(cpu / 'logits.json'), read(npu / 'logits.json'),
                         read(cpu / 'config.json'), read(npu / 'config.json'))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    except (ValueError, KeyError, IndexError, OSError, BundleError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
