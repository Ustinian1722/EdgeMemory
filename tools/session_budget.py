#!/usr/bin/env python3
"""Conservative local countdown from a user-observed QDC remaining time."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tools.qdc_session import SessionError, checked_run_id, run_directory


def initialize(run_id: str, remaining_minutes: int, buffer_minutes: int) -> dict:
    if not 1 <= remaining_minutes <= 1440 or not 0 <= buffer_minutes < remaining_minutes:
        raise SessionError('invalid observed remaining time or safety buffer')
    root = run_directory(checked_run_id(run_id))
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'observed-budget.json'
    if target.exists():
        raise SessionError('budget observation already exists; use a new run ID for a new observation')
    state = {'run_id': run_id, 'source': 'user-reported QDC page',
             'reported_remaining_minutes': remaining_minutes,
             'safety_buffer_minutes': buffer_minutes,
             'baseline_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
             'baseline_conservative_seconds': (remaining_minutes - buffer_minutes) * 60}
    target.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    return state


def phase_for_seconds(seconds: int) -> str:
    if seconds <= 10 * 60:
        return 'stop_remote_and_notify'
    if seconds <= 20 * 60:
        return 'collect_and_verify'
    if seconds <= 45 * 60:
        return 'finish_existing_experiments'
    return 'research_window'


def status(run_id: str) -> dict:
    path = run_directory(checked_run_id(run_id)) / 'observed-budget.json'
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
        baseline = datetime.fromisoformat(state['baseline_utc'])
        now = datetime.now(timezone.utc)
        elapsed = int((now - baseline).total_seconds())
        if elapsed < 0:
            raise ValueError('clock moved backwards')
        seconds = max(0, int(state['baseline_conservative_seconds']) - elapsed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SessionError('missing or invalid budget observation') from error
    return {'run_id': run_id, 'conservative_remaining_seconds': seconds,
            'phase': phase_for_seconds(seconds), 'source': state['source'],
            'as_of_utc': now.isoformat(timespec='seconds')}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('observe')
    init.add_argument('--run-id', required=True)
    init.add_argument('--remaining-minutes', type=int, required=True)
    init.add_argument('--buffer-minutes', type=int, default=0)
    check = sub.add_parser('status')
    check.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        result = (initialize(args.run_id, args.remaining_minutes, args.buffer_minutes)
                  if args.command == 'observe' else status(args.run_id))
    except SessionError as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
