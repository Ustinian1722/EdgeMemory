#!/usr/bin/env python3
"""Create or verify a private run's SHA256 inventory under the D: workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tools.hexagonflow import BundleError, _workspace_path

MANIFEST = 'integrity-manifest.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def create(run: Path) -> dict:
    run = _workspace_path(run)
    if not run.is_dir():
        raise ValueError('run directory does not exist')
    entries = []
    for path in sorted(run.rglob('*')):
        if path.is_symlink():
            raise ValueError('run contains a symbolic link')
        if path.is_file() and path.name != MANIFEST:
            entries.append({'path': path.relative_to(run).as_posix(), 'bytes': path.stat().st_size,
                            'sha256': digest(path)})
    if not entries:
        raise ValueError('refusing to manifest an empty run')
    result = {'schema_version': 1, 'run_id': run.name,
              'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'files': entries}
    temp = run / (MANIFEST + '.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temp.replace(run / MANIFEST)
    return result


def verify(run: Path) -> dict:
    run = _workspace_path(run)
    manifest = json.loads((run / MANIFEST).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('run_id') != run.name:
        raise ValueError('manifest identity or version mismatch')
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest has no files')
    expected = set()
    for item in entries:
        relative = Path(item['path'])
        if relative.is_absolute() or not relative.parts or any(p in ('', '.', '..') for p in relative.parts):
            raise ValueError('unsafe manifest path')
        unresolved = run / relative
        if unresolved.is_symlink():
            raise ValueError('symbolic link in run')
        target = unresolved.resolve()
        if not target.is_relative_to(run) or not target.is_file():
            raise ValueError('missing or unsafe run file')
        if relative.as_posix() in expected:
            raise ValueError('duplicate manifest path')
        expected.add(relative.as_posix())
        if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
            raise ValueError('run file hash or size mismatch: ' + relative.as_posix())
    paths = list(run.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('symbolic link in run')
    actual = {p.relative_to(run).as_posix() for p in paths if p.is_file() and p.name != MANIFEST}
    if actual != expected:
        raise ValueError('run file set changed after manifest creation')
    return {'run_id': run.name, 'verified_files': len(entries), 'status': 'verified'}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create', 'verify'])
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    try:
        result = create(args.run) if args.action == 'create' else verify(args.run)
    except (BundleError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + '\n')
    status = result.get('status') or ('created' if args.action == 'create' else 'verified')
    print(json.dumps({'run_id': result['run_id'],
                      'files': len(result['files']) if 'files' in result else result['verified_files'],
                      'status': status}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
