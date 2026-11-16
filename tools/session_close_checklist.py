#!/usr/bin/env python3
"""Seal local run directories and check readiness for a manual QDC session close."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import run_manifest, session_budget

ROOT = Path(__file__).resolve().parents[1]


def _inside_workspace(path: Path) -> Path:
    value = path if path.is_absolute() else ROOT / path
    value = value.resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError("path must stay inside the D: project workspace")
    return value


def seal_run_tree(runs_root: Path) -> dict[str, Any]:
    root = _inside_workspace(runs_root)
    if not root.is_dir():
        raise ValueError("runs directory does not exist")
    verified = []
    created = []
    empty = []
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            raise ValueError(f"symlink is not allowed in runs: {child.name}")
        if not child.is_dir():
            continue
        if child.name == "__pycache__":
            continue
        if not any(path.is_file() for path in child.rglob("*")):
            empty.append(child.name)
            continue
        manifest = child / run_manifest.MANIFEST
        if manifest.exists():
            run_manifest.verify(child)
        else:
            run_manifest.create(child)
            created.append(child.name)
        verified.append(child.name)
    return {"runs_root": root.relative_to(ROOT).as_posix(),
            "verified_run_count": len(verified), "created_manifest_run_ids": created,
            "empty_run_ids": empty, "verified_run_ids": verified}


def _result_status(run: Path) -> str | None:
    result_path = run / "result.json"
    if not result_path.is_file():
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return "invalid-result-json"
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    exit_code = value.get("exit_code", value.get("host_runner_exit_code",
                                                   config.get("exit_code")))
    status = value.get("request_status", value.get("request_a_status"))
    if exit_code not in (None, 0) or config.get("exit_code", 0) not in (None, 0) or (status is not None and status not in
            ("process-completed", "completed", "success", "ok")):
        return "completed-with-error-evidence"
    return "completed"


def create_checklist(runs_root: Path, session_run_id: str,
                     remote_check: dict[str, Any], budget: dict[str, Any],
                     evaluated_utc: str | None = None) -> dict[str, Any]:
    root = _inside_workspace(runs_root)
    if not isinstance(session_run_id, str) or not session_run_id or "/" in session_run_id or "\\" in session_run_id:
        raise ValueError("invalid session run ID")
    session_dir = (root / session_run_id).resolve()
    if not session_dir.is_relative_to(root) or not session_dir.is_dir():
        raise ValueError("session run directory is missing")

    run_statuses = []
    integrity_errors = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "__pycache__":
            continue
        try:
            verified = run_manifest.verify(child)
            run_statuses.append({"run_id": child.name,
                                 "integrity": "verified",
                                 "files": verified["verified_files"],
                                 "result_status": _result_status(child)})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            integrity_errors.append({"run_id": child.name,
                                     "error_type": type(error).__name__})

    active_count = remote_check.get("active_relevant_processes")
    remote_ready = remote_check.get("schema_version") == 1 and active_count == 0
    remaining = budget.get("conservative_remaining_seconds")
    budget_ready = isinstance(remaining, int) and remaining > 0
    integrity_ready = not integrity_errors and bool(run_statuses)
    return {
        "schema_version": 1,
        "session_run_id": session_run_id,
        "evaluated_utc": evaluated_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ready_for_user_to_close": integrity_ready and remote_ready and budget_ready,
        "qdc_session_closed_by_tool": False,
        "checks": {
            "local_run_manifests_verified": integrity_ready,
            "remote_experiment_processes_quiescent": remote_ready,
            "local_budget_record_available": budget_ready,
            "connection_parameters_stored": False,
            "private_key_logged": False,
        },
        "conservative_remaining_seconds": remaining if budget_ready else None,
        "verified_runs": run_statuses,
        "integrity_errors": integrity_errors,
        "failed_runs_with_collected_evidence": [
            item["run_id"] for item in run_statuses
            if item["result_status"] == "completed-with-error-evidence"
        ],
        "remote_check_source": "read_only_process_count_only",
        "manual_action": "If ready_for_user_to_close is true, the user may close the QDC web session.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal", help="create missing integrity manifests and verify every run directory")
    seal.add_argument("--runs-root", type=Path, default=Path("runs"))
    seal_check = sub.add_parser("check", help="evaluate local evidence plus a sanitized remote process-count record")
    seal_check.add_argument("--runs-root", type=Path, default=Path("runs"))
    seal_check.add_argument("--session-run-id", required=True)
    seal_check.add_argument("--remote-check", type=Path, required=True)
    seal_check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "seal":
            result = seal_run_tree(args.runs_root)
            print(json.dumps(result, indent=2))
            return 0 if not result["empty_run_ids"] else 2
        remote_path = _inside_workspace(args.remote_check)
        output_path = _inside_workspace(args.output)
        remote = json.loads(remote_path.read_text(encoding="utf-8"))
        budget = session_budget.status(args.session_run_id)
        result = create_checklist(args.runs_root, args.session_run_id, remote, budget)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            session_budget.SessionError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"output": output_path.relative_to(ROOT).as_posix(),
                      "ready_for_user_to_close": result["ready_for_user_to_close"],
                      "verified_run_count": len(result["verified_runs"]),
                      "integrity_error_count": len(result["integrity_errors"])}))
    return 0 if result["ready_for_user_to_close"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
