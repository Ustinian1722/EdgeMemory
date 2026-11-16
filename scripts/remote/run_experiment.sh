#!/bin/sh
# Run one experiment with an explicit wall-clock bound and preserve output.
set -u
RUN_ID=${1:-}
TIMEOUT_SEC=${2:-}
shift 2 2>/dev/null || true

case "$RUN_ID" in
    ''|*[!A-Za-z0-9._-]*) printf 'invalid run_id\n' >&2; exit 64 ;;
esac
case "$TIMEOUT_SEC" in
    ''|*[!0-9]*) printf 'timeout must be an integer\n' >&2; exit 64 ;;
esac
if [ "$TIMEOUT_SEC" -lt 1 ] || [ "$TIMEOUT_SEC" -gt 900 ]; then
    printf 'timeout must be in 1..900 seconds\n' >&2
    exit 64
fi
if [ "$#" -eq 0 ]; then printf 'command required\n' >&2; exit 64; fi

LOG_DIR=${HEXAGONFLOW_REMOTE_LOG_ROOT:-"/tmp/hexagonflow/$RUN_ID"}
mkdir -p "$LOG_DIR"
START_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || printf 'unknown')
printf 'run_id=%s\nstart_utc=%s\ntimeout_sec=%s\n' \
    "$RUN_ID" "$START_UTC" "$TIMEOUT_SEC" >"$LOG_DIR/experiment-metadata.txt"

if ! command -v timeout >/dev/null 2>&1; then
    printf 'timeout utility missing; refusing unbounded experiment\n' \
        | tee "$LOG_DIR/experiment-error.log" >&2
    printf 'status=blocked_timeout_unavailable\n' \
        >>"$LOG_DIR/experiment-metadata.txt"
    exit 69
fi
timeout --signal=TERM --kill-after=5s "${TIMEOUT_SEC}s" "$@" \
    >"$LOG_DIR/experiment.stdout.log" 2>"$LOG_DIR/experiment.stderr.log"
STATUS=$?
END_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || printf 'unknown')
printf 'end_utc=%s\nexit_code=%s\n' "$END_UTC" "$STATUS" \
    >>"$LOG_DIR/experiment-metadata.txt"

QDC_LOG_ROOT=${HEXAGONFLOW_QDC_LOG_ROOT:-"/qdc/logs/hexagonflow/$RUN_ID"}
if mkdir -p "$QDC_LOG_ROOT" 2>/dev/null && [ -w "$QDC_LOG_ROOT" ]; then
    cp -R "$LOG_DIR/." "$QDC_LOG_ROOT/"
    printf 'qdc_log_backup=available\n' >>"$LOG_DIR/experiment-metadata.txt"
else
    printf 'qdc_log_backup=unavailable; retain downloaded host copy\n' \
        >>"$LOG_DIR/experiment-metadata.txt"
fi
exit "$STATUS"
