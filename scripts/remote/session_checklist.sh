#!/bin/sh
# Final remote checklist. It reports state and never deletes run files.
set -u
RUN_ID=${1:-}
case "$RUN_ID" in
    ''|*[!A-Za-z0-9._-]*) printf 'invalid run_id\n' >&2; exit 64 ;;
esac
LOG_DIR=${HEXAGONFLOW_REMOTE_LOG_ROOT:-"/tmp/hexagonflow/$RUN_ID"}
QDC_LOG_DIR=${HEXAGONFLOW_QDC_LOG_ROOT:-"/qdc/logs/hexagonflow/$RUN_ID"}
printf 'run_id=%s\nremote_log_dir=%s\n' "$RUN_ID" "$LOG_DIR"
if [ -d "$LOG_DIR" ]; then
    find "$LOG_DIR" -maxdepth 2 -type f -print 2>/dev/null | sort
else
    printf 'remote_logs=missing\n'
fi
if [ -d "$QDC_LOG_DIR" ]; then
    printf 'qdc_log_backup=present\n'
    find "$QDC_LOG_DIR" -maxdepth 2 -type f -print \
        2>/dev/null | sort
else
    printf 'qdc_log_backup=not_available\n'
fi
