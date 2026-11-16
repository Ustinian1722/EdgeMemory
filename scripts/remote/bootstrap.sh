#!/bin/sh
# Files on the QDC device are disposable. No installs or privilege escalation.
set -eu
RUN_ID=${1:-}
case "$RUN_ID" in
    ''|*[!A-Za-z0-9._-]*) printf 'invalid run_id\n' >&2; exit 64 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUNDLE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
LOCAL_LOG_ROOT=${HEXAGONFLOW_REMOTE_LOG_ROOT:-"/tmp/hexagonflow/$RUN_ID"}
mkdir -p "$LOCAL_LOG_ROOT"

if ! command -v sha256sum >/dev/null 2>&1; then
    printf 'sha256sum missing; refusing unchecked bootstrap\n' \
        | tee "$LOCAL_LOG_ROOT/bootstrap-error.log" >&2
    exit 69
fi
if [ ! -f "$BUNDLE_ROOT/manifest.sha256" ]; then
    printf 'manifest.sha256 missing\n' | tee "$LOCAL_LOG_ROOT/bootstrap-error.log" >&2
    exit 65
fi
(
    cd "$BUNDLE_ROOT"
    sha256sum -c manifest.sha256
) >"$LOCAL_LOG_ROOT/manifest-check.log" 2>&1 || {
    cat "$LOCAL_LOG_ROOT/manifest-check.log" >&2
    printf 'bundle checksum verification failed\n' \
        | tee "$LOCAL_LOG_ROOT/bootstrap-error.log" >&2
    exit 65
}

"$SCRIPT_DIR/device_probe.sh" >"$LOCAL_LOG_ROOT/device-probe.txt" 2>&1
printf 'run_id=%s\nprobe_status=complete\n' "$RUN_ID" \
    >"$LOCAL_LOG_ROOT/bootstrap-status.txt"
printf 'bundle_root=%s\n' "$BUNDLE_ROOT" \
    >>"$LOCAL_LOG_ROOT/bootstrap-status.txt"

QDC_LOG_ROOT=${HEXAGONFLOW_QDC_LOG_ROOT:-"/qdc/logs/hexagonflow/$RUN_ID"}
if mkdir -p "$QDC_LOG_ROOT" 2>/dev/null && [ -w "$QDC_LOG_ROOT" ]; then
    cp -R "$LOCAL_LOG_ROOT/." "$QDC_LOG_ROOT/"
    printf 'qdc_log_backup=available\n' >>"$LOCAL_LOG_ROOT/bootstrap-status.txt"
else
    printf 'qdc_log_backup=unavailable; keep host-side copy\n' \
        >>"$LOCAL_LOG_ROOT/bootstrap-status.txt"
fi
cat "$LOCAL_LOG_ROOT/bootstrap-status.txt"
