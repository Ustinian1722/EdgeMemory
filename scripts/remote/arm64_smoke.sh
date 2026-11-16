#!/bin/sh
# Compile and run an ABI-only test if the device already has a C++ compiler.
# No package install, accelerator load, or privileged operation is attempted.
set -eu
RUN_ID=${1:-}
case "$RUN_ID" in
    ''|*[!A-Za-z0-9._-]*) printf 'invalid run_id\n' >&2; exit 64 ;;
esac
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUNDLE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
LOG_DIR=${HEXAGONFLOW_REMOTE_LOG_ROOT:-"/tmp/hexagonflow/$RUN_ID"}
mkdir -p "$LOG_DIR"
MACHINE=$(uname -m 2>/dev/null || printf 'unknown')
if [ "$MACHINE" != "aarch64" ]; then
    printf 'status=skipped_wrong_arch\nruntime_machine=%s\n' "$MACHINE" \
        >"$LOG_DIR/arm64-smoke-status.txt"
    exit 0
fi
COMPILER=
for tool in c++ g++ clang++; do
    if command -v "$tool" >/dev/null 2>&1; then COMPILER=$(command -v "$tool"); break; fi
done
if [ -z "$COMPILER" ]; then
    PREBUILT="$BUNDLE_ROOT/build/arm64-smoke"
    if [ ! -x "$PREBUILT" ]; then
        printf 'status=blocked_no_compiler_or_prebuilt_smoke\nruntime_machine=%s\n' "$MACHINE" \
            >"$LOG_DIR/arm64-smoke-status.txt"
        exit 0
    fi
    if "$PREBUILT" >"$LOG_DIR/arm64-smoke.log" 2>&1; then
        printf 'status=passed_prebuilt\nruntime_machine=%s\ncompiler=not-required\n' "$MACHINE" \
            >"$LOG_DIR/arm64-smoke-status.txt"
    else
        status=$?
        printf 'status=failed_prebuilt\nruntime_machine=%s\nexit_code=%s\n' "$MACHINE" "$status" \
            >"$LOG_DIR/arm64-smoke-status.txt"
        exit "$status"
    fi
    cat "$LOG_DIR/arm64-smoke-status.txt"
    exit 0
fi
OUTPUT="$LOG_DIR/hexagonflow-arm64-smoke"
{
    printf 'compiler=%s\n' "$COMPILER"
    "$COMPILER" -std=c++17 -O2 -Wall -Wextra -pedantic \
        "$BUNDLE_ROOT/src/arm64_smoke.cpp" -o "$OUTPUT"
    if command -v readelf >/dev/null 2>&1; then
        readelf -h "$OUTPUT" | sed -n '1,24p'
        readelf -l "$OUTPUT" | grep -E 'Requesting program interpreter|INTERP' || true
    fi
    if command -v ldd >/dev/null 2>&1; then ldd "$OUTPUT" || true; fi
    "$OUTPUT"
} >"$LOG_DIR/arm64-smoke.log" 2>&1
printf 'status=passed\nruntime_machine=%s\ncompiler=%s\n' "$MACHINE" "$COMPILER" \
    >"$LOG_DIR/arm64-smoke-status.txt"
cat "$LOG_DIR/arm64-smoke-status.txt"
