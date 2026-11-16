#!/bin/sh
# Read-only target inventory: no installs, root requests, or hardware assumptions.
set -u

section() { printf '\n=== %s ===\n' "$1"; }

section "identity"
date -u '+utc=%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || true
printf 'hostname='
hostname 2>/dev/null || printf 'unavailable\n'
printf 'uid='
id 2>/dev/null || printf 'unavailable\n'

section "os-and-abi"
uname -a 2>/dev/null || true
if [ -r /etc/os-release ]; then cat /etc/os-release; fi
printf 'machine='
uname -m 2>/dev/null || printf 'unavailable\n'
printf 'word_bits='
getconf LONG_BIT 2>/dev/null || printf 'unavailable\n'
printf 'glibc='
getconf GNU_LIBC_VERSION 2>/dev/null || printf 'not-reported-by-getconf\n'

section "cpu-memory-storage"
HOME_DIR=${HOME:-}
if command -v nproc >/dev/null 2>&1; then
    printf 'logical_cpus='
    nproc 2>/dev/null || true
fi
if command -v lscpu >/dev/null 2>&1; then
    lscpu 2>/dev/null | sed -n '1,24p'
elif [ -r /proc/cpuinfo ]; then
    sed -n '1,40p' /proc/cpuinfo
fi
if command -v free >/dev/null 2>&1; then free -b 2>/dev/null || true; fi
if command -v df >/dev/null 2>&1; then
    for path in / /tmp /var/tmp "$HOME_DIR" /qdc/logs; do
        if [ -e "$path" ]; then df -Pk "$path" 2>/dev/null || true; fi
    done
fi
for path in /tmp /var/tmp "$HOME_DIR" /qdc/logs; do
    if [ -e "$path" ]; then
        if [ -w "$path" ]; then printf 'writable=%s\n' "$path"
        else printf 'not_writable=%s\n' "$path"; fi
    fi
done

section "available-tools"
for tool in sh bash python3 adb ssh scp sftp timeout sha256sum tar gzip \
            gcc g++ c++ clang++ cmake make ninja readelf file ldd \
            qnn-net-run qnn-profile-viewer genie-t2t-run geniex llama-cli qpm-cli; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%s=%s\n' "$tool" "$(command -v "$tool")"
    else
        printf '%s=missing\n' "$tool"
    fi
done

section "known-runtime-help-smoke"
HELP_LOG_ROOT=${HEXAGONFLOW_REMOTE_LOG_ROOT:-"/tmp/hexagonflow/device-probe"}
if ! mkdir -p "$HELP_LOG_ROOT" 2>/dev/null; then
    printf 'runtime_help_status=skipped_log_directory_unavailable\n'
elif command -v timeout >/dev/null 2>&1; then
    for tool in qnn-net-run genie-t2t-run geniex llama-cli; do
        executable=$(command -v "$tool" 2>/dev/null || true)
        if [ -z "$executable" ]; then continue; fi
        help_log="$HELP_LOG_ROOT/runtime-help-$tool.log"
        printf 'runtime_help_tool=%s\n' "$tool"
        timeout 5s "$executable" --help >"$help_log" 2>&1
        help_status=$?
        case "$help_status" in
            0) printf 'runtime_help_status=%s\n' "$tool:completed" ;;
            124|137) printf 'runtime_help_status=%s\n' "$tool:timed_out" ;;
            *) printf 'runtime_help_status=%s\n' "$tool:exit_$help_status" ;;
        esac
        sed -n '1,24p' "$help_log" 2>/dev/null || true
    done
else
    printf 'runtime_help_status=skipped_timeout_utility_missing\n'
fi

section "geniex-runtime-version"
if command -v geniex >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    geniex_version_log="$HELP_LOG_ROOT/runtime-version-geniex.log"
    timeout 5s geniex --version >"$geniex_version_log" 2>&1
    geniex_version_status=$?
    printf 'geniex_version_probe_exit=%s\n' "$geniex_version_status"
    sed -n '1,24p' "$geniex_version_log" 2>/dev/null || true
else
    printf 'geniex_version_probe_status=unavailable\n'
fi

section "graphics-and-accelerator-nodes"
for path in /dev/kgsl-3d0 /dev/dri /dev/fastrpc* /dev/adsprpc* \
            /dev/cdsp* /dev/ion /dev/dma_heap/*; do
    if [ -e "$path" ]; then ls -ld "$path" 2>/dev/null || true; fi
done
printf 'device_nodes_matching_fastrpc_or_dsp:\n'
if command -v find >/dev/null 2>&1; then
    find /dev -maxdepth 3 \( -iname '*fastrpc*' -o -iname '*adsprpc*' \
        -o -iname '*cdsp*' -o -iname '*dsp*' -o -iname '*htp*' \) \
        -print 2>/dev/null | head -n 100
fi

section "qairt_qnn_libraries"
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -Ei 'lib(Qnn|cdsprpc|adsprpc|OpenCL)' \
        | head -n 100 || true
fi
if command -v find >/dev/null 2>&1; then
    for root in /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 \
                /lib /lib64 /opt/qualcomm /opt/qcom /vendor/lib64; do
        if [ -d "$root" ]; then
            find "$root" -maxdepth 3 -type f \
                \( -iname 'libQnn*.so*' -o -iname '*fastrpc*.so*' \
                   -o -iname '*adsprpc*.so*' -o -iname 'libOpenCL.so*' \) \
                -print 2>/dev/null | head -n 100
        fi
    done
fi

section "voice-ai-asr-assets"
for root in /opt/qcom/qpm/VoiceAI_ASR \
            /opt/genai-studio-models/speech-to-text \
            /opt/ASR_binary/Whisper; do
    if [ -e "$root" ]; then ls -ld "$root" 2>/dev/null || true; fi
done
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -Ei 'VoiceAI|Whisper|nnvad' \
        | head -n 60 || true
fi
if command -v find >/dev/null 2>&1; then
    for root in /opt/qcom/qpm/VoiceAI_ASR \
                /opt/genai-studio-models/speech-to-text \
                /opt/ASR_binary/Whisper; do
        if [ -d "$root" ]; then
            find "$root" -maxdepth 7 -type f \
                \( -name 'libnnvad_model.so' -o -name 'encoder.bin' \
                   -o -name 'decoder.bin' -o -name 'vocab.bin' \
                   -o -name '*.gguf' -o -name '*.dlc' -o -name '*.onnx' \) \
                -print 2>/dev/null | head -n 100
        fi
    done
fi

section "soc-identifiers"
for path in /sys/devices/soc0/soc_id /sys/devices/soc0/machine \
            /proc/device-tree/model /proc/device-tree/compatible; do
    if [ -r "$path" ]; then
        printf '%s=' "$path"
        tr '\000' ' ' < "$path" 2>/dev/null
        printf '\n'
    fi
done

section "probe-complete"
printf 'probe_mode=read_only\n'
