# Quickstart

## Local development

Requirements: Python 3.12, FFmpeg/ffprobe on PATH for host video ingestion, CMake 3.22+, a C++20 compiler, and Ninja on Linux. For real inference, start separately configured local OpenAI-compatible vision and text endpoints first; EdgeMemory does not start those models for you.

From the repository root:

```sh
python -m edgememory.cli init
python -m edgememory.cli serve \
  --caption-backend openai-compatible \
  --caption-url http://127.0.0.1:18181/v1 \
  --caption-model '<your-vlm-model-id>' \
  --answer-backend openai-compatible \
  --answer-url http://127.0.0.1:18182/v1 \
  --answer-model '<your-llm-model-id>'
```

Open http://127.0.0.1:8765. This host development route is distinct from the full device-resident CLIP/GenieX application. Use the CLI help for available embedding options. Do not expose the workbench to untrusted networks.

## IQ-9075 device path

Validated environment: Qualcomm Linux 1.9, GenieX v0.11.0, QNN SDK 2.43. You need a compatible target, a user-managed QDC session with an active SSH connection, host Python, an ARM64 build environment, sufficient storage, and authorized access to the model/runtime assets. See `configs/iq9075-board-assets.json` for pinned assets and their terms.

Create an ignored local SSH profile under `.local/edgememory/qdc-profile-4b-live.json`. Use actual values provided by the QDC Connect page, and a confirmed writable absolute remote directory. Example schema (replace placeholders; port is illustrative only):

```json
{
  "schema_version": 1,
  "transport": "ssh",
  "remote_root": "/path/to/writable/device-directory",
  "ssh": {
    "host": "replace-with-qdc-host",
    "port": 22,
    "username": "replace_with_qdc_username"
  }
}
```

Never place passwords, keys or session tokens in the profile or Git. If required, use the optional absolute local `identity_file` path.

```sh
python tools/qdc_device.py --profile .local/edgememory/qdc-profile-4b-live.json --devices HTP0 --execution-mode direct --video-workers 1 --pending-videos 2 --restore-geniex-runtime --restore-vlm-gguf --restore-llm-gguf --llm-backend auto --llm-device-id HTP4 --llm-geniex-context 512 --llm-geniex-max-tokens 96 --restore-minilm --restore-openai-clip --start-timeout 300 session
```

The helper requests asset restoration, launches the service, and forwards the HTTP port. The complete CLIP restoration path has not been independently verified on a fresh session; this command describes the intended deployment, not a guaranteed one-command installation. Inspect CLI help and device logs if startup fails. Keep the QDC session open while collecting results; the helper does not provision a cloud session.

The ARM64 ABI smoke step compiles `src/arm64_smoke.cpp` on the target when a C++ compiler is available. If the target has no compiler, the script uses an optional locally built `build/arm64-smoke` binary; source-only bundles omit that ignored build artifact and report the smoke step as blocked when neither route is available.

## Tests

```sh
cmake --preset wsl-gcc
cmake --build --preset wsl-gcc
ctest --preset wsl-gcc
```

The Windows preset is `windows-msvc`. Host simulation or mock outputs must not be reported as device benchmarks.
