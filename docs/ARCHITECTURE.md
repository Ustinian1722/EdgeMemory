# Architecture

EdgeMemory separates ingestion from interactive retrieval. The video sampler uses V4L2 hardware H.264 decoding and QIM GLES transformation; the appsink maps selected BGR frames, using actual GstVideoMeta stride/offset. JPEG evidence feeds the resident VLM. CLIP computes model-scoped visual vectors; MiniLM embeds captions. SQLite stores events, evidence and source timestamps.

Text queries use both embedding spaces; image queries use the CLIP image encoder. Rank fusion combines ranked results, not incomparable raw vectors. The current exact-search implementation targets a modest local corpus. An optional VLM review revisits original evidence before the independent LLM produces cited answers. Evidence batches and bounded prompts fit the validated 512-token LLM context; summaries report sampled coverage.

| Model | Runtime | Role |
| --- | --- | --- |
| Qwen3-VL-4B Q4_0 + projector | GenieX HTP0 | Caption, review |
| Qwen3-4B-Instruct-2507 Q4_0 | GenieX HTP4 / CPU fallback | Q&A, summary |
| CLIP ViT-B/32 ONNX | ONNX Runtime CPU | 512D image/text |
| MiniLM-L6-v2 Q8 | llama.cpp CPU | 384D text |

Direct execution is the primary device path. Persistent workers avoid repeated loading; HTP requests share a conservative resource lock and bounded waiting. The legacy HexagonFlow C++ runtime is included for development and scheduling experiments, not required as a novel inference engine.

## Code entry points

- `tools/remote/gstreamer_video_sampler.py`: video sampling, buffer telemetry, stride handling.
- `edgememory/device_agent.py`: device ingestion, retrieval, evidence review and HTTP APIs.
- `edgememory/device_store.py`: persistence and model-scoped search.
- `edgememory/visual_embeddings.py`: paired CLIP adapters.
- `edgememory/direct_execution.py`: admission and fallback.
- `src/geniex_vlm_backend.cpp`, `src/geniex_llm_backend.cpp`: native adapters.

## Engineering boundaries

The validated path is not end-to-end zero-copy. Physical HTP independence and GPU CLIP inference are unverified. Manual event retry succeeded; exhaustive automatic recovery and long-duration concurrent operation are not established. The browser workflow is implemented, but comprehensive browser acceptance has not been completed.
