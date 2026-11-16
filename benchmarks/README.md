# IQ-9075 device measurements

These are curated observations from separate device runs, not a new benchmark execution.

| Measurement | Result | Scope |
| --- | ---: | --- |
| Tears of Steel source | 734.167 s | 720p source |
| Corrected indexing wall time | 967.759 s | 0.759× source real time |
| Indexed events | 196 | 195 captions, 196 visual vectors, 0 indexing failures in this run |
| LLM evidence Q&A | 3.498 s | Qwen3-4B Q4_0, HTP4, single request |
| TTFT | 722.738 ms | Same request |
| Prefill | 369.755 tokens/s | Same request |
| Decode | 15.958 tokens/s | Same request |
| CPU answer | 46.683 s | Separate backend with differing thread/output configuration |
| Manual VLM retry | 5.119 s | HTP0, one linked retry |

The corrected run supersedes an earlier run with corrupted BGR frame layout. A successful event count is not a measure of semantic accuracy. The CPU comparison is not a controlled universal speedup. Earlier VLM and QIM tests had different inputs and should not be pooled with these measurements.

Raw private device logs, databases and session records are excluded from this source distribution. The [summary JSON](iq9075-summary.json) carries the measured values without private paths or credentials. Reproduction requires a compatible target and the external assets described in the quickstart; no fresh hardware benchmark was run for this release candidate.
