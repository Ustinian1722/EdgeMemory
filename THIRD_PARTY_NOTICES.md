# Third-party notices

The EdgeMemory source tree does not vendor or link third-party source code. The following projects were inspected in a Git-ignored, local-only source cache under `third_party/research-cache/`; those snapshots are not part of this repository or its release artifacts.

| Project | Inspected revision | Upstream license | Use in HexagonFlow |
| --- | --- | --- | --- |
| [Qualcomm GenieX](https://github.com/qualcomm/GenieX) | `eddb76f36597928b79d8ec4e0e14e65c8dcaaaf0` | BSD-3-Clause | Source/API and IQ9 device mapping research only |
| [GenieX-QAIRT-plugin](https://github.com/qualcomm/geniex-qairt-plugin) | `e9ab7d700300d0aa339910a2abb1cc32c3ed3643` | Plugin code BSD-3-Clause; bundled QNN headers/runtime have separate Qualcomm terms | Source research only; restricted headers and binaries were not copied |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `0fb8f157702702f33f4e018de8701b2569a1bc28` | MIT, with third-party notices in the project | Backend/device design research only |
| [Qualcomm AIMET](https://github.com/qualcomm/aimet) | `16c9ddd0f307e9e1895c5497dafbb46880f07bb9` | BSD-3-Clause | Quantization-tool research only |
| [Qualcomm AI Hub Apps](https://github.com/qualcomm/ai-hub-apps) | `e6477cfa238acaffe01ed45de4cfcbd53abbc99d` | BSD-3-Clause; model/API assets may have separate terms | Workflow/tutorial research only |
| [Alibaba MNN](https://github.com/alibaba/MNN) | `ff0250ef9b5dc21fbf7b188e63f4ac1711afa0a1` | Apache-2.0 | QNN adapter architecture research only |
| [QCS9075 llama.cpp community reference](https://github.com/r11922135/qcs9075-llamacpp) | `5a9867a4ec17523522e1c9fd6aaaec3acc9e960d` | No root license file found at inspection | Unlicensed reference; no source or binaries reused |

The revisions and the distinction between source-code licenses, SDK terms, model terms, and AI Hub service terms were documented during development. Each upstream repository remains the authoritative source for its license. A later release that copies, links, or distributes any upstream artifact must re-evaluate its current license, notices, and redistribution conditions.
