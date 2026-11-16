#include "hexagonflow/types.hpp"

namespace hexagonflow {

std::string to_string(WorkloadKind value) {
    switch (value) {
        case WorkloadKind::LlmPrefill: return "llm_prefill";
        case WorkloadKind::LlmDecode: return "llm_decode";
        case WorkloadKind::Asr: return "asr";
        case WorkloadKind::Embedding: return "embedding";
        case WorkloadKind::Custom: return "custom";
    }
    return "unknown";
}

std::string to_string(TaskState value) {
    switch (value) {
        case TaskState::Queued: return "queued";
        case TaskState::Dispatched: return "dispatched";
        case TaskState::Completed: return "completed";
        case TaskState::Failed: return "failed";
        case TaskState::Cancelled: return "cancelled";
    }
    return "unknown";
}

std::string to_string(BackendFamily value) {
    switch (value) {
        case BackendFamily::Mock: return "mock";
        case BackendFamily::GenieX: return "geniex";
        case BackendFamily::QnnQairt: return "qnn_qairt";
        case BackendFamily::LlamaCppHexagon: return "llama_cpp_hexagon";
        case BackendFamily::Cpu: return "cpu";
        case BackendFamily::Gpu: return "gpu";
    }
    return "unknown";
}

}  // namespace hexagonflow
