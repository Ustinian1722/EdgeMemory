#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace hexagonflow {

enum class WorkloadKind {
    LlmPrefill,
    LlmDecode,
    Asr,
    Embedding,
    Custom,
};

enum class TaskState {
    Queued,
    Dispatched,
    Completed,
    Failed,
    Cancelled,
};

enum class BackendFamily {
    Mock,
    GenieX,
    QnnQairt,
    LlamaCppHexagon,
    Cpu,
    Gpu,
};

struct Workload {
    std::string id;
    std::string model_id;
    WorkloadKind kind{WorkloadKind::Custom};
    std::int32_t priority{0};
    std::uint64_t arrival_sequence{0};
    std::optional<std::uint64_t> absolute_deadline_us;
    std::uint64_t estimated_service_us{0};
    std::uint64_t model_memory_bytes{0};
    std::uint64_t requested_output_tokens{0};
    // Caller-defined scale for the input to this task kind (for example prompt
    // tokens for LLM work). Zero means unknown and gets its own calibration bucket.
    std::uint64_t input_units{0};
    std::vector<std::string> allowed_backend_ids;
    // Optional prompt and media paths for generative backends. The scheduler
    // treats them as opaque metadata; the selected backend owns interpretation.
    std::string prompt_utf8;
    std::vector<std::string> media_paths;
    // Calibrated per-backend end-to-end service time for this workload class.
    // PredictedLatency refuses uncalibrated backends instead of guessing.
    std::map<std::string, std::uint64_t> service_estimates_by_backend_us;
};

struct ResourceSpec {
    // This id is supplied by configuration or observed at runtime. No hardware
    // resource name is assumed by the core.
    std::string id;
    std::string backend_id;
    BackendFamily family{BackendFamily::Mock};
    std::uint32_t max_concurrency{1};
    std::uint64_t memory_capacity_bytes{0};
    // Explicit shared-contention group. Empty keeps this resource in its own
    // domain; the scheduler never infers topology from the resource ID.
    std::string interference_domain_id;
    // Explicit physical claims from current-session evidence. Multiple
    // selectors may claim the same unit; unknown mappings stay empty.
    std::vector<std::string> physical_resource_ids;
    // Conservative exclusion group. This serializes unknown overlapping
    // accelerators without asserting that they are the same physical unit.
    std::string exclusive_group_id;
};

struct DispatchDecision {
    std::string task_id;
    std::string resource_id;
    bool dispatched{false};
    std::string reason;
    std::vector<std::string> evicted_models;
};

struct ResourceSnapshot {
    ResourceSpec spec;
    std::uint32_t active_tasks{0};
    std::uint64_t resident_memory_bytes{0};
    std::vector<std::string> resident_models;
};

std::string to_string(WorkloadKind value);
std::string to_string(TaskState value);
std::string to_string(BackendFamily value);

}  // namespace hexagonflow
