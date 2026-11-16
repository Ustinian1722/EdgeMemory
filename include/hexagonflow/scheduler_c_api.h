#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(HEXAGONFLOW_SCHEDULER_C_EXPORTS)
#    define HF_SCHEDULER_API __declspec(dllexport)
#  else
#    define HF_SCHEDULER_API __declspec(dllimport)
#  endif
#else
#  define HF_SCHEDULER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hf_scheduler hf_scheduler;

typedef struct hf_resource_spec {
    const char* id;
    const char* backend_id;
    const char* interference_domain_id;
    int family; /* 0=Mock, 1=GenieX, 2=QNN/QAIRT, 3=llama.cpp, 4=CPU, 5=GPU */
    uint32_t max_concurrency;
    uint64_t memory_capacity_bytes;
    const char* const* physical_resource_ids;
    size_t physical_resource_count;
    const char* exclusive_group_id;
} hf_resource_spec;

typedef struct hf_workload_spec {
    const char* id;
    const char* model_id;
    int kind; /* 0=LLM prefill, 1=LLM decode, 2=ASR, 3=embedding, 4=custom */
    int32_t priority;
    int has_deadline;
    uint64_t absolute_deadline_us;
    uint64_t estimated_service_us;
    uint64_t model_memory_bytes;
    uint64_t requested_output_tokens;
    uint64_t input_units;
    const char* const* allowed_backend_ids;
    size_t allowed_backend_count;
    const struct hf_service_estimate* service_estimates_by_backend;
    size_t service_estimate_count;
} hf_workload_spec;

typedef struct hf_service_estimate {
    const char* backend_id;
    uint64_t service_us;
} hf_service_estimate;

typedef struct hf_dispatch_decision {
    char task_id[128];
    char resource_id[128];
    char reason[192];
    int dispatched;
} hf_dispatch_decision;

HF_SCHEDULER_API hf_scheduler* hf_scheduler_create(
    int policy, const hf_resource_spec* resources, size_t resource_count,
    char* error, size_t error_capacity);
HF_SCHEDULER_API void hf_scheduler_destroy(hf_scheduler* scheduler);
HF_SCHEDULER_API int hf_scheduler_submit(
    hf_scheduler* scheduler, const hf_workload_spec* workload,
    char* error, size_t error_capacity);
HF_SCHEDULER_API ptrdiff_t hf_scheduler_dispatch(
    hf_scheduler* scheduler, hf_dispatch_decision* decisions,
    size_t decision_capacity, char* error, size_t error_capacity);
HF_SCHEDULER_API int hf_scheduler_complete(hf_scheduler* scheduler,
                                           const char* task_id);
HF_SCHEDULER_API int hf_scheduler_fail(hf_scheduler* scheduler,
                                       const char* task_id,
                                       const char* reason);
HF_SCHEDULER_API int hf_scheduler_cancel_queued(hf_scheduler* scheduler,
                                                const char* task_id);
HF_SCHEDULER_API int hf_scheduler_load_interference_tsv(
    hf_scheduler* scheduler, const char* profile_path,
    char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
