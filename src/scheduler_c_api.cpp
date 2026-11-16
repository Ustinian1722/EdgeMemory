#define HEXAGONFLOW_SCHEDULER_C_EXPORTS
#include "hexagonflow/scheduler_c_api.h"

#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"
#include "hexagonflow/interference_io.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct hf_scheduler {
    std::unique_ptr<hexagonflow::ResourceManager> resources;
    std::unique_ptr<hexagonflow::Scheduler> scheduler;
};

namespace {

void write_text(char* destination, size_t capacity, const std::string& text) noexcept {
    if (destination == nullptr || capacity == 0U) return;
    const auto count = std::min(capacity - 1U, text.size());
    std::memcpy(destination, text.data(), count);
    destination[count] = '\0';
}

hexagonflow::BackendFamily backend_family(int family) {
    using hexagonflow::BackendFamily;
    switch (family) {
        case 0: return BackendFamily::Mock;
        case 1: return BackendFamily::GenieX;
        case 2: return BackendFamily::QnnQairt;
        case 3: return BackendFamily::LlamaCppHexagon;
        case 4: return BackendFamily::Cpu;
        case 5: return BackendFamily::Gpu;
        default: throw std::invalid_argument("unknown backend family");
    }
}

hexagonflow::WorkloadKind workload_kind(int kind) {
    using hexagonflow::WorkloadKind;
    switch (kind) {
        case 0: return WorkloadKind::LlmPrefill;
        case 1: return WorkloadKind::LlmDecode;
        case 2: return WorkloadKind::Asr;
        case 3: return WorkloadKind::Embedding;
        case 4: return WorkloadKind::Custom;
        default: throw std::invalid_argument("unknown workload kind");
    }
}

hexagonflow::SchedulingPolicy scheduling_policy(int policy) {
    using hexagonflow::SchedulingPolicy;
    switch (policy) {
        case 0: return SchedulingPolicy::Fifo;
        case 1: return SchedulingPolicy::Priority;
        case 2: return SchedulingPolicy::EarliestDeadlineFirst;
        case 3: return SchedulingPolicy::PredictedLatency;
        case 4: return SchedulingPolicy::InterferenceAwareLatency;
        default: throw std::invalid_argument("unknown scheduling policy");
    }
}

}  // namespace

extern "C" HF_SCHEDULER_API hf_scheduler* hf_scheduler_create(
    int policy, const hf_resource_spec* resources, size_t resource_count,
    char* error, size_t error_capacity) {
    try {
        if (resources == nullptr || resource_count == 0U || resource_count > 64U) {
            throw std::invalid_argument("resource count must be in 1..64");
        }
        std::vector<hexagonflow::ResourceSpec> specs;
        specs.reserve(resource_count);
        for (size_t index = 0; index < resource_count; ++index) {
            const auto& item = resources[index];
            if (item.id == nullptr || item.backend_id == nullptr ||
                item.interference_domain_id == nullptr ||
                item.exclusive_group_id == nullptr ||
                (item.physical_resource_count > 0U && item.physical_resource_ids == nullptr)) {
                throw std::invalid_argument("resource string fields are required");
            }
            hexagonflow::ResourceSpec spec{
                item.id, item.backend_id, backend_family(item.family),
                item.max_concurrency, item.memory_capacity_bytes,
                item.interference_domain_id};
            spec.exclusive_group_id = item.exclusive_group_id;
            for (size_t resource_index = 0;
                 resource_index < item.physical_resource_count; ++resource_index) {
                if (item.physical_resource_ids[resource_index] == nullptr) {
                    throw std::invalid_argument("physical resource IDs cannot be null");
                }
                spec.physical_resource_ids.emplace_back(
                    item.physical_resource_ids[resource_index]);
            }
            specs.push_back(std::move(spec));
        }
        auto result = std::make_unique<hf_scheduler>();
        result->resources = std::make_unique<hexagonflow::ResourceManager>(std::move(specs));
        const auto selected_policy = scheduling_policy(policy);
        std::shared_ptr<hexagonflow::InterferencePredictor> predictor;
        if (selected_policy == hexagonflow::SchedulingPolicy::InterferenceAwareLatency) {
            predictor = std::make_shared<hexagonflow::PairwiseEmaPredictor>();
        }
        result->scheduler = std::make_unique<hexagonflow::Scheduler>(
            selected_policy, *result->resources, std::map<std::string, std::string>{},
            std::move(predictor), hexagonflow::ObservationSource::Hardware);
        if (error != nullptr && error_capacity > 0U) error[0] = '\0';
        return result.release();
    } catch (const std::exception& exception) {
        write_text(error, error_capacity, exception.what());
        return nullptr;
    } catch (...) {
        write_text(error, error_capacity, "unknown scheduler initialization error");
        return nullptr;
    }
}

extern "C" HF_SCHEDULER_API void hf_scheduler_destroy(hf_scheduler* scheduler) {
    delete scheduler;
}

extern "C" HF_SCHEDULER_API int hf_scheduler_submit(
    hf_scheduler* scheduler, const hf_workload_spec* workload,
    char* error, size_t error_capacity) {
    try {
        if (scheduler == nullptr || workload == nullptr || workload->id == nullptr ||
            workload->model_id == nullptr ||
            (workload->allowed_backend_count > 0U && workload->allowed_backend_ids == nullptr)) {
            throw std::invalid_argument("invalid scheduler or workload arguments");
        }
        hexagonflow::Workload item;
        item.id = workload->id;
        item.model_id = workload->model_id;
        item.kind = workload_kind(workload->kind);
        item.priority = workload->priority;
        if (workload->has_deadline != 0) item.absolute_deadline_us = workload->absolute_deadline_us;
        item.estimated_service_us = workload->estimated_service_us;
        item.model_memory_bytes = workload->model_memory_bytes;
        item.requested_output_tokens = workload->requested_output_tokens;
        item.input_units = workload->input_units;
        for (size_t index = 0; index < workload->allowed_backend_count; ++index) {
            if (workload->allowed_backend_ids[index] == nullptr) {
                throw std::invalid_argument("allowed backend id is null");
            }
            item.allowed_backend_ids.emplace_back(workload->allowed_backend_ids[index]);
        }
        if (workload->service_estimate_count > 0U &&
            workload->service_estimates_by_backend == nullptr) {
            throw std::invalid_argument("service estimate array is null");
        }
        for (size_t index = 0; index < workload->service_estimate_count; ++index) {
            const auto& estimate = workload->service_estimates_by_backend[index];
            if (estimate.backend_id == nullptr || estimate.service_us == 0U) {
                throw std::invalid_argument("service estimates require backend IDs and positive times");
            }
            if (!item.service_estimates_by_backend_us.emplace(
                    estimate.backend_id, estimate.service_us).second) {
                throw std::invalid_argument("duplicate service estimate backend ID");
            }
        }
        std::string reason;
        if (!scheduler->scheduler->submit(std::move(item), &reason)) {
            write_text(error, error_capacity, reason);
            return 0;
        }
        if (error != nullptr && error_capacity > 0U) error[0] = '\0';
        return 1;
    } catch (const std::exception& exception) {
        write_text(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_text(error, error_capacity, "unknown scheduler submit error");
        return 0;
    }
}

extern "C" HF_SCHEDULER_API ptrdiff_t hf_scheduler_dispatch(
    hf_scheduler* scheduler, hf_dispatch_decision* decisions,
    size_t decision_capacity, char* error, size_t error_capacity) {
    try {
        if (scheduler == nullptr) throw std::invalid_argument("scheduler is null");
        const auto queued_count = scheduler->scheduler->queued_order().size();
        if (queued_count > decision_capacity ||
            (queued_count > 0U && decisions == nullptr)) {
            throw std::invalid_argument("decision buffer is smaller than the queued task count");
        }
        const auto result = scheduler->scheduler->dispatch_ready();
        for (size_t index = 0; index < result.size(); ++index) {
            write_text(decisions[index].task_id, sizeof(decisions[index].task_id), result[index].task_id);
            write_text(decisions[index].resource_id, sizeof(decisions[index].resource_id), result[index].resource_id);
            write_text(decisions[index].reason, sizeof(decisions[index].reason), result[index].reason);
            decisions[index].dispatched = result[index].dispatched ? 1 : 0;
        }
        if (error != nullptr && error_capacity > 0U) error[0] = '\0';
        return static_cast<ptrdiff_t>(result.size());
    } catch (const std::exception& exception) {
        write_text(error, error_capacity, exception.what());
        return -1;
    } catch (...) {
        write_text(error, error_capacity, "unknown scheduler dispatch error");
        return -1;
    }
}

extern "C" HF_SCHEDULER_API int hf_scheduler_complete(hf_scheduler* scheduler,
                                                       const char* task_id) {
    if (scheduler == nullptr || task_id == nullptr) return 0;
    try { return scheduler->scheduler->complete(task_id) ? 1 : 0; }
    catch (...) { return 0; }
}

extern "C" HF_SCHEDULER_API int hf_scheduler_fail(hf_scheduler* scheduler,
                                                   const char* task_id,
                                                   const char* reason) {
    if (scheduler == nullptr || task_id == nullptr) return 0;
    try { return scheduler->scheduler->fail(task_id, reason == nullptr ? "" : reason) ? 1 : 0; }
    catch (...) { return 0; }
}

extern "C" HF_SCHEDULER_API int hf_scheduler_cancel_queued(hf_scheduler* scheduler,
                                                            const char* task_id) {
    if (scheduler == nullptr || task_id == nullptr) return 0;
    try { return scheduler->scheduler->cancel_queued(task_id) ? 1 : 0; }
    catch (...) { return 0; }
}

extern "C" HF_SCHEDULER_API int hf_scheduler_load_interference_tsv(
    hf_scheduler* scheduler, const char* profile_path,
    char* error, size_t error_capacity) {
    try {
        if (scheduler == nullptr || profile_path == nullptr || *profile_path == '\0') {
            throw std::invalid_argument("scheduler and interference profile path are required");
        }
        std::ifstream input(profile_path);
        if (!input) throw std::runtime_error("cannot open interference profile");
        const auto count = hexagonflow::load_interference_observations_tsv(
            input, *scheduler->scheduler);
        if (count == 0U) throw std::runtime_error("interference profile contains no observations");
        if (error != nullptr && error_capacity > 0U) error[0] = '\0';
        return static_cast<int>(count);
    } catch (const std::exception& exception) {
        write_text(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_text(error, error_capacity, "unknown interference profile load error");
        return 0;
    }
}
