#pragma once

#include "hexagonflow/interference.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/types.hpp"

#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace hexagonflow {

enum class SchedulingPolicy {
    Fifo,
    Priority,
    EarliestDeadlineFirst,
    StaticAllocation,
    PredictedLatency,
    InterferenceAwareLatency,
};

std::string to_string(SchedulingPolicy policy);

struct TaskSnapshot {
    Workload workload;
    TaskState state{TaskState::Queued};
    std::string resource_id;
    std::string error;
};

class Scheduler {
public:
    Scheduler(SchedulingPolicy policy,
              ResourceManager& resources,
              std::map<std::string, std::string> static_model_bindings = {},
              std::shared_ptr<InterferencePredictor> interference_predictor = {},
              ObservationSource interference_source = ObservationSource::Hardware);

    bool submit(Workload workload, std::string* error = nullptr);
    void observe_interference(const InterferenceObservation& observation);
    std::vector<DispatchDecision> dispatch_ready();
    bool complete(const std::string& task_id);
    bool fail(const std::string& task_id, std::string reason);
    bool cancel_queued(const std::string& task_id);

    [[nodiscard]] TaskSnapshot task(const std::string& task_id) const;
    [[nodiscard]] std::vector<TaskSnapshot> tasks() const;
    [[nodiscard]] std::vector<std::string> queued_order() const;
    [[nodiscard]] SchedulingPolicy policy() const noexcept;

private:
    std::vector<std::string> ordered_queue() const;
    [[nodiscard]] InterferenceQuery interference_query(
        const Workload& target, const ResourceSpec& resource) const;
    bool finish(const std::string& task_id, TaskState state, std::string error);

    SchedulingPolicy policy_;
    ResourceManager& resources_;
    std::map<std::string, std::string> static_model_bindings_;
    std::shared_ptr<InterferencePredictor> interference_predictor_;
    ObservationSource interference_source_{ObservationSource::Hardware};
    std::unordered_map<std::string, TaskSnapshot> tasks_;
    std::vector<std::string> queued_ids_;
    std::uint64_t next_arrival_sequence_{0};
};

}  // namespace hexagonflow
