#include "hexagonflow/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace hexagonflow {

std::string to_string(SchedulingPolicy policy) {
    switch (policy) {
        case SchedulingPolicy::Fifo: return "fifo";
        case SchedulingPolicy::Priority: return "priority";
        case SchedulingPolicy::EarliestDeadlineFirst: return "edf";
        case SchedulingPolicy::StaticAllocation: return "static";
        case SchedulingPolicy::PredictedLatency: return "predicted_latency";
        case SchedulingPolicy::InterferenceAwareLatency: return "interference_aware_latency";
    }
    return "unknown";
}

Scheduler::Scheduler(
    SchedulingPolicy policy,
    ResourceManager& resources,
    std::map<std::string, std::string> static_model_bindings,
    std::shared_ptr<InterferencePredictor> interference_predictor,
    ObservationSource interference_source)
    : policy_(policy),
      resources_(resources),
      static_model_bindings_(std::move(static_model_bindings)),
      interference_predictor_(std::move(interference_predictor)),
      interference_source_(interference_source) {
    if (policy_ == SchedulingPolicy::InterferenceAwareLatency &&
        !interference_predictor_) {
        throw std::invalid_argument(
            "interference-aware scheduling requires an interference predictor");
    }
}

void Scheduler::observe_interference(
    const InterferenceObservation& observation) {
    if (!interference_predictor_) {
        throw std::logic_error("scheduler has no interference predictor");
    }
    interference_predictor_->observe(observation);
}

bool Scheduler::submit(Workload workload, std::string* error) {
    if (workload.id.empty()) {
        if (error != nullptr) *error = "task-id-required";
        return false;
    }
    if (workload.model_id.empty()) {
        if (error != nullptr) *error = "model-id-required";
        return false;
    }
    if (tasks_.contains(workload.id)) {
        if (error != nullptr) *error = "duplicate-task-id";
        return false;
    }

    workload.arrival_sequence = next_arrival_sequence_++;
    const auto task_id = workload.id;
    tasks_.emplace(task_id, TaskSnapshot{std::move(workload), TaskState::Queued,
                                         {}, {}});
    queued_ids_.push_back(task_id);
    return true;
}

std::vector<std::string> Scheduler::ordered_queue() const {
    auto ids = queued_ids_;
    const auto arrival = [this](const std::string& left,
                                const std::string& right) {
        return tasks_.at(left).workload.arrival_sequence <
               tasks_.at(right).workload.arrival_sequence;
    };

    if (policy_ == SchedulingPolicy::Priority) {
        std::stable_sort(ids.begin(), ids.end(), [this, &arrival](
                                                    const std::string& left,
                                                    const std::string& right) {
            const auto& l = tasks_.at(left).workload;
            const auto& r = tasks_.at(right).workload;
            if (l.priority != r.priority) return l.priority > r.priority;
            return arrival(left, right);
        });
    } else if (policy_ == SchedulingPolicy::EarliestDeadlineFirst) {
        std::stable_sort(ids.begin(), ids.end(), [this, &arrival](
                                                    const std::string& left,
                                                    const std::string& right) {
            const auto& l = tasks_.at(left).workload;
            const auto& r = tasks_.at(right).workload;
            if (l.absolute_deadline_us.has_value() !=
                r.absolute_deadline_us.has_value()) {
                return l.absolute_deadline_us.has_value();
            }
            if (l.absolute_deadline_us.has_value() &&
                l.absolute_deadline_us != r.absolute_deadline_us) {
                return l.absolute_deadline_us < r.absolute_deadline_us;
            }
            return arrival(left, right);
        });
    } else {
        std::stable_sort(ids.begin(), ids.end(), arrival);
    }
    return ids;
}

std::vector<DispatchDecision> Scheduler::dispatch_ready() {
    std::vector<DispatchDecision> decisions;
    const auto ordered = ordered_queue();
    for (const auto& task_id : ordered) {
        auto& task_snapshot = tasks_.at(task_id);
        std::vector<const ResourceSpec*> candidates;
        if (policy_ == SchedulingPolicy::StaticAllocation) {
            const auto binding =
                static_model_bindings_.find(task_snapshot.workload.model_id);
            if (binding == static_model_bindings_.end()) {
                decisions.push_back(
                    {task_id, {}, false, "static-model-binding-missing", {}});
                continue;
            }
            const auto found =
                std::find_if(resources_.resources().begin(),
                             resources_.resources().end(),
                             [&binding](const ResourceSpec& spec) {
                                 return spec.id == binding->second;
                             });
            if (found == resources_.resources().end()) {
                decisions.push_back(
                    {task_id, binding->second, false,
                     "static-resource-not-configured", {}});
                continue;
            }
            candidates.push_back(&*found);
        } else {
            for (const auto& resource : resources_.resources()) {
                candidates.push_back(&resource);
            }
        }
        // Filter by the request's backend allowlist before checking capacity.
        // Otherwise an occupied allowed resource can yield
        // concurrency-capacity-exhausted, then an unrelated resource can
        // overwrite that queued condition with backend-not-allowed.
        const auto& allowed_backends = task_snapshot.workload.allowed_backend_ids;
        if (!allowed_backends.empty()) {
            candidates.erase(
                std::remove_if(candidates.begin(), candidates.end(),
                               [&allowed_backends](const ResourceSpec* resource) {
                                   return std::find(allowed_backends.begin(),
                                                    allowed_backends.end(),
                                                    resource->backend_id) ==
                                          allowed_backends.end();
                               }),
                candidates.end());
            if (candidates.empty()) {
                decisions.push_back({task_id, {}, false, "backend-not-allowed", {}});
                continue;
            }
        }
        std::string last_reason = "no-compatible-resource";
        std::vector<std::string> fallback_estimated_resources;

        if (policy_ == SchedulingPolicy::PredictedLatency ||
            policy_ == SchedulingPolicy::InterferenceAwareLatency) {
            std::map<std::string, std::uint64_t> predicted_services;
            bool missing_service_estimate = false;
            bool missing_interference_estimate = false;
            for (const auto* resource : candidates) {
                const auto found = task_snapshot.workload.service_estimates_by_backend_us.find(
                    resource->backend_id);
                if (found == task_snapshot.workload.service_estimates_by_backend_us.end() ||
                    found->second == 0U) {
                    missing_service_estimate = true;
                    continue;
                }
                auto service_us = found->second;
                if (policy_ == SchedulingPolicy::InterferenceAwareLatency) {
                    const auto penalty = interference_predictor_->estimate_penalty_us(
                        interference_query(task_snapshot.workload, *resource));
                    if (!penalty.has_value()) {
                        missing_interference_estimate = true;
                        continue;
                    }
                    const auto maximum = std::numeric_limits<std::uint64_t>::max();
                    service_us = *penalty > maximum - service_us
                                     ? maximum : service_us + *penalty;
                }
                predicted_services.emplace(resource->id, service_us);
            }
            if (policy_ == SchedulingPolicy::PredictedLatency &&
                !predicted_services.empty() && missing_service_estimate) {
                // Keep an unmeasured fallback schedulable after one backend has
                // produced observations. The conservative estimate is the
                // slowest observed candidate service time, never a claimed
                // measurement for the unobserved backend. If a calibrated
                // accelerator is busy, this lets an available CPU fallback run
                // and return its own first real sample to the next request.
                const auto fallback_service = std::max_element(
                    predicted_services.begin(), predicted_services.end(),
                    [](const auto& left, const auto& right) {
                        return left.second < right.second;
                    })->second;
                for (const auto* resource : candidates) {
                    if (!predicted_services.contains(resource->id)) {
                        predicted_services.emplace(resource->id, fallback_service);
                        fallback_estimated_resources.push_back(resource->id);
                    }
                }
            }
            const bool prediction_cold_start =
                predicted_services.empty() &&
                policy_ == SchedulingPolicy::PredictedLatency;
            if (prediction_cold_start) {
                // A fresh QDC session has no device-local timing history. Keep
                // the caller's deterministic resource order for the first real
                // request so it can run and provide the first observation.
                last_reason = "prediction-cold-start-priority-fallback";
            } else {
                candidates.erase(
                    std::remove_if(candidates.begin(), candidates.end(),
                                   [&predicted_services](const ResourceSpec* resource) {
                                       return !predicted_services.contains(resource->id);
                                   }),
                    candidates.end());
                const auto estimate_for = [&predicted_services](const ResourceSpec& resource) {
                    return predicted_services.at(resource.id);
                };
                const auto now_us = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                const auto projected_finish = [&task_snapshot, &resources = resources_, now_us,
                                               &estimate_for](const ResourceSpec& resource) {
                    const auto service = estimate_for(resource);
                    const auto snapshot = resources.snapshot(resource.id);
                    const std::uint64_t capacity = std::max<std::uint32_t>(
                        1U, resource.max_concurrency);
                    const std::uint64_t queued =
                        static_cast<std::uint64_t>(snapshot.active_tasks) + 1U;
                    const std::uint64_t waves = queued / capacity +
                                                (queued % capacity == 0U ? 0U : 1U);
                    const auto maximum = std::numeric_limits<std::uint64_t>::max();
                    const auto service_with_queue = service > maximum / waves
                                                        ? maximum : service * waves;
                    return service_with_queue > maximum - now_us
                               ? maximum : now_us + service_with_queue;
                };
                const auto score = [&task_snapshot, &projected_finish](const ResourceSpec& resource) {
                    const auto finish = projected_finish(resource);
                    const auto& deadline = task_snapshot.workload.absolute_deadline_us;
                    if (!deadline || finish <= *deadline) {
                        return std::tuple<unsigned int, std::uint64_t, std::uint64_t>{0U, 0U, finish};
                    }
                    return std::tuple<unsigned int, std::uint64_t, std::uint64_t>{
                        1U, finish - *deadline, finish};
                };
                std::stable_sort(candidates.begin(), candidates.end(),
                                 [&score](const ResourceSpec* left, const ResourceSpec* right) {
                                     return score(*left) < score(*right);
                                 });
            }
            if (candidates.empty()) {
                if (missing_service_estimate && missing_interference_estimate) {
                    last_reason = "no-calibrated-service-or-interference-estimate";
                } else if (missing_interference_estimate) {
                    last_reason = "no-calibrated-interference-estimate";
                } else if (missing_service_estimate) {
                    last_reason = "no-calibrated-service-estimate";
                }
            }
        }

        bool dispatched = false;
        for (const auto* resource : candidates) {
            const auto result =
                resources_.try_reserve(task_snapshot.workload, resource->id);
            if (!result.reserved) {
                last_reason = result.reason;
                continue;
            }
            task_snapshot.state = TaskState::Dispatched;
            task_snapshot.resource_id = resource->id;
            const bool fallback_estimated = std::find(
                fallback_estimated_resources.begin(), fallback_estimated_resources.end(),
                resource->id) != fallback_estimated_resources.end();
            const auto reason = fallback_estimated
                                    ? "dispatched-unmeasured-fallback-estimate"
                                    : (last_reason == "no-compatible-resource"
                                           ? "dispatched" : last_reason);
            decisions.push_back({task_id, resource->id, true, reason,
                                 result.evicted_models});
            dispatched = true;
            break;
        }
        if (!dispatched) {
            decisions.push_back({task_id, {}, false, last_reason, {}});
        }
    }

    queued_ids_.erase(
        std::remove_if(queued_ids_.begin(), queued_ids_.end(),
                       [this](const std::string& task_id) {
                           return tasks_.at(task_id).state != TaskState::Queued;
                       }),
        queued_ids_.end());
    return decisions;
}

InterferenceQuery Scheduler::interference_query(
    const Workload& target, const ResourceSpec& resource) const {
    const auto domain_for = [](const ResourceSpec& spec) -> const std::string& {
        return spec.interference_domain_id.empty() ? spec.id
                                                   : spec.interference_domain_id;
    };
    const auto target_snapshot = resources_.snapshot(resource.id);
    InterferenceQuery query;
    query.resource_domain_id = domain_for(resource);
    query.target = {target.model_id, resource.backend_id, target.kind,
                    target.input_units, target.requested_output_tokens,
                    std::find(target_snapshot.resident_models.begin(),
                              target_snapshot.resident_models.end(), target.model_id) !=
                        target_snapshot.resident_models.end()};
    query.source = interference_source_;

    for (const auto& [task_id, snapshot] : tasks_) {
        (void)task_id;
        if (snapshot.state != TaskState::Dispatched) continue;
        const auto active_resource = std::find_if(
            resources_.resources().begin(), resources_.resources().end(),
            [&snapshot](const ResourceSpec& candidate) {
                return candidate.id == snapshot.resource_id;
            });
        if (active_resource == resources_.resources().end() ||
            domain_for(*active_resource) != query.resource_domain_id) {
            continue;
        }
        const auto active_snapshot = resources_.snapshot(active_resource->id);
        const bool resident = std::find(active_snapshot.resident_models.begin(),
                                        active_snapshot.resident_models.end(),
                                        snapshot.workload.model_id) !=
                              active_snapshot.resident_models.end();
        query.co_running.push_back({snapshot.workload.model_id,
                                    active_resource->backend_id,
                                    snapshot.workload.kind,
                                    snapshot.workload.input_units,
                                    snapshot.workload.requested_output_tokens,
                                    resident});
    }
    return query;
}

bool Scheduler::finish(const std::string& task_id,
                       TaskState state,
                       std::string error) {
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end() || found->second.state != TaskState::Dispatched) {
        return false;
    }
    if (!resources_.release(task_id)) {
        throw std::logic_error("dispatched task has no resource reservation");
    }
    found->second.state = state;
    found->second.error = std::move(error);
    return true;
}

bool Scheduler::complete(const std::string& task_id) {
    return finish(task_id, TaskState::Completed, {});
}

bool Scheduler::fail(const std::string& task_id, std::string reason) {
    if (reason.empty()) reason = "unspecified-backend-failure";
    return finish(task_id, TaskState::Failed, std::move(reason));
}

bool Scheduler::cancel_queued(const std::string& task_id) {
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end() || found->second.state != TaskState::Queued) {
        return false;
    }
    found->second.state = TaskState::Cancelled;
    queued_ids_.erase(std::remove(queued_ids_.begin(), queued_ids_.end(), task_id),
                      queued_ids_.end());
    return true;
}

TaskSnapshot Scheduler::task(const std::string& task_id) const {
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        throw std::invalid_argument("unknown task: " + task_id);
    }
    return found->second;
}

std::vector<TaskSnapshot> Scheduler::tasks() const {
    std::vector<TaskSnapshot> result;
    result.reserve(tasks_.size());
    for (const auto& [task_id, snapshot] : tasks_) {
        (void)task_id;
        result.push_back(snapshot);
    }
    std::sort(result.begin(), result.end(),
              [](const TaskSnapshot& left, const TaskSnapshot& right) {
                  return left.workload.arrival_sequence <
                         right.workload.arrival_sequence;
              });
    return result;
}

std::vector<std::string> Scheduler::queued_order() const {
    return ordered_queue();
}

SchedulingPolicy Scheduler::policy() const noexcept {
    return policy_;
}

}  // namespace hexagonflow
