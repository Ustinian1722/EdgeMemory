#include "hexagonflow/resource_manager.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hexagonflow {

ResourceManager::ResourceManager(std::vector<ResourceSpec> resources)
    : specs_(std::move(resources)) {
    for (const auto& spec : specs_) {
        if (spec.id.empty() || spec.backend_id.empty()) {
            throw std::invalid_argument("resource id and backend id are required");
        }
        if (spec.max_concurrency == 0U) {
            throw std::invalid_argument("resource max_concurrency must be positive");
        }
        if (states_.contains(spec.id)) {
            throw std::invalid_argument("duplicate resource id: " + spec.id);
        }
        states_.emplace(spec.id, ResourceState{spec, {}, 0U});
    }
}

const std::vector<ResourceSpec>& ResourceManager::resources() const noexcept {
    return specs_;
}

ResourceManager::ResourceState* ResourceManager::find_resource(
    const std::string& resource_id) {
    const auto found = states_.find(resource_id);
    return found == states_.end() ? nullptr : &found->second;
}

const ResourceManager::ResourceState* ResourceManager::find_resource(
    const std::string& resource_id) const {
    const auto found = states_.find(resource_id);
    return found == states_.end() ? nullptr : &found->second;
}

std::uint64_t ResourceManager::resident_bytes(const ResourceState& state) {
    std::uint64_t total = 0;
    for (const auto& [model_id, model] : state.models) {
        (void)model_id;
        total += model.bytes;
    }
    return total;
}

ResourceSnapshot ResourceManager::snapshot(const std::string& resource_id) const {
    const auto* state = find_resource(resource_id);
    if (state == nullptr) {
        throw std::invalid_argument("unknown resource: " + resource_id);
    }

    ResourceSnapshot result{state->spec, state->active_tasks,
                            resident_bytes(*state), {}};
    result.resident_models.reserve(state->models.size());
    for (const auto& [model_id, model] : state->models) {
        (void)model;
        result.resident_models.push_back(model_id);
    }
    std::sort(result.resident_models.begin(), result.resident_models.end());
    return result;
}

std::vector<ResourceSnapshot> ResourceManager::snapshots() const {
    std::vector<ResourceSnapshot> result;
    result.reserve(specs_.size());
    for (const auto& spec : specs_) {
        result.push_back(snapshot(spec.id));
    }
    return result;
}

ReservationResult ResourceManager::try_reserve(const Workload& workload,
                                                const std::string& resource_id) {
    if (workload.id.empty() || workload.model_id.empty()) {
        return {false, "task-and-model-ids-required", {}};
    }
    if (reservations_.contains(workload.id)) {
        return {false, "task-already-reserved", {}};
    }

    auto* state = find_resource(resource_id);
    if (state == nullptr) {
        return {false, "unknown-resource", {}};
    }
    if (!workload.allowed_backend_ids.empty() &&
        std::find(workload.allowed_backend_ids.begin(),
                  workload.allowed_backend_ids.end(), state->spec.backend_id) ==
            workload.allowed_backend_ids.end()) {
        return {false, "backend-not-allowed", {}};
    }
    if (state->active_tasks >= state->spec.max_concurrency) {
        return {false, "concurrency-capacity-exhausted", {}};
    }
    if (!state->spec.exclusive_group_id.empty()) {
        for (const auto& [active_id, active_state] : states_) {
            (void)active_id;
            if (active_state.active_tasks > 0U &&
                active_state.spec.exclusive_group_id == state->spec.exclusive_group_id) {
                return {false, "exclusive-contention-group-busy", {}};
            }
        }
    }
    if (!state->spec.physical_resource_ids.empty()) {
        for (const auto& [active_id, active_state] : states_) {
            (void)active_id;
            if (active_state.active_tasks == 0U ||
                active_state.spec.physical_resource_ids.empty()) continue;
            const bool overlaps = std::any_of(
                state->spec.physical_resource_ids.begin(),
                state->spec.physical_resource_ids.end(),
                [&active_state](const std::string& physical_id) {
                    return std::find(active_state.spec.physical_resource_ids.begin(),
                                     active_state.spec.physical_resource_ids.end(),
                                     physical_id) != active_state.spec.physical_resource_ids.end();
                });
            if (overlaps) return {false, "verified-physical-resource-overlap", {}};
        }
    }
    if (workload.model_memory_bytes > state->spec.memory_capacity_bytes) {
        return {false, "model-exceeds-resource-memory", {}};
    }

    const auto existing = state->models.find(workload.model_id);
    if (existing != state->models.end() &&
        existing->second.bytes != workload.model_memory_bytes) {
        return {false, "model-memory-estimate-changed", {}};
    }

    std::vector<std::string> evictions;
    if (existing == state->models.end()) {
        const auto used = resident_bytes(*state);
        const auto free = state->spec.memory_capacity_bytes - used;
        std::uint64_t reclaimed = 0;

        std::vector<std::pair<std::string, ResidentModel>> candidates;
        candidates.reserve(state->models.size());
        for (const auto& [model_id, model] : state->models) {
            if (model.active_tasks == 0U) {
                candidates.emplace_back(model_id, model);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& left, const auto& right) {
                      return left.second.last_touch < right.second.last_touch;
                  });

        for (const auto& [model_id, model] : candidates) {
            if (free + reclaimed >= workload.model_memory_bytes) {
                break;
            }
            reclaimed += model.bytes;
            evictions.push_back(model_id);
        }
        if (free + reclaimed < workload.model_memory_bytes) {
            return {false, "resident-model-memory-pinned", {}};
        }
        for (const auto& model_id : evictions) {
            state->models.erase(model_id);
        }
        state->models.emplace(
            workload.model_id,
            ResidentModel{workload.model_memory_bytes, 0U, ++touch_counter_});
    }

    auto& resident = state->models.at(workload.model_id);
    ++resident.active_tasks;
    resident.last_touch = ++touch_counter_;
    ++state->active_tasks;
    reservations_.emplace(workload.id,
                          Reservation{resource_id, workload.model_id});
    return {true, "reserved", std::move(evictions)};
}

bool ResourceManager::release(const std::string& task_id) {
    const auto found = reservations_.find(task_id);
    if (found == reservations_.end()) {
        return false;
    }

    auto* state = find_resource(found->second.resource_id);
    if (state == nullptr) {
        return false;
    }
    const auto resident = state->models.find(found->second.model_id);
    if (resident == state->models.end() || resident->second.active_tasks == 0U ||
        state->active_tasks == 0U) {
        throw std::logic_error("resource reservation accounting is inconsistent");
    }

    --resident->second.active_tasks;
    resident->second.last_touch = ++touch_counter_;
    --state->active_tasks;
    reservations_.erase(found);
    return true;
}

bool ResourceManager::is_reserved(const std::string& task_id) const {
    return reservations_.contains(task_id);
}

}  // namespace hexagonflow
