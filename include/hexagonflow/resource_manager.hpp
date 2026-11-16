#pragma once

#include "hexagonflow/types.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hexagonflow {

struct ReservationResult {
    bool reserved{false};
    std::string reason;
    std::vector<std::string> evicted_models;
};

class ResourceManager {
public:
    explicit ResourceManager(std::vector<ResourceSpec> resources);

    [[nodiscard]] const std::vector<ResourceSpec>& resources() const noexcept;
    [[nodiscard]] ResourceSnapshot snapshot(const std::string& resource_id) const;
    [[nodiscard]] std::vector<ResourceSnapshot> snapshots() const;

    ReservationResult try_reserve(const Workload& workload,
                                  const std::string& resource_id);
    bool release(const std::string& task_id);
    [[nodiscard]] bool is_reserved(const std::string& task_id) const;

private:
    struct ResidentModel {
        std::uint64_t bytes{0};
        std::uint32_t active_tasks{0};
        std::uint64_t last_touch{0};
    };

    struct ResourceState {
        ResourceSpec spec;
        std::unordered_map<std::string, ResidentModel> models;
        std::uint32_t active_tasks{0};
    };

    struct Reservation {
        std::string resource_id;
        std::string model_id;
    };

    ResourceState* find_resource(const std::string& resource_id);
    const ResourceState* find_resource(const std::string& resource_id) const;
    static std::uint64_t resident_bytes(const ResourceState& state);

    std::vector<ResourceSpec> specs_;
    std::unordered_map<std::string, ResourceState> states_;
    std::unordered_map<std::string, Reservation> reservations_;
    std::uint64_t touch_counter_{0};
};

}  // namespace hexagonflow
