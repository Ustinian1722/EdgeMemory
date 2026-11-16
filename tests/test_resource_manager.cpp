#include "hexagonflow/resource_manager.hpp"
#include "test_framework.hpp"

#include <vector>

namespace {

hexagonflow::ResourceManager make_manager(std::uint64_t capacity,
                                          std::uint32_t concurrency = 2U) {
    return hexagonflow::ResourceManager({
        {"sim-resource", "mock", hexagonflow::BackendFamily::Mock,
         concurrency, capacity},
    });
}

hexagonflow::Workload workload(const std::string& id,
                               const std::string& model,
                               std::uint64_t bytes) {
    hexagonflow::Workload result;
    result.id = id;
    result.model_id = model;
    result.model_memory_bytes = bytes;
    result.allowed_backend_ids = {"mock"};
    return result;
}

}  // namespace

HF_TEST(resource_manager_reserves_and_releases_concurrency) {
    auto manager = make_manager(100U, 1U);
    HF_REQUIRE(manager.try_reserve(workload("t1", "m1", 60U),
                                   "sim-resource").reserved);
    HF_REQUIRE(!manager.try_reserve(workload("t2", "m2", 10U),
                                    "sim-resource").reserved);
    HF_REQUIRE(manager.release("t1"));
    HF_REQUIRE(!manager.release("t1"));
    HF_REQUIRE(manager.try_reserve(workload("t2", "m2", 10U),
                                   "sim-resource").reserved);
}

HF_TEST(resource_manager_lru_evicts_only_inactive_models) {
    auto manager = make_manager(100U);
    HF_REQUIRE(manager.try_reserve(workload("t1", "model-a", 60U),
                                   "sim-resource").reserved);
    HF_REQUIRE(manager.release("t1"));
    const auto result =
        manager.try_reserve(workload("t2", "model-b", 60U), "sim-resource");
    HF_REQUIRE(result.reserved);
    HF_REQUIRE(result.evicted_models.size() == 1U);
    HF_REQUIRE(result.evicted_models.front() == "model-a");
    const auto state = manager.snapshot("sim-resource");
    HF_REQUIRE(state.resident_memory_bytes == 60U);
    HF_REQUIRE(state.resident_models.size() == 1U);
    HF_REQUIRE(state.resident_models.front() == "model-b");
}

HF_TEST(resource_manager_does_not_evict_in_use_models) {
    auto manager = make_manager(100U);
    HF_REQUIRE(manager.try_reserve(workload("t1", "model-a", 60U),
                                   "sim-resource").reserved);
    const auto blocked =
        manager.try_reserve(workload("t2", "model-b", 60U), "sim-resource");
    HF_REQUIRE(!blocked.reserved);
    HF_REQUIRE(blocked.reason == "resident-model-memory-pinned");
    HF_REQUIRE(manager.snapshot("sim-resource").resident_models.size() == 1U);
}

HF_TEST(resource_manager_rejects_model_larger_than_capacity) {
    auto manager = make_manager(100U);
    const auto result =
        manager.try_reserve(workload("t1", "model-large", 101U),
                            "sim-resource");
    HF_REQUIRE(!result.reserved);
    HF_REQUIRE(result.reason == "model-exceeds-resource-memory");
}
