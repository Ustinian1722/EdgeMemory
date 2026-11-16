#include "hexagonflow/scheduler.hpp"
#include "test_framework.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace {

hexagonflow::Workload make_task(std::string id,
                                std::string model,
                                std::int32_t priority,
                                std::uint64_t deadline) {
    hexagonflow::Workload task;
    task.id = std::move(id);
    task.model_id = std::move(model);
    task.priority = priority;
    task.absolute_deadline_us = deadline;
    task.model_memory_bytes = 1U;
    task.allowed_backend_ids = {"mock"};
    return task;
}

std::vector<std::string> drain_order(hexagonflow::SchedulingPolicy policy,
                                     std::map<std::string, std::string> bindings = {}) {
    hexagonflow::ResourceManager resources({
        {"sim-resource-a", "mock", hexagonflow::BackendFamily::Mock, 1U, 100U},
        {"sim-resource-b", "mock", hexagonflow::BackendFamily::Mock, 1U, 100U},
    });
    hexagonflow::Scheduler scheduler(policy, resources, std::move(bindings));
    HF_REQUIRE(scheduler.submit(make_task("first", "model-a", 1, 300)));
    HF_REQUIRE(scheduler.submit(make_task("second", "model-b", 9, 200)));
    HF_REQUIRE(scheduler.submit(make_task("third", "model-a", 5, 100)));

    std::vector<std::string> order;
    while (!scheduler.queued_order().empty()) {
        const auto decisions = scheduler.dispatch_ready();
        std::vector<std::string> batch;
        for (const auto& decision : decisions) {
            if (decision.dispatched) {
                order.push_back(decision.task_id);
                batch.push_back(decision.task_id);
            }
        }
        HF_REQUIRE(!batch.empty());
        for (const auto& id : batch) HF_REQUIRE(scheduler.complete(id));
    }
    return order;
}

}  // namespace

HF_TEST(fifo_orders_by_submission) {
    const auto order = drain_order(hexagonflow::SchedulingPolicy::Fifo);
    HF_REQUIRE(order.size() == 3U);
    HF_REQUIRE(order[0] == "first");
    HF_REQUIRE(order[1] == "second");
    HF_REQUIRE(order[2] == "third");
}

HF_TEST(priority_orders_larger_priority_first) {
    const auto order = drain_order(hexagonflow::SchedulingPolicy::Priority);
    HF_REQUIRE(order[0] == "second");
    HF_REQUIRE(order[1] == "third");
    HF_REQUIRE(order[2] == "first");
}

HF_TEST(edf_orders_deadlines_and_places_no_deadline_last) {
    const auto order =
        drain_order(hexagonflow::SchedulingPolicy::EarliestDeadlineFirst);
    HF_REQUIRE(order[0] == "third");
    HF_REQUIRE(order[1] == "second");
    HF_REQUIRE(order[2] == "first");
}

HF_TEST(static_allocation_uses_configured_model_bindings) {
    hexagonflow::ResourceManager resources({
        {"sim-resource-a", "mock", hexagonflow::BackendFamily::Mock, 2U, 100U},
        {"sim-resource-b", "mock", hexagonflow::BackendFamily::Mock, 2U, 100U},
    });
    hexagonflow::Scheduler scheduler(
        hexagonflow::SchedulingPolicy::StaticAllocation, resources,
        {{"model-a", "sim-resource-b"}, {"model-b", "sim-resource-a"}});
    HF_REQUIRE(scheduler.submit(make_task("a", "model-a", 0, 1)));
    HF_REQUIRE(scheduler.submit(make_task("b", "model-b", 0, 1)));
    const auto decisions = scheduler.dispatch_ready();
    HF_REQUIRE(decisions.size() == 2U);
    HF_REQUIRE(decisions[0].resource_id == "sim-resource-b");
    HF_REQUIRE(decisions[1].resource_id == "sim-resource-a");
}

HF_TEST(predicted_latency_uses_calibrated_service_and_current_resource_load) {
    hexagonflow::ResourceManager resources({
        {"nsp-0", "geniex-htp1", hexagonflow::BackendFamily::GenieX, 1U, 1000U},
        {"nsp-1", "geniex-htp4", hexagonflow::BackendFamily::GenieX, 1U, 1000U},
    });
    hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::PredictedLatency,
                                     resources);
    const auto now = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    auto first = make_task("first", "smollm2-q4_0", 0, now + 120000U);
    first.allowed_backend_ids = {"geniex-htp1", "geniex-htp4"};
    first.service_estimates_by_backend_us = {{"geniex-htp1", 80000U},
                                              {"geniex-htp4", 100000U}};
    auto second = first;
    second.id = "second";
    HF_REQUIRE(scheduler.submit(std::move(first)));
    HF_REQUIRE(scheduler.submit(std::move(second)));
    const auto decisions = scheduler.dispatch_ready();
    HF_REQUIRE(decisions.size() == 2U);
    HF_REQUIRE(decisions[0].resource_id == "nsp-0");
    HF_REQUIRE(decisions[1].resource_id == "nsp-1");
    HF_REQUIRE(decisions[0].dispatched);
    HF_REQUIRE(decisions[1].dispatched);
    HF_REQUIRE(scheduler.complete("first"));
    HF_REQUIRE(scheduler.complete("second"));
}

HF_TEST(predicted_latency_uses_deterministic_cold_start_fallback) {
    hexagonflow::ResourceManager resources({
        {"nsp-0", "geniex-htp1", hexagonflow::BackendFamily::GenieX, 1U, 1000U},
    });
    hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::PredictedLatency,
                                     resources);
    auto task = make_task("uncalibrated", "smollm2-q4_0", 0, 0);
    task.allowed_backend_ids = {"geniex-htp1"};
    HF_REQUIRE(scheduler.submit(std::move(task)));
    const auto decisions = scheduler.dispatch_ready();
    HF_REQUIRE(decisions.size() == 1U);
    HF_REQUIRE(decisions.front().dispatched);
    HF_REQUIRE(decisions.front().resource_id == "nsp-0");
    HF_REQUIRE(decisions.front().reason ==
               "prediction-cold-start-priority-fallback");
    HF_REQUIRE(resources.snapshot("nsp-0").active_tasks == 1U);
    HF_REQUIRE(scheduler.complete("uncalibrated"));
    HF_REQUIRE(resources.snapshot("nsp-0").active_tasks == 0U);
}

HF_TEST(interference_aware_scheduler_uses_directional_observations) {
    auto predictor = std::make_shared<hexagonflow::PairwiseEmaPredictor>(1.0);
    hexagonflow::ResourceManager resources({
        {"active-resource", "backend-active", hexagonflow::BackendFamily::Cpu,
         1U, 1000U, "shared-domain"},
        {"shared-candidate", "backend-target", hexagonflow::BackendFamily::Gpu,
         1U, 1000U, "shared-domain"},
        {"isolated-candidate", "backend-isolated", hexagonflow::BackendFamily::GenieX,
         1U, 1000U, "isolated-domain"},
    });
    hexagonflow::Scheduler scheduler(
        hexagonflow::SchedulingPolicy::InterferenceAwareLatency, resources, {}, predictor,
        hexagonflow::ObservationSource::MockFixture);

    hexagonflow::Workload active;
    active.id = "active-asr";
    active.model_id = "competitor-asr";
    active.kind = hexagonflow::WorkloadKind::Asr;
    active.input_units = 128U;
    active.model_memory_bytes = 10U;
    active.service_estimates_by_backend_us = {{"backend-active", 100U}};
    active.allowed_backend_ids = {"backend-active"};
    HF_REQUIRE(scheduler.submit(active));
    const auto active_decisions = scheduler.dispatch_ready();
    HF_REQUIRE(active_decisions.size() == 1U);
    HF_REQUIRE(active_decisions.front().dispatched);
    HF_REQUIRE(active_decisions.front().resource_id == "active-resource");

    const hexagonflow::InterferenceQuery observed_query{
        "shared-domain",
        {"target-llm", "backend-target", hexagonflow::WorkloadKind::LlmDecode,
         256U, 64U, false},
        {{"competitor-asr", "backend-active", hexagonflow::WorkloadKind::Asr,
          128U, 0U, true}},
        hexagonflow::ObservationSource::MockFixture};
    scheduler.observe_interference({observed_query, 1000U, 1800U, "fixture-pair-1"});

    hexagonflow::Workload target;
    target.id = "target-llm";
    target.model_id = "target-llm";
    target.kind = hexagonflow::WorkloadKind::LlmDecode;
    target.input_units = 256U;
    target.requested_output_tokens = 64U;
    target.model_memory_bytes = 10U;
    target.service_estimates_by_backend_us = {
        {"backend-target", 1000U}, {"backend-isolated", 1500U}};
    target.allowed_backend_ids = {"backend-target", "backend-isolated"};
    HF_REQUIRE(scheduler.submit(target));
    const auto target_decisions = scheduler.dispatch_ready();
    HF_REQUIRE(target_decisions.size() == 1U);
    HF_REQUIRE(target_decisions.front().dispatched);
    HF_REQUIRE(target_decisions.front().resource_id == "isolated-candidate");

    HF_REQUIRE(scheduler.complete("active-asr"));
    HF_REQUIRE(scheduler.complete("target-llm"));
}

HF_TEST(interference_aware_scheduler_fails_closed_on_an_unseen_overlap) {
    auto predictor = std::make_shared<hexagonflow::PairwiseEmaPredictor>();
    hexagonflow::ResourceManager resources({
        {"active-resource", "backend-active", hexagonflow::BackendFamily::Cpu,
         1U, 1000U, "shared-domain"},
        {"shared-candidate", "backend-target", hexagonflow::BackendFamily::Gpu,
         1U, 1000U, "shared-domain"},
    });
    hexagonflow::Scheduler scheduler(
        hexagonflow::SchedulingPolicy::InterferenceAwareLatency, resources, {}, predictor,
        hexagonflow::ObservationSource::MockFixture);

    hexagonflow::Workload active;
    active.id = "active-task";
    active.model_id = "active-model";
    active.model_memory_bytes = 10U;
    active.service_estimates_by_backend_us = {{"backend-active", 100U}};
    active.allowed_backend_ids = {"backend-active"};
    HF_REQUIRE(scheduler.submit(active));
    HF_REQUIRE(scheduler.dispatch_ready().front().dispatched);

    hexagonflow::Workload target;
    target.id = "target-task";
    target.model_id = "target-model";
    target.model_memory_bytes = 10U;
    target.service_estimates_by_backend_us = {
        {"backend-active", 900U}, {"backend-target", 1000U}};
    target.allowed_backend_ids = {"backend-target"};
    HF_REQUIRE(scheduler.submit(target));
    const auto decisions = scheduler.dispatch_ready();
    HF_REQUIRE(decisions.size() == 1U);
    HF_REQUIRE(!decisions.front().dispatched);
    HF_REQUIRE(decisions.front().reason == "no-calibrated-interference-estimate");
    HF_REQUIRE(resources.snapshot("shared-candidate").active_tasks == 0U);
    HF_REQUIRE(scheduler.complete("active-task"));
}

HF_TEST(failed_task_releases_reservation_and_retains_reason) {
    hexagonflow::ResourceManager resources({
        {"sim-resource", "mock", hexagonflow::BackendFamily::Mock, 1U, 100U},
    });
    hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::Fifo,
                                     resources);
    HF_REQUIRE(scheduler.submit(make_task("a", "model-a", 0, 1)));
    HF_REQUIRE(scheduler.dispatch_ready().front().dispatched);
    HF_REQUIRE(scheduler.fail("a", "mock-error"));
    HF_REQUIRE(scheduler.task("a").state == hexagonflow::TaskState::Failed);
    HF_REQUIRE(scheduler.task("a").error == "mock-error");
    HF_REQUIRE(resources.snapshot("sim-resource").active_tasks == 0U);
}

HF_TEST(queued_task_can_be_cancelled_without_reservation) {
    hexagonflow::ResourceManager resources({
        {"sim-resource", "mock", hexagonflow::BackendFamily::Mock, 1U, 100U},
    });
    hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::Fifo,
                                     resources);
    HF_REQUIRE(scheduler.submit(make_task("a", "model-a", 0, 1)));
    HF_REQUIRE(scheduler.cancel_queued("a"));
    HF_REQUIRE(scheduler.task("a").state == hexagonflow::TaskState::Cancelled);
    HF_REQUIRE(resources.snapshot("sim-resource").active_tasks == 0U);
}
