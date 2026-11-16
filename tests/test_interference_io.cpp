#include "hexagonflow/interference_io.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"
#include "test_framework.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* kHeader =
    "schema_version\tsource\trun_id\tresource_domain_id\t"
    "target_model_id\ttarget_backend_id\ttarget_kind\ttarget_input_units\t"
    "target_output_units\ttarget_model_resident\tisolated_latency_us\t"
    "concurrent_latency_us\tco_running\n";

std::string fixture_row(const std::string& run_id = "fixture-run-1") {
    return "1\tmock_fixture\t" + run_id +
           "\tshared-domain\ttarget-llm\tbackend-target\tllm_decode"
           "\t128\t32\t0\t1000\t1300"
           "\tcompetitor-asr:backend-active:asr:64:0:1\n";
}

}  // namespace

HF_TEST(interference_tsv_reader_preserves_context_and_repeated_actors) {
    const std::string repeated =
        "competitor-asr:backend-active:asr:64:0:1,"
        "competitor-asr:backend-active:asr:64:0:1";
    const std::string row =
        "1\tmock_fixture\tfixture-run-1\tshared-domain\ttarget-llm"
        "\tbackend-target\tllm_decode\t128\t32\t0\t1000\t1300\t" +
        repeated + "\n";
    std::istringstream input(std::string(kHeader) + row);
    const auto observations = hexagonflow::read_interference_observations_tsv(input);
    HF_REQUIRE(observations.size() == 1U);
    HF_REQUIRE(observations.front().query.source ==
               hexagonflow::ObservationSource::MockFixture);
    HF_REQUIRE(observations.front().query.target.input_units == 128U);
    HF_REQUIRE(observations.front().query.target.output_units == 32U);
    HF_REQUIRE(!observations.front().query.target.model_resident);
    HF_REQUIRE(observations.front().query.co_running.size() == 2U);
    HF_REQUIRE(observations.front().query.co_running.front().model_resident);
    HF_REQUIRE(observations.front().isolated_latency_us == 1000U);
    HF_REQUIRE(observations.front().concurrent_latency_us == 1300U);
}

HF_TEST(interference_tsv_reader_rejects_unknown_schema_and_malformed_rows) {
    std::istringstream bad_header("wrong-header\n");
    bool header_rejected = false;
    try {
        (void)hexagonflow::read_interference_observations_tsv(bad_header);
    } catch (const std::invalid_argument&) {
        header_rejected = true;
    }
    HF_REQUIRE(header_rejected);

    auto bad_row = fixture_row();
    const auto residency = bad_row.find("\t0\t1000\t");
    HF_REQUIRE(residency != std::string::npos);
    bad_row.replace(residency, 3U, "\t2\t");
    std::istringstream malformed(std::string(kHeader) + bad_row);
    bool row_rejected = false;
    try {
        (void)hexagonflow::read_interference_observations_tsv(malformed);
    } catch (const std::invalid_argument&) {
        row_rejected = true;
    }
    HF_REQUIRE(row_rejected);
}

HF_TEST(interference_tsv_loader_updates_scheduler_before_placement) {
    auto predictor = std::make_shared<hexagonflow::PairwiseEmaPredictor>(1.0);
    hexagonflow::ResourceManager resources({
        {"active-resource", "backend-active", hexagonflow::BackendFamily::Cpu,
         1U, 1000U, "shared-domain"},
        {"shared-target", "backend-target", hexagonflow::BackendFamily::Gpu,
         1U, 1000U, "shared-domain"},
        {"isolated-target", "backend-isolated", hexagonflow::BackendFamily::GenieX,
         1U, 1000U, "isolated-domain"},
    });
    hexagonflow::Scheduler scheduler(
        hexagonflow::SchedulingPolicy::InterferenceAwareLatency, resources, {}, predictor,
        hexagonflow::ObservationSource::MockFixture);

    hexagonflow::Workload active;
    active.id = "active-asr";
    active.model_id = "competitor-asr";
    active.kind = hexagonflow::WorkloadKind::Asr;
    active.input_units = 64U;
    active.model_memory_bytes = 10U;
    active.service_estimates_by_backend_us = {{"backend-active", 100U}};
    active.allowed_backend_ids = {"backend-active"};
    HF_REQUIRE(scheduler.submit(active));
    HF_REQUIRE(scheduler.dispatch_ready().front().dispatched);

    std::istringstream profile(std::string(kHeader) + fixture_row());
    HF_REQUIRE(hexagonflow::load_interference_observations_tsv(profile, scheduler) == 1U);

    hexagonflow::Workload target;
    target.id = "target-request";
    target.model_id = "target-llm";
    target.kind = hexagonflow::WorkloadKind::LlmDecode;
    target.input_units = 128U;
    target.requested_output_tokens = 32U;
    target.model_memory_bytes = 10U;
    target.allowed_backend_ids = {"backend-target", "backend-isolated"};
    target.service_estimates_by_backend_us = {
        {"backend-target", 1000U}, {"backend-isolated", 900U}};
    HF_REQUIRE(scheduler.submit(target));
    const auto decisions = scheduler.dispatch_ready();
    HF_REQUIRE(decisions.size() == 1U);
    HF_REQUIRE(decisions.front().dispatched);
    HF_REQUIRE(decisions.front().resource_id == "isolated-target");

    HF_REQUIRE(scheduler.complete("active-asr"));
    HF_REQUIRE(scheduler.complete("target-request"));
}

HF_TEST(interference_tsv_loader_parses_before_updating_predictor) {
    auto predictor = std::make_shared<hexagonflow::PairwiseEmaPredictor>(1.0);
    hexagonflow::ResourceManager resources({
        {"resource", "backend-target", hexagonflow::BackendFamily::Cpu,
         1U, 1000U, "shared-domain"},
    });
    hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::InterferenceAwareLatency,
                                     resources, {}, predictor,
                                     hexagonflow::ObservationSource::MockFixture);
    std::istringstream profile(std::string(kHeader) + fixture_row() +
                               "1\tunknown\tbad\tshared-domain\tbad\tbackend-target"
                               "\tllm_decode\t1\t1\t0\t1\t1\tother:backend:asr:1:0:1\n");
    bool rejected = false;
    try {
        (void)hexagonflow::load_interference_observations_tsv(profile, scheduler);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    HF_REQUIRE(rejected);
    const hexagonflow::InterferenceQuery query{
        "shared-domain",
        {"target-llm", "backend-target", hexagonflow::WorkloadKind::LlmDecode,
         128U, 32U, false},
        {{"competitor-asr", "backend-active", hexagonflow::WorkloadKind::Asr,
          64U, 0U, true}},
        hexagonflow::ObservationSource::MockFixture};
    HF_REQUIRE(!predictor->estimate_penalty_us(query).has_value());
}
