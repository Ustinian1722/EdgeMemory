#include "hexagonflow/interference.hpp"
#include "test_framework.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

hexagonflow::InterferenceWorkload actor(std::string model,
                                        std::string backend,
                                        hexagonflow::WorkloadKind kind,
                                        std::uint64_t input_units,
                                        std::uint64_t output_units,
                                        bool resident) {
    return {std::move(model), std::move(backend), kind, input_units, output_units,
            resident};
}

hexagonflow::InterferenceQuery query(hexagonflow::ObservationSource source) {
    return {"domain-shared",
            actor("target-llm", "backend-npu", hexagonflow::WorkloadKind::LlmDecode,
                  128U, 32U, false),
            {actor("competitor-asr", "backend-cpu", hexagonflow::WorkloadKind::Asr,
                   64U, 0U, true)},
            source};
}

}  // namespace

HF_TEST(interference_estimates_are_directional_and_evidence_scoped) {
    hexagonflow::PairwiseEmaPredictor predictor(0.5);
    auto observed = query(hexagonflow::ObservationSource::MockFixture);
    predictor.observe({observed, 1000U, 1400U, "fixture-1"});

    const auto mock = predictor.estimate_penalty_us(observed);
    HF_REQUIRE(mock.has_value());
    HF_REQUIRE(mock.value() == 400U);

    auto hardware = observed;
    hardware.source = hexagonflow::ObservationSource::Hardware;
    HF_REQUIRE(!predictor.estimate_penalty_us(hardware).has_value());

    auto reversed = observed;
    reversed.target = observed.co_running.front();
    reversed.co_running = {observed.target};
    HF_REQUIRE(!predictor.estimate_penalty_us(reversed).has_value());

    auto incomplete = observed;
    incomplete.co_running.push_back(actor("", "", hexagonflow::WorkloadKind::Custom,
                                               0U, 0U, false));
    HF_REQUIRE(!predictor.estimate_penalty_us(incomplete).has_value());
}

HF_TEST(interference_ema_deduplicates_and_matches_context_buckets) {
    hexagonflow::PairwiseEmaPredictor predictor(0.5);
    auto observed = query(hexagonflow::ObservationSource::MockFixture);
    observed.co_running.push_back(observed.co_running.front());
    predictor.observe({observed, 1000U, 1400U, "fixture-1"});
    predictor.observe({query(hexagonflow::ObservationSource::MockFixture),
                       1000U, 1600U, "fixture-2"});

    auto same_bucket = query(hexagonflow::ObservationSource::MockFixture);
    same_bucket.target.input_units = 250U;
    same_bucket.target.output_units = 60U;
    same_bucket.co_running.front().input_units = 127U;
    same_bucket.co_running.front().output_units = 0U;
    const auto estimate = predictor.estimate_penalty_us(same_bucket);
    HF_REQUIRE(estimate.has_value());
    HF_REQUIRE(estimate.value() == 400U);

    same_bucket.co_running.push_back(same_bucket.co_running.front());
    const auto two_instances = predictor.estimate_penalty_us(same_bucket);
    HF_REQUIRE(two_instances.has_value());
    HF_REQUIRE(two_instances.value() == 800U);

    auto different_input_bucket = same_bucket;
    different_input_bucket.target.input_units = 256U;
    HF_REQUIRE(!predictor.estimate_penalty_us(different_input_bucket).has_value());

    auto different_output_bucket = same_bucket;
    different_output_bucket.target.output_units = 64U;
    HF_REQUIRE(!predictor.estimate_penalty_us(different_output_bucket).has_value());

    auto different_domain = same_bucket;
    different_domain.resource_domain_id = "domain-other";
    HF_REQUIRE(!predictor.estimate_penalty_us(different_domain).has_value());

    auto different_device = same_bucket;
    different_device.target.backend_id = "backend-other";
    HF_REQUIRE(!predictor.estimate_penalty_us(different_device).has_value());

    auto different_residency = same_bucket;
    different_residency.target.model_resident = true;
    HF_REQUIRE(!predictor.estimate_penalty_us(different_residency).has_value());
}

HF_TEST(interference_predictor_rejects_nonfinite_alpha) {
    bool rejected = false;
    try {
        hexagonflow::PairwiseEmaPredictor predictor(
            std::numeric_limits<double>::quiet_NaN());
        (void)predictor;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    HF_REQUIRE(rejected);
}
