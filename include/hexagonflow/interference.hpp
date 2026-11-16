#pragma once

#include "hexagonflow/types.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace hexagonflow {

enum class ObservationSource {
    Hardware,
    MockFixture,
};

struct InterferenceWorkload {
    std::string model_id;
    std::string backend_id;
    WorkloadKind kind{WorkloadKind::Custom};
    // Caller-defined scale for this workload kind; the predictor maps it to a
    // power-of-two bucket. Zero remains the unknown-input bucket.
    std::uint64_t input_units{0};
    // Separate output scale for autoregressive decode workloads.
    std::uint64_t output_units{0};
    // ResourceManager's residency accounting, not a measured runtime load state.
    bool model_resident{false};

    friend bool operator<(const InterferenceWorkload& left,
                          const InterferenceWorkload& right) {
        return std::tie(left.model_id, left.backend_id, left.kind, left.input_units,
                        left.output_units, left.model_resident) <
               std::tie(right.model_id, right.backend_id, right.kind, right.input_units,
                        right.output_units, right.model_resident);
    }
};

struct InterferenceQuery {
    std::string resource_domain_id;
    InterferenceWorkload target;
    // One entry per active request. Repeated signatures preserve concurrency.
    std::vector<InterferenceWorkload> co_running;
    ObservationSource source{ObservationSource::Hardware};
};

struct InterferenceObservation {
    InterferenceQuery query;
    std::uint64_t isolated_latency_us{0};
    std::uint64_t concurrent_latency_us{0};
    std::string run_id;
};

class InterferencePredictor {
public:
    virtual ~InterferencePredictor() = default;

    virtual void observe(const InterferenceObservation& observation) = 0;
    [[nodiscard]] virtual std::optional<std::uint64_t> estimate_penalty_us(
        const InterferenceQuery& query) const = 0;
};

// A transparent online estimator. Hardware and mock-fixture observations are
// kept in separate buckets; production estimates default to hardware evidence.
class PairwiseEmaPredictor final : public InterferencePredictor {
public:
    explicit PairwiseEmaPredictor(double alpha = 0.25);

    void observe(const InterferenceObservation& observation) override;
    [[nodiscard]] std::optional<std::uint64_t> estimate_penalty_us(
        const InterferenceQuery& query) const override;

private:
    struct PairKey {
        std::string resource_domain_id;
        InterferenceWorkload target;
        InterferenceWorkload interferer;
        ObservationSource source{ObservationSource::Hardware};

        friend bool operator<(const PairKey& left, const PairKey& right) {
            return std::tie(left.source, left.resource_domain_id, left.target,
                            left.interferer) <
                   std::tie(right.source, right.resource_domain_id, right.target,
                            right.interferer);
        }
    };

    struct Estimate {
        double penalty_us{0.0};
        std::uint64_t samples{0};
    };

    static PairKey make_key(const InterferenceQuery& query,
                            const InterferenceWorkload& interferer);
    static std::uint32_t input_bucket(std::uint64_t input_units) noexcept;

    double alpha_;
    std::map<PairKey, Estimate> estimates_;
    mutable std::mutex mutex_;
};

}  // namespace hexagonflow
