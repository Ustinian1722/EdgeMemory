#include "hexagonflow/interference.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace hexagonflow {

PairwiseEmaPredictor::PairwiseEmaPredictor(double alpha) : alpha_(alpha) {
    if (!std::isfinite(alpha_) || alpha_ <= 0.0 || alpha_ > 1.0) {
        throw std::invalid_argument("EMA alpha must be in (0, 1]");
    }
}

PairwiseEmaPredictor::PairKey PairwiseEmaPredictor::make_key(
    const InterferenceQuery& query,
    const InterferenceWorkload& interferer) {
    auto target = query.target;
    auto other = interferer;
    target.input_units = input_bucket(target.input_units);
    target.output_units = input_bucket(target.output_units);
    other.input_units = input_bucket(other.input_units);
    other.output_units = input_bucket(other.output_units);
    return {query.resource_domain_id, std::move(target), std::move(other), query.source};
}

std::uint32_t PairwiseEmaPredictor::input_bucket(
    std::uint64_t input_units) noexcept {
    return static_cast<std::uint32_t>(std::bit_width(input_units));
}

void PairwiseEmaPredictor::observe(
    const InterferenceObservation& observation) {
    const auto& query = observation.query;
    if (query.resource_domain_id.empty() || query.target.model_id.empty() ||
        query.target.backend_id.empty() || observation.run_id.empty() ||
        observation.isolated_latency_us == 0U || query.co_running.empty()) {
        return;
    }

    std::map<PairKey, std::size_t> pair_counts;
    for (const auto& workload : query.co_running) {
        if (workload.model_id.empty() || workload.backend_id.empty()) return;
        ++pair_counts[make_key(query, workload)];
    }
    std::size_t total_instances = 0U;
    for (const auto& [key, count] : pair_counts) {
        (void)key;
        total_instances += count;
    }
    if (total_instances == 0U) {
        return;
    }

    const auto delta = observation.concurrent_latency_us >
                               observation.isolated_latency_us
                           ? observation.concurrent_latency_us -
                                 observation.isolated_latency_us
                           : 0U;
    const double per_pair =
        static_cast<double>(delta) / static_cast<double>(total_instances);

    const std::lock_guard lock(mutex_);
    for (const auto& [key, count] : pair_counts) {
        (void)count;
        auto& estimate = estimates_[key];
        if (estimate.samples == 0U) {
            estimate.penalty_us = per_pair;
        } else {
            estimate.penalty_us =
                alpha_ * per_pair + (1.0 - alpha_) * estimate.penalty_us;
        }
        ++estimate.samples;
    }
}

std::optional<std::uint64_t> PairwiseEmaPredictor::estimate_penalty_us(
    const InterferenceQuery& query) const {
    if (query.resource_domain_id.empty() || query.target.model_id.empty() ||
        query.target.backend_id.empty()) {
        return std::nullopt;
    }
    if (query.co_running.empty()) {
        return 0U;
    }

    std::map<PairKey, std::size_t> pair_counts;
    for (const auto& workload : query.co_running) {
        if (workload.model_id.empty() || workload.backend_id.empty()) {
            return std::nullopt;
        }
        ++pair_counts[make_key(query, workload)];
    }
    if (pair_counts.empty()) {
        return 0U;
    }

    long double total = 0.0L;
    const std::lock_guard lock(mutex_);
    for (const auto& [key, count] : pair_counts) {
        const auto found = estimates_.find(key);
        if (found == estimates_.end() || found->second.samples == 0U) {
            return std::nullopt;
        }
        total += static_cast<long double>(found->second.penalty_us) *
                 static_cast<long double>(count);
    }
    const auto maximum = static_cast<long double>(
        std::numeric_limits<std::uint64_t>::max());
    if (total >= maximum) return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t>(std::round(total));
}

}  // namespace hexagonflow
