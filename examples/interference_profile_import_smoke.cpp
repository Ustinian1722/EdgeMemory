#include "hexagonflow/interference_io.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* source_name(hexagonflow::ObservationSource source) {
    switch (source) {
        case hexagonflow::ObservationSource::Hardware: return "hardware";
        case hexagonflow::ObservationSource::MockFixture: return "mock_fixture";
    }
    throw std::invalid_argument("unsupported observation source");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument(
                "usage: hexagonflow-interference-profile-check <private-profile.tsv>");
        }
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("cannot open observation profile");
        const auto records = hexagonflow::read_interference_observations_tsv(input);
        const auto source = records.front().query.source;
        for (const auto& record : records) {
            if (record.query.source != source) {
                throw std::invalid_argument("profile mixes evidence sources");
            }
        }

        auto predictor = std::make_shared<hexagonflow::PairwiseEmaPredictor>();
        hexagonflow::ResourceManager resources({
            {"profile-check-resource", "profile-check-backend",
             hexagonflow::BackendFamily::Mock, 1U, 1U},
        });
        hexagonflow::Scheduler scheduler(
            hexagonflow::SchedulingPolicy::InterferenceAwareLatency, resources, {},
            predictor, source);

        std::ifstream import_stream(argv[1]);
        if (!import_stream) throw std::runtime_error("cannot reopen observation profile");
        const auto imported =
            hexagonflow::load_interference_observations_tsv(import_stream, scheduler);
        std::size_t matched = 0U;
        for (const auto& record : records) {
            if (predictor->estimate_penalty_us(record.query).has_value()) ++matched;
        }
        if (imported != records.size() || matched != records.size()) {
            throw std::runtime_error("not every profile row reached the predictor");
        }
        std::cout << "{\"status\":\"loaded\",\"observations\":" << imported
                  << ",\"matched_estimates\":" << matched
                  << ",\"source\":\"" << source_name(source)
                  << "\",\"device_execution\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interference profile check: " << error.what() << '\n';
        return 2;
    }
}
