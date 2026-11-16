#include "hexagonflow/interference_io.hpp"

#include "hexagonflow/scheduler.hpp"

#include <charconv>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace hexagonflow {
namespace {

constexpr std::string_view kHeader =
    "schema_version\tsource\trun_id\tresource_domain_id\t"
    "target_model_id\ttarget_backend_id\ttarget_kind\ttarget_input_units\t"
    "target_output_units\ttarget_model_resident\tisolated_latency_us\t"
    "concurrent_latency_us\tco_running";

std::vector<std::string_view> split(std::string_view value, char delimiter) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (true) {
        const auto end = value.find(delimiter, start);
        parts.push_back(value.substr(start, end == std::string_view::npos
                                               ? end : end - start));
        if (end == std::string_view::npos) break;
        start = end + 1U;
    }
    return parts;
}

bool safe_token(std::string_view value) {
    if (value.empty()) return false;
    for (const auto raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        const bool ascii_letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        const bool digit = c >= '0' && c <= '9';
        if (!ascii_letter && !digit && c != '_' && c != '-' && c != '.') return false;
    }
    return true;
}

std::uint64_t parse_u64(std::string_view value, std::size_t line_number,
                        const char* field) {
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (value.empty() || error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("line " + std::to_string(line_number) +
                                    ": invalid " + field);
    }
    return parsed;
}

bool parse_bool(std::string_view value, std::size_t line_number,
                const char* field) {
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                ": invalid " + field);
}

WorkloadKind parse_kind(std::string_view value, std::size_t line_number) {
    if (value == "llm_prefill") return WorkloadKind::LlmPrefill;
    if (value == "llm_decode") return WorkloadKind::LlmDecode;
    if (value == "asr") return WorkloadKind::Asr;
    if (value == "embedding") return WorkloadKind::Embedding;
    if (value == "custom") return WorkloadKind::Custom;
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                ": unsupported workload kind");
}

ObservationSource parse_source(std::string_view value, std::size_t line_number) {
    if (value == "hardware") return ObservationSource::Hardware;
    if (value == "mock_fixture") return ObservationSource::MockFixture;
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                ": unsupported observation source");
}

InterferenceWorkload parse_actor(std::string_view value,
                                 std::size_t line_number) {
    const auto fields = split(value, ':');
    if (fields.size() != 6U || !safe_token(fields[0]) || !safe_token(fields[1])) {
        throw std::invalid_argument("line " + std::to_string(line_number) +
                                    ": malformed co-running actor");
    }
    return {std::string(fields[0]), std::string(fields[1]),
            parse_kind(fields[2], line_number),
            parse_u64(fields[3], line_number, "co-runner input units"),
            parse_u64(fields[4], line_number, "co-runner output units"),
            parse_bool(fields[5], line_number, "co-runner residency")};
}

InterferenceObservation parse_row(std::string_view line,
                                  std::size_t line_number) {
    const auto fields = split(line, '\t');
    if (fields.size() != 13U || fields[0] != "1") {
        throw std::invalid_argument("line " + std::to_string(line_number) +
                                    ": invalid v1 observation row");
    }
    if (!safe_token(fields[2]) || !safe_token(fields[3]) ||
        !safe_token(fields[4]) || !safe_token(fields[5])) {
        throw std::invalid_argument("line " + std::to_string(line_number) +
                                    ": unsafe or empty identifier");
    }

    InterferenceQuery query;
    query.source = parse_source(fields[1], line_number);
    query.resource_domain_id = fields[3];
    query.target = {std::string(fields[4]), std::string(fields[5]),
                    parse_kind(fields[6], line_number),
                    parse_u64(fields[7], line_number, "target input units"),
                    parse_u64(fields[8], line_number, "target output units"),
                    parse_bool(fields[9], line_number, "target residency")};

    const auto isolated = parse_u64(fields[10], line_number, "isolated latency");
    const auto concurrent = parse_u64(fields[11], line_number, "concurrent latency");
    if (isolated == 0U) {
        throw std::invalid_argument("line " + std::to_string(line_number) +
                                    ": isolated latency must be positive");
    }
    if (fields[12].empty() || fields[12] == "-") {
        throw std::invalid_argument("line " + std::to_string(line_number) +
                                    ": at least one co-running actor is required");
    }
    for (const auto actor : split(fields[12], ',')) {
        query.co_running.push_back(parse_actor(actor, line_number));
    }
    return {std::move(query), isolated, concurrent, std::string(fields[2])};
}

}  // namespace

std::vector<InterferenceObservation> read_interference_observations_tsv(
    std::istream& input) {
    std::vector<InterferenceObservation> observations;
    std::string line;
    std::size_t line_number = 0;
    bool has_header = false;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (!has_header) {
            if (line != kHeader) {
                throw std::invalid_argument("line 1: unexpected observation profile header");
            }
            has_header = true;
            continue;
        }
        observations.push_back(parse_row(line, line_number));
    }
    if (!input.eof() && input.fail()) {
        throw std::invalid_argument("failed reading observation profile");
    }
    if (!has_header || observations.empty()) {
        throw std::invalid_argument("observation profile contains no data rows");
    }
    return observations;
}

std::size_t load_interference_observations_tsv(std::istream& input,
                                              Scheduler& scheduler) {
    const auto observations = read_interference_observations_tsv(input);
    for (const auto& observation : observations) {
        scheduler.observe_interference(observation);
    }
    return observations.size();
}

}  // namespace hexagonflow
