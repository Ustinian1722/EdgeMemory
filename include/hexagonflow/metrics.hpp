#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>

namespace hexagonflow {

struct ExecutionRecord {
    std::string run_id;
    std::string task_id;
    std::string model_id;
    std::string resource_id;
    std::string backend_id;
    std::string status;
    bool simulated{false};
    std::uint64_t start_unix_us{0};
    std::uint64_t end_unix_us{0};
    std::uint64_t input_tokens{0};
    std::uint64_t output_tokens{0};
    bool token_counts_known{true};
    std::optional<std::uint64_t> peak_process_rss_bytes;
};

class MetricsSink {
public:
    virtual ~MetricsSink() = default;
    virtual void append(const ExecutionRecord& record) = 0;
};

class JsonlMetricsSink final : public MetricsSink {
public:
    explicit JsonlMetricsSink(const std::string& path);
    void append(const ExecutionRecord& record) override;

private:
    std::ofstream stream_;
};

}  // namespace hexagonflow
