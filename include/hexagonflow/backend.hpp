#pragma once

#include "hexagonflow/types.hpp"

#include <optional>
#include <string>
#include <utility>

namespace hexagonflow {

struct BackendResult {
    BackendResult() = default;
    BackendResult(bool succeeded, bool simulated_result, std::uint64_t tokens,
                  std::string status)
        : success(succeeded), simulated(simulated_result), output_tokens(tokens),
          message(std::move(status)) {}

    bool success{false};
    bool simulated{false};
    std::uint64_t output_tokens{0};
    std::string message;
    std::uint64_t ttft_us{0};
    std::uint64_t media_us{0};
    std::uint64_t prefill_us{0};
    std::uint64_t decode_us{0};
    std::uint64_t queue_wait_us{0};
    std::uint64_t worker_elapsed_us{0};
    double prefill_tokens_per_second{0.0};
    double decode_tokens_per_second{0.0};
    std::string output_text;
    // Peak RSS for the direct child process when the host reports it. This is
    // not accelerator memory and does not include the child's descendants.
    std::optional<std::uint64_t> peak_process_rss_bytes;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    [[nodiscard]] virtual std::string id() const = 0;
    [[nodiscard]] virtual BackendFamily family() const = 0;
    virtual BackendResult execute(const Workload& workload) = 0;
};

// The mock backend validates integration and lifecycle behavior only. It does
// not run a model and never produces hardware performance measurements.
class MockBackend final : public InferenceBackend {
public:
    explicit MockBackend(std::string backend_id = "mock");

    [[nodiscard]] std::string id() const override;
    [[nodiscard]] BackendFamily family() const override;
    BackendResult execute(const Workload& workload) override;

    void fail_model(std::string model_id);
    void clear_failure();

private:
    std::string backend_id_;
    std::string failing_model_id_;
};

}  // namespace hexagonflow
