#pragma once

#include "hexagonflow/backend.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace hexagonflow {

// A bounded process adapter for an externally installed inference engine.
// Every execution starts a fresh process: model residency is not retained.
struct ProcessBackendOptions {
    std::string executable;
    std::vector<std::string> arguments;
    std::string output_path;
    std::chrono::milliseconds timeout{60000};
    std::string backend_id{"cpu-process"};
    BackendFamily backend_family{BackendFamily::Cpu};
    bool append_output{false};
};

class ProcessBackend final : public InferenceBackend {
public:
    explicit ProcessBackend(ProcessBackendOptions options);
    [[nodiscard]] std::string id() const override;
    [[nodiscard]] BackendFamily family() const override;
    BackendResult execute(const Workload& workload) override;

private:
    ProcessBackendOptions options_;
};

}  // namespace hexagonflow
