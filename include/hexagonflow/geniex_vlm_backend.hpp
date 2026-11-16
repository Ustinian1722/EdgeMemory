#pragma once

#include "hexagonflow/backend.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace hexagonflow {

struct GenieXVlmBackendConfig {
    std::string backend_id{"geniex-vlm"};
    std::string model_id;
    std::string model_path;
    std::string mmproj_path;
    std::string tokenizer_path;
    std::string plugin_id{"llama_cpp"};
    // Shared-library filename or absolute runtime path. Empty uses the platform default.
    std::string library_path;
    // Empty selects GenieX's runtime-resolved default; no physical HTP id is assumed.
    std::string device_id;
    // Empty keeps GenieX's default spare-session selection for the vision encoder.
    std::string vision_device_id;
    std::string plugin_directory;
    std::int32_t context_length{4096};
    std::int32_t threads{4};
    std::int32_t gpu_layers{-1};
    std::int32_t default_max_tokens{96};
};

/**
 * Long-lived VLM adapter around the official GenieX C SDK.
 * The constructor creates one runtime model handle; execute() resets its KV
 * state and reuses that resident model for each image/audio request.
 * Build this adapter only when a target-matched external GenieX SDK is present.
 */
class GenieXVlmBackend final : public InferenceBackend {
public:
    explicit GenieXVlmBackend(GenieXVlmBackendConfig config);
    ~GenieXVlmBackend() override;

    GenieXVlmBackend(const GenieXVlmBackend&) = delete;
    GenieXVlmBackend& operator=(const GenieXVlmBackend&) = delete;

    [[nodiscard]] std::string id() const override;
    [[nodiscard]] BackendFamily family() const override;
    BackendResult execute(const Workload& workload) override;

    [[nodiscard]] std::string runtime_version() const;
    [[nodiscard]] std::string plugin_version() const;

private:
    struct RuntimeHandle;
    GenieXVlmBackendConfig config_;
    std::unique_ptr<RuntimeHandle> runtime_;
    std::mutex execute_mutex_;
};

}  // namespace hexagonflow
