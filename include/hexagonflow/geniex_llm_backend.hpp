#pragma once

#include "hexagonflow/backend.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace hexagonflow {

struct GenieXLLMBackendConfig {
    std::string backend_id{"geniex-llm"};
    std::string model_id;
    std::string model_path;
    std::string tokenizer_path;
    std::string plugin_id{"llama_cpp"};
    std::string library_path;
    std::string device_id;
    std::string plugin_directory;
    std::int32_t context_length{2048};
    std::int32_t threads{4};
    std::int32_t batch_size{2048};
    std::int32_t physical_batch_size{128};
    std::int32_t gpu_layers{-1};
    std::int32_t default_max_tokens{192};
};

/** Long-lived text-only GenieX C API adapter. The model is created once and
 * its KV state is reset between requests; inference stays in the upstream
 * GenieX llama_cpp plugin. */
class GenieXLLMBackend final : public InferenceBackend {
public:
    explicit GenieXLLMBackend(GenieXLLMBackendConfig config);
    ~GenieXLLMBackend() override;

    GenieXLLMBackend(const GenieXLLMBackend&) = delete;
    GenieXLLMBackend& operator=(const GenieXLLMBackend&) = delete;

    [[nodiscard]] std::string id() const override;
    [[nodiscard]] BackendFamily family() const override;
    BackendResult execute(const Workload& workload) override;
    [[nodiscard]] std::string runtime_version() const;
    [[nodiscard]] std::string plugin_version() const;

private:
    struct RuntimeHandle;
    GenieXLLMBackendConfig config_;
    std::unique_ptr<RuntimeHandle> runtime_;
    std::mutex execute_mutex_;
};

}  // namespace hexagonflow
