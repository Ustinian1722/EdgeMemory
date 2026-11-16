#include "hexagonflow/geniex_llm_backend.hpp"

#include <geniex.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace hexagonflow {
namespace {

std::mutex runtime_mutex;
std::mutex model_create_mutex;
std::weak_ptr<class RuntimeApi> runtime_weak;

void set_plugin_directory(const std::string& directory) {
    if (directory.empty()) return;
    const char* current = std::getenv("GENIEX_PLUGIN_PATH");
    if (current && *current && directory != current) {
        throw std::runtime_error("geniex-plugin-path-already-initialized");
    }
#if defined(_WIN32)
    if (_putenv_s("GENIEX_PLUGIN_PATH", directory.c_str()) != 0) {
        throw std::runtime_error("geniex-plugin-path-configuration-failed");
    }
#else
    if (setenv("GENIEX_PLUGIN_PATH", directory.c_str(), 0) != 0) {
        throw std::runtime_error("geniex-plugin-path-configuration-failed");
    }
#endif
}

std::uint64_t positive_u64(std::int64_t value) {
    return value > 0 ? static_cast<std::uint64_t>(value) : 0U;
}

void free_template_grammar(geniex_ChatTemplateGrammar& grammar,
                           void (*free_fn)(void*)) {
    if (grammar.triggers) {
        for (std::int32_t index = 0; index < grammar.trigger_count; ++index) {
            if (grammar.triggers[index].value) free_fn(grammar.triggers[index].value);
        }
        free_fn(grammar.triggers);
    }
    if (grammar.grammar) free_fn(grammar.grammar);
    if (grammar.generation_prompt) free_fn(grammar.generation_prompt);
    if (grammar.preserved_tokens) free_fn(grammar.preserved_tokens);
    grammar = {};
}

class RuntimeApi {
public:
    using SetLogLevel = decltype(&geniex_set_log_level);
    using Init = decltype(&geniex_init);
    using Deinit = decltype(&geniex_deinit);
    using GetPluginList = decltype(&geniex_get_plugin_list);
    using Free = decltype(&geniex_free);
    using Version = decltype(&geniex_version);
    using PluginVersion = decltype(&geniex_get_plugin_version);
    using LlmCreate = decltype(&geniex_llm_create);
    using LlmDestroy = decltype(&geniex_llm_destroy);
    using LlmReset = decltype(&geniex_llm_reset);
    using LlmApplyTemplate = decltype(&geniex_llm_apply_chat_template);
    using LlmGenerate = decltype(&geniex_llm_generate);

    explicit RuntimeApi(const std::string& requested_library)
        : library_path_(requested_library.empty() ? default_library_name() : requested_library) {
#if defined(_WIN32)
        library_ = LoadLibraryA(library_path_.c_str());
#else
        library_ = dlopen(library_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
        if (!library_) throw std::runtime_error("geniex-library-load-failed");
        try {
            set_log_level = symbol<SetLogLevel>("geniex_set_log_level");
            init = symbol<Init>("geniex_init");
            deinit = symbol<Deinit>("geniex_deinit");
            get_plugin_list = symbol<GetPluginList>("geniex_get_plugin_list");
            free = symbol<Free>("geniex_free");
            version = symbol<Version>("geniex_version");
            plugin_version = symbol<PluginVersion>("geniex_get_plugin_version");
            llm_create = symbol<LlmCreate>("geniex_llm_create");
            llm_destroy = symbol<LlmDestroy>("geniex_llm_destroy");
            llm_reset = symbol<LlmReset>("geniex_llm_reset");
            llm_apply_template = symbol<LlmApplyTemplate>("geniex_llm_apply_chat_template");
            llm_generate = symbol<LlmGenerate>("geniex_llm_generate");
            if (set_log_level(GENIEX_LOG_LEVEL_ERROR) != GENIEX_SUCCESS) {
                throw std::runtime_error("geniex-log-level-configuration-failed");
            }
            if (init() != GENIEX_SUCCESS) throw std::runtime_error("geniex-init-failed");
            initialized_ = true;
        } catch (...) {
            unload();
            throw;
        }
    }

    ~RuntimeApi() {
        if (initialized_ && deinit) (void)deinit();
        unload();
    }

    RuntimeApi(const RuntimeApi&) = delete;
    RuntimeApi& operator=(const RuntimeApi&) = delete;

    std::string library_path() const { return library_path_; }
    SetLogLevel set_log_level{};
    Init init{};
    Deinit deinit{};
    GetPluginList get_plugin_list{};
    Free free{};
    Version version{};
    PluginVersion plugin_version{};
    LlmCreate llm_create{};
    LlmDestroy llm_destroy{};
    LlmReset llm_reset{};
    LlmApplyTemplate llm_apply_template{};
    LlmGenerate llm_generate{};

private:
    static std::string default_library_name() {
#if defined(_WIN32)
        return "geniex.dll";
#else
        return "libgeniex.so";
#endif
    }

    template <typename Function>
    Function symbol(const char* name) {
#if defined(_WIN32)
        const auto address = GetProcAddress(library_, name);
#else
        const auto address = dlsym(library_, name);
#endif
        if (!address) throw std::runtime_error(std::string("geniex-symbol-missing-") + name);
        return reinterpret_cast<Function>(address);
    }

    void unload() noexcept {
        if (!library_) return;
#if defined(_WIN32)
        (void)FreeLibrary(library_);
#else
        (void)dlclose(library_);
#endif
        library_ = nullptr;
    }

    std::string library_path_;
#if defined(_WIN32)
    HMODULE library_{nullptr};
#else
    void* library_{nullptr};
#endif
    bool initialized_{false};
};

std::shared_ptr<RuntimeApi> acquire_runtime(const std::string& library_path,
                                            const std::string& plugin_directory) {
    std::lock_guard lock(runtime_mutex);
    set_plugin_directory(plugin_directory);
    auto api = runtime_weak.lock();
    if (api) {
        if (!library_path.empty() && api->library_path() != library_path) {
            throw std::runtime_error("geniex-runtime-already-loaded-from-another-path");
        }
        return api;
    }
    api = std::make_shared<RuntimeApi>(library_path);
    runtime_weak = api;
    return api;
}

bool plugin_is_available(RuntimeApi& api, std::string_view requested) {
    geniex_GetPluginListOutput output{};
    if (api.get_plugin_list(&output) != GENIEX_SUCCESS) return false;
    bool found = false;
    for (std::int32_t index = 0; index < output.plugin_count; ++index) {
        if (output.plugin_ids && output.plugin_ids[index] &&
            requested == output.plugin_ids[index]) {
            found = true;
            break;
        }
    }
    if (output.plugin_ids) api.free(output.plugin_ids);
    return found;
}

}  // namespace

struct GenieXLLMBackend::RuntimeHandle {
    geniex_LLM* llm{nullptr};
    std::shared_ptr<RuntimeApi> api;
    std::string runtime_version;
    std::string plugin_version;

    ~RuntimeHandle() {
        if (llm) {
            std::lock_guard lock(model_create_mutex);
            (void)api->llm_destroy(llm);
            llm = nullptr;
        }
        api.reset();
    }
};

GenieXLLMBackend::GenieXLLMBackend(GenieXLLMBackendConfig config)
    : config_(std::move(config)), runtime_(std::make_unique<RuntimeHandle>()) {
    if (config_.backend_id.empty() || config_.model_id.empty() || config_.model_path.empty() ||
        config_.plugin_id.empty() || config_.context_length < 1 || config_.threads < 1 ||
        config_.default_max_tokens < 1) {
        throw std::invalid_argument("invalid-geniex-llm-configuration");
    }
    if (!std::filesystem::is_regular_file(config_.model_path)) {
        throw std::invalid_argument("geniex-llm-model-artifact-missing");
    }
    if (!config_.tokenizer_path.empty() &&
        !std::filesystem::is_regular_file(config_.tokenizer_path)) {
        throw std::invalid_argument("geniex-llm-tokenizer-artifact-missing");
    }

    runtime_->api = acquire_runtime(config_.library_path, config_.plugin_directory);
    if (!plugin_is_available(*runtime_->api, config_.plugin_id)) {
        throw std::runtime_error("geniex-plugin-unavailable");
    }

    geniex_LlmCreateInput input{};
    input.model_path = config_.model_path.c_str();
    input.tokenizer_path = config_.tokenizer_path.empty() ? nullptr : config_.tokenizer_path.c_str();
    input.plugin_id = config_.plugin_id.c_str();
    input.device_id = config_.device_id.empty() ? nullptr : config_.device_id.c_str();
    input.config.n_ctx = config_.context_length;
    input.config.n_threads = config_.threads;
    input.config.n_threads_batch = config_.threads;
    input.config.n_batch = config_.batch_size;
    input.config.n_ubatch = config_.physical_batch_size;
    input.config.n_seq_max = 1;
    input.config.n_gpu_layers = config_.gpu_layers;
    input.config.power_mode = GENIEX_POWER_MODE_BURST;

    std::lock_guard lock(model_create_mutex);
    const auto rc = runtime_->api->llm_create(&input, &runtime_->llm);
    if (rc != GENIEX_SUCCESS || !runtime_->llm) {
        throw std::runtime_error("geniex-llm-create-" + std::to_string(rc));
    }
    const char* version = runtime_->api->version();
    runtime_->runtime_version = version ? version : "unknown";
    const char* plugin = runtime_->api->plugin_version(config_.plugin_id.c_str());
    runtime_->plugin_version = plugin ? plugin : "unknown";
}

GenieXLLMBackend::~GenieXLLMBackend() = default;

std::string GenieXLLMBackend::id() const { return config_.backend_id; }
BackendFamily GenieXLLMBackend::family() const { return BackendFamily::GenieX; }

BackendResult GenieXLLMBackend::execute(const Workload& workload) {
    std::lock_guard execute_lock(execute_mutex_);
    BackendResult result;
    result.simulated = false;
    if (!runtime_ || !runtime_->llm) {
        result.message = "geniex-llm-handle-unavailable";
        return result;
    }
    if (workload.model_id != config_.model_id) {
        result.message = "worker-model-mismatch";
        return result;
    }
    if (workload.prompt_utf8.empty()) {
        result.message = "llm-prompt-missing";
        return result;
    }

    auto& api = *runtime_->api;
    const auto started = std::chrono::steady_clock::now();
    const auto reset_rc = api.llm_reset(runtime_->llm);
    if (reset_rc != GENIEX_SUCCESS) {
        result.message = "geniex-llm-reset-" + std::to_string(reset_rc);
        return result;
    }

    geniex_LlmChatMessage message{};
    message.role = "user";
    message.content = workload.prompt_utf8.c_str();
    geniex_LlmApplyChatTemplateInput template_input{};
    template_input.messages = &message;
    template_input.message_count = 1;
    template_input.enable_thinking = false;
    template_input.add_generation_prompt = true;
    geniex_LlmApplyChatTemplateOutput template_output{};
    const auto template_rc = api.llm_apply_template(runtime_->llm, &template_input, &template_output);
    if (template_rc != GENIEX_SUCCESS || !template_output.formatted_text) {
        if (template_output.formatted_text) api.free(template_output.formatted_text);
        free_template_grammar(template_output.chat_template_grammar, api.free);
        result.message = "geniex-llm-template-" + std::to_string(template_rc);
        return result;
    }

    geniex_SamplerConfig sampler{};
    sampler.temperature = 0.0F;
    sampler.top_p = 1.0F;
    sampler.repetition_penalty = 1.0F;
    sampler.seed = 42;
    sampler.chat_template_grammar = template_output.chat_template_grammar;
    geniex_GenerationConfig generation{};
    const auto token_limit = workload.requested_output_tokens > 0
        ? workload.requested_output_tokens : static_cast<std::uint64_t>(config_.default_max_tokens);
    generation.max_tokens = static_cast<std::int32_t>(std::min<std::uint64_t>(
        token_limit, static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())));
    generation.sampler_config = &sampler;
    geniex_LlmGenerateInput generate_input{};
    generate_input.prompt_utf8 = template_output.formatted_text;
    generate_input.config = &generation;
    geniex_LlmGenerateOutput generate_output{};
    const auto generate_rc = api.llm_generate(runtime_->llm, &generate_input, &generate_output);
    api.free(template_output.formatted_text);
    free_template_grammar(template_output.chat_template_grammar, api.free);
    if (generate_rc != GENIEX_SUCCESS) {
        if (generate_output.full_text) api.free(generate_output.full_text);
        result.message = "geniex-llm-generate-" + std::to_string(generate_rc);
        return result;
    }

    result.success = true;
    result.message = "completed";
    result.output_text = generate_output.full_text ? generate_output.full_text : "";
    result.output_tokens = positive_u64(generate_output.profile_data.generated_tokens);
    result.ttft_us = positive_u64(generate_output.profile_data.ttft);
    result.prefill_us = positive_u64(generate_output.profile_data.prompt_time);
    result.decode_us = positive_u64(generate_output.profile_data.decode_time);
    result.prefill_tokens_per_second = std::max(0.0, generate_output.profile_data.prefill_speed);
    result.decode_tokens_per_second = std::max(0.0, generate_output.profile_data.decoding_speed);
    result.worker_elapsed_us = static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
    if (generate_output.full_text) api.free(generate_output.full_text);
    return result;
}

std::string GenieXLLMBackend::runtime_version() const {
    return runtime_ ? runtime_->runtime_version : "uninitialized";
}

std::string GenieXLLMBackend::plugin_version() const {
    return runtime_ ? runtime_->plugin_version : "uninitialized";
}

}  // namespace hexagonflow
