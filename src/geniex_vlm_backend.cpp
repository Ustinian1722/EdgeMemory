#include "hexagonflow/geniex_vlm_backend.hpp"

#include <geniex.h>

#include <algorithm>
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

class RuntimeApi {
public:
    using SetLogLevel = decltype(&geniex_set_log_level);
    using Init = decltype(&geniex_init);
    using Deinit = decltype(&geniex_deinit);
    using GetPluginList = decltype(&geniex_get_plugin_list);
    using Free = decltype(&geniex_free);
    using Version = decltype(&geniex_version);
    using PluginVersion = decltype(&geniex_get_plugin_version);
    using VlmCreate = decltype(&geniex_vlm_create);
    using VlmDestroy = decltype(&geniex_vlm_destroy);
    using VlmReset = decltype(&geniex_vlm_reset);
    using VlmApplyTemplate = decltype(&geniex_vlm_apply_chat_template);
    using VlmGenerate = decltype(&geniex_vlm_generate);

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
            vlm_create = symbol<VlmCreate>("geniex_vlm_create");
            vlm_destroy = symbol<VlmDestroy>("geniex_vlm_destroy");
            vlm_reset = symbol<VlmReset>("geniex_vlm_reset");
            vlm_apply_template = symbol<VlmApplyTemplate>("geniex_vlm_apply_chat_template");
            vlm_generate = symbol<VlmGenerate>("geniex_vlm_generate");
            const auto log_rc = set_log_level(GENIEX_LOG_LEVEL_ERROR);
            if (log_rc != GENIEX_SUCCESS) throw std::runtime_error("geniex-log-level-configuration-failed");
            const auto init_rc = init();
            if (init_rc != GENIEX_SUCCESS) throw std::runtime_error("geniex-init-" + std::to_string(init_rc));
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
    VlmCreate vlm_create{};
    VlmDestroy vlm_destroy{};
    VlmReset vlm_reset{};
    VlmApplyTemplate vlm_apply_template{};
    VlmGenerate vlm_generate{};

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
    const auto rc = api.get_plugin_list(&output);
    if (rc != GENIEX_SUCCESS) return false;
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

struct GenieXVlmBackend::RuntimeHandle {
    geniex_VLM* vlm{nullptr};
    std::shared_ptr<RuntimeApi> api;
    std::string runtime_version;
    std::string plugin_version;

    ~RuntimeHandle() {
        if (vlm) {
            std::lock_guard lock(model_create_mutex);
            (void)api->vlm_destroy(vlm);
            vlm = nullptr;
        }
        api.reset();
    }
};

GenieXVlmBackend::GenieXVlmBackend(GenieXVlmBackendConfig config)
    : config_(std::move(config)), runtime_(std::make_unique<RuntimeHandle>()) {
    if (config_.backend_id.empty() || config_.model_path.empty() || config_.mmproj_path.empty() ||
        config_.plugin_id.empty() || config_.context_length < 0 || config_.threads < 0 ||
        config_.default_max_tokens < 1) {
        throw std::invalid_argument("invalid-geniex-vlm-configuration");
    }
    if (!std::filesystem::is_regular_file(config_.model_path) ||
        !std::filesystem::is_regular_file(config_.mmproj_path)) {
        throw std::invalid_argument("geniex-vlm-model-artifact-missing");
    }
    if (!config_.tokenizer_path.empty() &&
        !std::filesystem::is_regular_file(config_.tokenizer_path)) {
        throw std::invalid_argument("geniex-vlm-tokenizer-artifact-missing");
    }

    runtime_->api = acquire_runtime(config_.library_path, config_.plugin_directory);
    if (!plugin_is_available(*runtime_->api, config_.plugin_id)) {
        throw std::runtime_error("geniex-plugin-unavailable");
    }

    geniex_VlmCreateInput input{};
    input.model_path = config_.model_path.c_str();
    input.mmproj_path = config_.mmproj_path.c_str();
    input.tokenizer_path = config_.tokenizer_path.empty() ? nullptr : config_.tokenizer_path.c_str();
    input.plugin_id = config_.plugin_id.c_str();
    input.device_id = config_.device_id.empty() ? nullptr : config_.device_id.c_str();
    input.vit_device_id = config_.vision_device_id.empty()
        ? nullptr : config_.vision_device_id.c_str();
    input.config.n_ctx = config_.context_length;
    input.config.n_threads = config_.threads;
    input.config.n_gpu_layers = config_.gpu_layers;
    input.config.power_mode = GENIEX_POWER_MODE_BALANCED;

    std::lock_guard lock(model_create_mutex);
    const auto rc = runtime_->api->vlm_create(&input, &runtime_->vlm);
    if (rc != GENIEX_SUCCESS || !runtime_->vlm) {
        throw std::runtime_error("geniex-vlm-create-" + std::to_string(rc));
    }
    const char* version = runtime_->api->version();
    runtime_->runtime_version = version ? version : "unknown";
    const char* plugin = runtime_->api->plugin_version(config_.plugin_id.c_str());
    runtime_->plugin_version = plugin ? plugin : "unknown";
}

GenieXVlmBackend::~GenieXVlmBackend() = default;

std::string GenieXVlmBackend::id() const { return config_.backend_id; }

BackendFamily GenieXVlmBackend::family() const { return BackendFamily::GenieX; }

BackendResult GenieXVlmBackend::execute(const Workload& workload) {
    std::lock_guard execute_lock(execute_mutex_);
    BackendResult result;
    result.simulated = false;
    if (!runtime_ || !runtime_->vlm) {
        result.message = "geniex-vlm-handle-unavailable";
        return result;
    }
    if (!config_.model_id.empty() && workload.model_id != config_.model_id) {
        result.message = "worker-model-mismatch";
        return result;
    }
    if (workload.prompt_utf8.empty() || workload.media_paths.empty()) {
        result.message = "vlm-prompt-or-media-missing";
        return result;
    }
    for (const auto& path : workload.media_paths) {
        if (!std::filesystem::is_regular_file(path)) {
            result.message = "vlm-media-artifact-missing";
            return result;
        }
    }

    auto& api = *runtime_->api;
    const auto reset_rc = api.vlm_reset(runtime_->vlm);
    if (reset_rc != GENIEX_SUCCESS) {
        result.message = "geniex-vlm-reset-" + std::to_string(reset_rc);
        return result;
    }

    const bool audio_input = workload.kind == WorkloadKind::Asr;
    std::vector<geniex_VlmContent> contents;
    contents.reserve(workload.media_paths.size() + 1U);
    contents.push_back({"text", workload.prompt_utf8.c_str()});
    std::vector<geniex_Path> media_paths;
    media_paths.reserve(workload.media_paths.size());
    for (const auto& path : workload.media_paths) {
        contents.push_back({audio_input ? "audio" : "image", path.c_str()});
        media_paths.push_back(path.c_str());
    }
    geniex_VlmChatMessage message{};
    message.role = "user";
    message.contents = contents.data();
    message.content_count = static_cast<std::int64_t>(contents.size());
    geniex_VlmApplyChatTemplateInput template_input{};
    template_input.messages = &message;
    template_input.message_count = 1;
    template_input.enable_thinking = false;
    geniex_VlmApplyChatTemplateOutput template_output{};
    const auto template_rc = api.vlm_apply_template(
        runtime_->vlm, &template_input, &template_output);
    if (template_rc != GENIEX_SUCCESS || !template_output.formatted_text) {
        if (template_output.formatted_text) api.free(template_output.formatted_text);
        result.message = "geniex-vlm-template-" + std::to_string(template_rc);
        return result;
    }

    geniex_SamplerConfig sampler{};
    sampler.temperature = 0.0F;
    sampler.top_p = 1.0F;
    sampler.repetition_penalty = 1.0F;
    sampler.seed = 42;
    geniex_GenerationConfig generation{};
    const auto token_limit = workload.requested_output_tokens > 0
        ? workload.requested_output_tokens : static_cast<std::uint64_t>(config_.default_max_tokens);
    generation.max_tokens = static_cast<std::int32_t>(std::min<std::uint64_t>(
        token_limit, static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())));
    generation.sampler_config = &sampler;
    generation.image_paths = audio_input ? nullptr : media_paths.data();
    generation.image_count = audio_input ? 0 : static_cast<std::int32_t>(media_paths.size());
    generation.audio_paths = audio_input ? media_paths.data() : nullptr;
    generation.audio_count = audio_input ? static_cast<std::int32_t>(media_paths.size()) : 0;

    geniex_VlmGenerateInput generate_input{};
    generate_input.prompt_utf8 = template_output.formatted_text;
    generate_input.config = &generation;
    geniex_VlmGenerateOutput generate_output{};
    const auto generate_rc = api.vlm_generate(
        runtime_->vlm, &generate_input, &generate_output);
    api.free(template_output.formatted_text);
    if (generate_rc != GENIEX_SUCCESS) {
        if (generate_output.full_text) api.free(generate_output.full_text);
        result.message = "geniex-vlm-generate-" + std::to_string(generate_rc);
        return result;
    }

    result.success = true;
    result.message = "completed";
    result.output_text = generate_output.full_text ? generate_output.full_text : "";
    result.output_tokens = positive_u64(generate_output.profile_data.generated_tokens);
    result.ttft_us = positive_u64(generate_output.profile_data.ttft);
    result.media_us = positive_u64(generate_output.profile_data.media_time);
    result.prefill_us = positive_u64(generate_output.profile_data.prompt_time);
    result.decode_us = positive_u64(generate_output.profile_data.decode_time);
    result.prefill_tokens_per_second = std::max(0.0, generate_output.profile_data.prefill_speed);
    result.decode_tokens_per_second = std::max(0.0, generate_output.profile_data.decoding_speed);
    if (generate_output.full_text) api.free(generate_output.full_text);
    return result;
}

std::string GenieXVlmBackend::runtime_version() const {
    return runtime_ ? runtime_->runtime_version : "uninitialized";
}

std::string GenieXVlmBackend::plugin_version() const {
    return runtime_ ? runtime_->plugin_version : "uninitialized";
}

}  // namespace hexagonflow
