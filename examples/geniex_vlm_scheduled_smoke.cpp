#include "hexagonflow/geniex_vlm_backend.hpp"
#include "hexagonflow/persistent_worker.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::string option(int argc, char** argv, const std::string& name,
                   const std::string& fallback = {}) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

std::uint64_t number_option(int argc, char** argv, const std::string& name,
                            std::uint64_t fallback) {
    const auto text = option(argc, argv, name);
    if (text.empty()) return fallback;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("invalid numeric option " + name);
    }
    return value;
}

std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 8U);
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character >= 0x20U) output.push_back(static_cast<char>(character));
        }
    }
    return output;
}

void require_value(const std::string& name, const std::string& value) {
    if (value.empty()) throw std::invalid_argument("missing " + name);
}

struct RuntimeMetadata {
    std::mutex mutex;
    std::string runtime_version;
    std::string plugin_version;
};

struct Target {
    std::string device_id;
    std::string backend_id;
    std::uint64_t service_us{0};
    std::shared_ptr<RuntimeMetadata> metadata;
};

hexagonflow::Workload make_workload(const std::string& id,
                                    const std::string& model_id,
                                    const std::string& prompt,
                                    const std::string& image,
                                    std::uint64_t tokens,
                                    std::uint64_t estimated_model_bytes,
                                    const std::map<std::string, std::uint64_t>& estimates) {
    hexagonflow::Workload workload;
    workload.id = id;
    workload.model_id = model_id;
    workload.kind = hexagonflow::WorkloadKind::Custom;
    workload.prompt_utf8 = prompt;
    workload.media_paths = {image};
    workload.requested_output_tokens = tokens;
    workload.model_memory_bytes = estimated_model_bytes;
    workload.service_estimates_by_backend_us = estimates;
    return workload;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto model = option(argc, argv, "--model");
        const auto mmproj = option(argc, argv, "--mmproj");
        const auto image = option(argc, argv, "--image");
        const auto model_id = option(argc, argv, "--model-id");
        const auto device_a = option(argc, argv, "--device-a");
        const auto device_b = option(argc, argv, "--device-b");
        const auto library = option(argc, argv, "--library");
        const auto plugin_directory = option(argc, argv, "--plugin-directory");
        const auto service_a = number_option(argc, argv, "--service-us-a", 0U);
        const auto service_b = number_option(argc, argv, "--service-us-b", 0U);
        const auto context = number_option(argc, argv, "--context", 2048U);
        const auto threads = number_option(argc, argv, "--threads", 4U);
        const auto max_tokens = number_option(argc, argv, "--max-tokens", 24U);
        const auto requests = number_option(argc, argv, "--requests", 2U);
        const auto prompt = option(argc, argv, "--prompt",
            "Describe only visible colors and shapes in one concise sentence.");

        for (const auto& item : std::vector<std::pair<std::string, std::string>>{
                 {"--model", model}, {"--mmproj", mmproj}, {"--image", image},
                 {"--model-id", model_id}, {"--device-a", device_a},
                 {"--device-b", device_b}}) {
            require_value(item.first, item.second);
        }
        if (device_a == device_b || service_a == 0U || service_b == 0U ||
            context == 0U || context > 131072U || threads == 0U || threads > 256U ||
            max_tokens == 0U || max_tokens > 4096U || requests == 0U || requests > 32U) {
            throw std::invalid_argument("invalid devices, service estimates, or workload bounds");
        }
        constexpr auto int32_max = static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max());
        if (context > int32_max || threads > int32_max || max_tokens > int32_max) {
            throw std::invalid_argument("numeric option exceeds the GenieX C API range");
        }

        const auto model_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(model)) +
                                 static_cast<std::uint64_t>(std::filesystem::file_size(mmproj));
        if (model_bytes == 0U) throw std::invalid_argument("model bundle is empty");

        Target targets[2] = {
            {device_a, "geniex-vlm-" + device_a, service_a, std::make_shared<RuntimeMetadata>()},
            {device_b, "geniex-vlm-" + device_b, service_b, std::make_shared<RuntimeMetadata>()},
        };
        std::map<std::string, std::unique_ptr<hexagonflow::PersistentWorker>> workers;

        // Initialize one runtime context at a time to make cold-start order explicit.
        for (auto& target : targets) {
            auto config = hexagonflow::GenieXVlmBackendConfig{};
            config.backend_id = target.backend_id;
            config.model_id = model_id;
            config.model_path = model;
            config.mmproj_path = mmproj;
            config.device_id = target.device_id;
            config.library_path = library;
            config.plugin_directory = plugin_directory;
            config.context_length = static_cast<std::int32_t>(context);
            config.threads = static_cast<std::int32_t>(threads);
            config.default_max_tokens = static_cast<std::int32_t>(max_tokens);
            const hexagonflow::PersistentWorkerOptions options{
                target.backend_id, model_id, model_bytes, 4U};
            const auto metadata = target.metadata;
            auto worker = std::make_unique<hexagonflow::PersistentWorker>(
                options, [config = std::move(config), metadata]() mutable {
                    auto backend = std::make_unique<hexagonflow::GenieXVlmBackend>(config);
                    std::lock_guard lock(metadata->mutex);
                    metadata->runtime_version = backend->runtime_version();
                    metadata->plugin_version = backend->plugin_version();
                    return backend;
                });
            if (!worker->wait_until_ready(std::chrono::minutes(4))) {
                throw std::runtime_error("backend initialization failed for " + target.backend_id);
            }
            workers.emplace(target.device_id, std::move(worker));
            std::lock_guard lock(target.metadata->mutex);
            std::cout << "{\"event\":\"worker_ready\",\"device_id\":\""
                      << json_escape(target.device_id) << "\",\"backend_id\":\""
                      << json_escape(target.backend_id) << "\",\"runtime_version\":\""
                      << json_escape(target.metadata->runtime_version)
                      << "\",\"plugin_version\":\""
                      << json_escape(target.metadata->plugin_version)
                      << "\",\"estimated_resident_bytes\":" << model_bytes
                      << ",\"memory_value_is_model_plus_mmproj_file_size\":true}\n"
                      << std::flush;
        }

        const std::map<std::string, std::uint64_t> estimates{
            {targets[0].backend_id, targets[0].service_us},
            {targets[1].backend_id, targets[1].service_us},
        };
        for (auto& target : targets) {
            auto warmup = make_workload("warmup-" + target.device_id, model_id, prompt,
                                        image, max_tokens, model_bytes, estimates);
            const auto result = workers.at(target.device_id)->submit(std::move(warmup)).get();
            if (!result.success || result.simulated) {
                throw std::runtime_error("backend warm-up failed for " + target.backend_id);
            }
            std::cout << "{\"event\":\"warmup\",\"backend_id\":\""
                      << json_escape(target.backend_id) << "\",\"success\":true}\n"
                      << std::flush;
        }

        std::vector<hexagonflow::ResourceSpec> resource_specs;
        resource_specs.reserve(2U);
        for (const auto& target : targets) {
            resource_specs.push_back({target.device_id, target.backend_id,
                hexagonflow::BackendFamily::GenieX, 1U, model_bytes});
        }
        hexagonflow::ResourceManager resources(std::move(resource_specs));
        hexagonflow::Scheduler scheduler(
            hexagonflow::SchedulingPolicy::PredictedLatency, resources);
        for (std::uint64_t index = 0; index < requests; ++index) {
            auto workload = make_workload("scheduled-" + std::to_string(index + 1U), model_id,
                                          prompt, image, max_tokens, model_bytes, estimates);
            std::string error;
            if (!scheduler.submit(std::move(workload), &error)) {
                throw std::runtime_error("scheduler rejected workload: " + error);
            }
        }

        struct InFlight {
            std::string task_id;
            std::string resource_id;
            std::uint64_t predicted_service_us{0};
            std::future<hexagonflow::BackendResult> result;
        };
        const auto backend_for_resource = [&targets](const std::string& resource_id) {
            for (const auto& target : targets) {
                if (target.device_id == resource_id) return target.backend_id;
            }
            throw std::runtime_error("scheduler selected an unknown resource");
        };
        std::uint64_t finished = 0U;
        bool any_failure = false;
        while (finished < requests) {
            const auto decisions = scheduler.dispatch_ready();
            std::vector<InFlight> in_flight;
            for (const auto& decision : decisions) {
                if (!decision.dispatched) continue;
                const auto task = scheduler.task(decision.task_id);
                const auto estimate = task.workload.service_estimates_by_backend_us.at(
                    backend_for_resource(decision.resource_id));
                auto future = workers.at(decision.resource_id)->submit(task.workload);
                in_flight.push_back({decision.task_id, decision.resource_id, estimate,
                                     std::move(future)});
                std::cout << "{\"event\":\"scheduled\",\"task_id\":\""
                          << json_escape(decision.task_id) << "\",\"resource_id\":\""
                          << json_escape(decision.resource_id) << "\",\"backend_id\":\""
                          << json_escape(backend_for_resource(decision.resource_id))
                          << "\",\"predicted_service_us\":" << estimate << "}\n"
                          << std::flush;
            }
            if (in_flight.empty()) {
                throw std::runtime_error("scheduler made no progress with queued workloads");
            }
            for (auto& item : in_flight) {
                const auto result = item.result.get();
                const bool success = result.success && !result.simulated;
                if (success) {
                    (void)scheduler.complete(item.task_id);
                } else {
                    (void)scheduler.fail(item.task_id, result.message);
                    any_failure = true;
                }
                ++finished;
                std::cout << "{\"event\":\"result\",\"task_id\":\""
                          << json_escape(item.task_id) << "\",\"resource_id\":\""
                          << json_escape(item.resource_id) << "\",\"success\":"
                          << (success ? "true" : "false")
                          << ",\"simulated\":" << (result.simulated ? "true" : "false")
                          << ",\"predicted_service_us\":" << item.predicted_service_us
                          << ",\"worker_elapsed_us\":" << result.worker_elapsed_us
                          << ",\"ttft_us\":" << result.ttft_us
                          << ",\"decode_tokens_per_second\":"
                          << result.decode_tokens_per_second
                          << ",\"output_tokens\":" << result.output_tokens
                          << ",\"text\":\"" << json_escape(result.output_text) << "\"}\n"
                          << std::flush;
            }
        }

        for (auto& [device_id, worker] : workers) {
            (void)device_id;
            worker->stop(hexagonflow::WorkerStopMode::Drain);
        }
        return any_failure ? 3 : 0;
    } catch (const std::exception& error) {
        std::cerr << "geniex-vlm-scheduled-smoke: " << error.what() << '\n';
        return 2;
    }
}
