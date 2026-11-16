#include "hexagonflow/geniex_vlm_backend.hpp"
#include "hexagonflow/persistent_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string option(int argc, char** argv, const std::string& name,
                   const std::string& fallback = {}) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

std::uint64_t integer_option(int argc, char** argv, const std::string& name,
                             std::uint64_t fallback) {
    const auto value = option(argc, argv, name);
    return value.empty() ? fallback : std::stoull(value);
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    if (values.empty()) return 0;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2U != 0U) return *middle;
    const auto lower = *std::max_element(values.begin(), middle);
    return lower + (*middle - lower) / 2U;
}

void print_result(const std::string& scenario, std::uint64_t round,
                  const std::string& device_id,
                  const hexagonflow::BackendResult& result,
                  std::uint64_t pair_wall_us = 0) {
    std::cout << "{\"event\":\"sample\",\"scenario\":\"" << scenario
              << "\",\"round\":" << round << ",\"device_id\":\""
              << device_id << "\",\"success\":"
              << (result.success ? "true" : "false")
              << ",\"simulated\":" << (result.simulated ? "true" : "false")
              << ",\"output_tokens\":" << result.output_tokens
              << ",\"queue_wait_us\":" << result.queue_wait_us
              << ",\"worker_elapsed_us\":" << result.worker_elapsed_us
              << ",\"ttft_us\":" << result.ttft_us
              << ",\"media_us\":" << result.media_us
              << ",\"prefill_us\":" << result.prefill_us
              << ",\"decode_us\":" << result.decode_us
              << ",\"prefill_tokens_per_second\":"
              << result.prefill_tokens_per_second
              << ",\"decode_tokens_per_second\":"
              << result.decode_tokens_per_second
              << ",\"pair_wall_us\":" << pair_wall_us
              << ",\"message\":\"" << result.message << "\"}\n"
              << std::flush;
}

hexagonflow::Workload workload(const std::string& task_id,
                               const std::string& model_id,
                               const std::string& prompt,
                               const std::string& image,
                               std::uint64_t tokens,
                               std::uint64_t model_bytes) {
    hexagonflow::Workload task;
    task.id = task_id;
    task.model_id = model_id;
    task.kind = hexagonflow::WorkloadKind::Custom;
    task.prompt_utf8 = prompt;
    task.media_paths = {image};
    task.requested_output_tokens = tokens;
    task.model_memory_bytes = model_bytes;
    return task;
}

struct DeviceSamples {
    std::string id;
    std::vector<std::uint64_t> solo_worker_us;
    std::vector<std::uint64_t> concurrent_worker_us;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string model = option(argc, argv, "--model");
        const std::string mmproj = option(argc, argv, "--mmproj");
        const std::string image = option(argc, argv, "--image");
        const std::string device_a = option(argc, argv, "--device-a");
        const std::string device_b = option(argc, argv, "--device-b");
        const std::string model_id = option(argc, argv, "--model-id");
        const std::string prompt = option(argc, argv, "--prompt",
            "Describe the colors and shapes visible. Answer in one concise sentence.");
        const auto context = integer_option(argc, argv, "--context", 2048);
        const auto threads = integer_option(argc, argv, "--threads", 4);
        const auto tokens = integer_option(argc, argv, "--max-tokens", 64);
        const auto rounds = integer_option(argc, argv, "--rounds", 3);
        if (model.empty() || mmproj.empty() || image.empty() || model_id.empty() ||
            device_a.empty() || device_b.empty() || device_a == device_b ||
            rounds == 0 || rounds > 5 ||
            context == 0 || context > 131072 || threads == 0 || threads > 256 ||
            tokens == 0 || tokens > 4096) {
            throw std::invalid_argument("invalid dual-worker smoke configuration");
        }

        const auto model_bytes = static_cast<std::uint64_t>(
            std::filesystem::file_size(model) + std::filesystem::file_size(mmproj));
        auto make_worker = [&](const std::string& device_id) {
            hexagonflow::GenieXVlmBackendConfig config;
            config.backend_id = "geniex-vlm-" + device_id;
            config.model_id = model_id;
            config.model_path = model;
            config.mmproj_path = mmproj;
            config.device_id = device_id;
            config.context_length = static_cast<std::int32_t>(context);
            config.threads = static_cast<std::int32_t>(threads);
            config.default_max_tokens = static_cast<std::int32_t>(tokens);
            hexagonflow::PersistentWorkerOptions worker_options{
                "worker-" + device_id, model_id, model_bytes, 4};
            return std::make_unique<hexagonflow::PersistentWorker>(
                std::move(worker_options),
                [config = std::move(config)]() mutable {
                    return std::make_unique<hexagonflow::GenieXVlmBackend>(config);
                });
        };

        auto worker_a = make_worker(device_a);
        auto worker_b = make_worker(device_b);
        constexpr auto ready_timeout = std::chrono::minutes(4);
        if (!worker_a->wait_until_ready(ready_timeout) ||
            !worker_b->wait_until_ready(ready_timeout)) {
            throw std::runtime_error("one-or-more-geniex-workers-failed-to-load");
        }
        std::cout << "{\"event\":\"workers_ready\",\"model_id\":\"" << model_id
                  << "\",\"device_a\":\"" << device_a << "\",\"device_b\":\""
                  << device_b << "\",\"estimated_resident_bytes_per_worker\":"
                  << model_bytes << ",\"estimate_is_not_measured_rss\":true}\n"
                  << std::flush;

        auto submit = [&](hexagonflow::PersistentWorker& worker,
                          const std::string& id) {
            return worker.submit(workload(id, model_id, prompt, image, tokens, model_bytes));
        };
        const auto warm_result_a = submit(*worker_a, "warmup-" + device_a).get();
        if (!warm_result_a.success || warm_result_a.simulated) {
            throw std::runtime_error("worker-a-warmup-failed");
        }
        std::cout << "{\"event\":\"warmup\",\"device_id\":\"" << device_a
                  << "\",\"success\":true}\n" << std::flush;
        const auto warm_result_b = submit(*worker_b, "warmup-" + device_b).get();
        if (!warm_result_b.success || warm_result_b.simulated) {
            throw std::runtime_error("worker-b-warmup-failed");
        }
        std::cout << "{\"event\":\"warmup\",\"device_id\":\"" << device_b
                  << "\",\"success\":true}\n" << std::flush;

        DeviceSamples samples_a{device_a, {}, {}};
        DeviceSamples samples_b{device_b, {}, {}};
        for (std::uint64_t round = 1; round <= rounds; ++round) {
            const auto solo_a = submit(*worker_a, "solo-" + device_a + "-" + std::to_string(round)).get();
            if (!solo_a.success || solo_a.simulated) throw std::runtime_error("solo-worker-a-failed");
            samples_a.solo_worker_us.push_back(solo_a.worker_elapsed_us);
            print_result("solo", round, device_a, solo_a);

            const auto solo_b = submit(*worker_b, "solo-" + device_b + "-" + std::to_string(round)).get();
            if (!solo_b.success || solo_b.simulated) throw std::runtime_error("solo-worker-b-failed");
            samples_b.solo_worker_us.push_back(solo_b.worker_elapsed_us);
            print_result("solo", round, device_b, solo_b);

            const auto pair_start = std::chrono::steady_clock::now();
            auto future_a = submit(*worker_a, "concurrent-" + device_a + "-" + std::to_string(round));
            auto future_b = submit(*worker_b, "concurrent-" + device_b + "-" + std::to_string(round));
            const auto concurrent_a = future_a.get();
            const auto concurrent_b = future_b.get();
            const auto pair_end = std::chrono::steady_clock::now();
            const auto pair_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                pair_end - pair_start).count();
            if (!concurrent_a.success || concurrent_a.simulated ||
                !concurrent_b.success || concurrent_b.simulated) {
                throw std::runtime_error("concurrent-worker-pair-failed");
            }
            samples_a.concurrent_worker_us.push_back(concurrent_a.worker_elapsed_us);
            samples_b.concurrent_worker_us.push_back(concurrent_b.worker_elapsed_us);
            const auto pair_us = pair_elapsed > 0
                ? static_cast<std::uint64_t>(pair_elapsed) : 0U;
            print_result("concurrent", round, device_a, concurrent_a, pair_us);
            print_result("concurrent", round, device_b, concurrent_b, pair_us);
        }

        const auto solo_a_median = median(samples_a.solo_worker_us);
        const auto solo_b_median = median(samples_b.solo_worker_us);
        const auto concurrent_a_median = median(samples_a.concurrent_worker_us);
        const auto concurrent_b_median = median(samples_b.concurrent_worker_us);
        const auto slowdown = [](std::uint64_t solo, std::uint64_t concurrent) {
            if (solo == 0) return 0.0;
            return (static_cast<double>(concurrent) / static_cast<double>(solo)) - 1.0;
        };
        std::cout << "{\"event\":\"summary\",\"rounds\":" << rounds
                  << ",\"device_a\":\"" << device_a << "\",\"device_b\":\"" << device_b
                  << "\",\"median_solo_worker_us_a\":" << solo_a_median
                  << ",\"median_solo_worker_us_b\":" << solo_b_median
                  << ",\"median_concurrent_worker_us_a\":" << concurrent_a_median
                  << ",\"median_concurrent_worker_us_b\":" << concurrent_b_median
                  << ",\"slowdown_fraction_a\":" << slowdown(solo_a_median, concurrent_a_median)
                  << ",\"slowdown_fraction_b\":" << slowdown(solo_b_median, concurrent_b_median)
                  << ",\"exploratory_without_confidence_interval\":true}\n"
                  << std::flush;

        worker_a->stop(hexagonflow::WorkerStopMode::Drain);
        worker_b->stop(hexagonflow::WorkerStopMode::Drain);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "geniex-vlm-dual-worker-smoke: " << error.what() << '\n';
        return 2;
    }
}
