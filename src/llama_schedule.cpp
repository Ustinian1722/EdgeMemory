#include "hexagonflow/metrics.hpp"
#include "hexagonflow/process_backend.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::uint64_t now_us() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("expected --option value");
            const std::string key(argv[i]);
            if (key != "--engine" && key != "--model" && key != "--log" &&
                key != "--metrics" && key != "--run-id" && key != "--prompt" &&
                key != "--tokens" && key != "--threads" && key != "--timeout-s") {
                throw std::invalid_argument("unknown option: " + key);
            }
            if (!args.emplace(key, argv[i + 1]).second) {
                throw std::invalid_argument("duplicate option: " + key);
            }
        }
        for (const auto* key : {"--engine", "--model", "--log", "--metrics", "--run-id", "--prompt"}) {
            if (!args.contains(key) || args.at(key).empty()) {
                throw std::invalid_argument(std::string("missing ") + key);
            }
        }
        const auto parse = [&args](const std::string& key, unsigned long fallback,
                                   unsigned long maximum) {
            const auto value = args.contains(key) ? std::stoul(args.at(key)) : fallback;
            if (value == 0 || value > maximum) throw std::invalid_argument("invalid " + key);
            return value;
        };
        const auto tokens = parse("--tokens", 16, 256);
        const auto threads = parse("--threads", 4, 64);
        const auto timeout = parse("--timeout-s", 120, 900);
        const auto model_size = std::filesystem::file_size(args.at("--model"));

        hexagonflow::ProcessBackend backend({
            args.at("--engine"),
            {"-m", args.at("--model"), "-p", args.at("--prompt"),
             "-n", std::to_string(tokens), "-t", std::to_string(threads),
             "-ngl", "0", "--no-display-prompt", "--no-conversation", "--no-warmup", "--temp", "0"},
            args.at("--log"), std::chrono::seconds(timeout)});
        hexagonflow::ResourceManager resources({
            {"cpu", backend.id(), backend.family(), 1U, model_size}});
        hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::Fifo, resources);
        hexagonflow::Workload work;
        work.id = "request-1";
        work.model_id = "llama-cpp-model";
        work.kind = hexagonflow::WorkloadKind::LlmDecode;
        work.model_memory_bytes = model_size;
        work.requested_output_tokens = tokens;
        work.allowed_backend_ids = {backend.id()};
        if (!scheduler.submit(work)) throw std::runtime_error("task submission failed");
        const auto decisions = scheduler.dispatch_ready();
        if (decisions.empty() || !decisions.front().dispatched) {
            throw std::runtime_error("resource admission failed");
        }
        const auto start = now_us();
        const auto steady_start = std::chrono::steady_clock::now();
        const auto result = backend.execute(work);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - steady_start).count();
        const auto end = now_us();
        if (result.success) scheduler.complete(work.id);
        else scheduler.fail(work.id, result.message);
        hexagonflow::JsonlMetricsSink metrics(args.at("--metrics"));
        metrics.append({args.at("--run-id"), work.id, work.model_id, "cpu", backend.id(),
                        result.message, false, start, end, 0, 0, false,
                        result.peak_process_rss_bytes});
        std::cout << "device=cpu simulated=false status=" << result.message
                  << " elapsed_ms=" << elapsed_ms
                  << " token_counts=unavailable model_residency=per_process\n";
        return result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
