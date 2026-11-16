#include "hexagonflow/metrics.hpp"
#include "hexagonflow/process_backend.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::uint64_t clock_us() {
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
            if (key != "--bench" && key != "--model" && key != "--log" &&
                key != "--metrics" && key != "--bench-json" && key != "--run-id" &&
                key != "--device" && key != "--device-id" && key != "--prompt-tokens" && key != "--gen-tokens" &&
                key != "--threads" && key != "--repetitions" && key != "--timeout-s" &&
                key != "--geniex-log-level") {
                throw std::invalid_argument("unknown option: " + key);
            }
            if (!args.emplace(key, argv[i + 1]).second) throw std::invalid_argument("duplicate option");
        }
        for (const auto* key : {"--bench", "--model", "--log", "--metrics", "--bench-json", "--run-id", "--device"}) {
            if (!args.contains(key) || args.at(key).empty()) {
                throw std::invalid_argument(std::string("missing ") + key);
            }
        }
        const auto device = args.at("--device");
        const auto device_id = args.contains("--device-id") ? args.at("--device-id") : "";
        if (args.contains("--device-id") && device_id.empty()) {
            throw std::invalid_argument("--device-id cannot be empty");
        }
        const auto geniex_log_level = args.contains("--geniex-log-level")
                                          ? args.at("--geniex-log-level")
                                          : "error";
        if (geniex_log_level.empty()) {
            throw std::invalid_argument("--geniex-log-level cannot be empty");
        }
        if (device != "cpu" && device != "npu" && device != "hybrid") {
            throw std::invalid_argument("device must be cpu, npu or hybrid");
        }
        const auto number = [&args](const std::string& key, unsigned long fallback,
                                    unsigned long maximum) {
            const auto value = args.contains(key) ? std::stoul(args.at(key)) : fallback;
            if (!value || value > maximum) throw std::invalid_argument("invalid " + key);
            return value;
        };
        const auto prompt = number("--prompt-tokens", 32, 4096);
        const auto tokens = number("--gen-tokens", 16, 512);
        const auto threads = number("--threads", 4, 64);
        const auto repetitions = number("--repetitions", 3, 20);
        const auto timeout = number("--timeout-s", 120, 900);
        const auto model_size = std::filesystem::file_size(args.at("--model"));
        std::vector<std::string> engine_args{
            "--plugin", "llama_cpp", "--device", device, "-m", args.at("--model"),
            "-p", std::to_string(prompt), "-n", std::to_string(tokens), "-c",
            std::to_string(std::max(256UL, prompt + tokens + 16UL)),
            "-r", std::to_string(repetitions), "--warmup", "1", "-t", std::to_string(threads),
            "--seed", "42", "--log", geniex_log_level,
            "--output-json", args.at("--bench-json"), "--cell-id", args.at("--run-id")};
        if (!device_id.empty()) {
            engine_args.insert(engine_args.end(), {"--device-id", device_id});
        }
        const std::string backend_id = device_id.empty()
                                           ? "geniex-process-auto"
                                           : "geniex-process-" + device_id;
        hexagonflow::ProcessBackend backend({
            args.at("--bench"), std::move(engine_args), args.at("--log"),
            std::chrono::seconds(timeout), backend_id, hexagonflow::BackendFamily::GenieX});
        const std::string resource = device_id.empty()
                                         ? device + "-auto-requested"
                                         : device_id + "-requested";
        hexagonflow::ResourceManager resources({{resource, backend.id(), backend.family(), 1U, model_size}});
        hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::Fifo, resources);
        hexagonflow::Workload work;
        work.id = "request-1";
        work.model_id = "geniex-llama-model";
        work.kind = hexagonflow::WorkloadKind::LlmDecode;
        work.model_memory_bytes = model_size;
        work.requested_output_tokens = tokens;
        work.allowed_backend_ids = {backend.id()};
        if (!scheduler.submit(work)) throw std::runtime_error("task submission failed");
        const auto decisions = scheduler.dispatch_ready();
        if (decisions.empty() || !decisions.front().dispatched) throw std::runtime_error("admission failed");
        const auto start = clock_us();
        const auto steady_start = std::chrono::steady_clock::now();
        const auto result = backend.execute(work);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - steady_start).count();
        const auto end = clock_us();
        if (result.success) scheduler.complete(work.id);
        else scheduler.fail(work.id, result.message);
        hexagonflow::JsonlMetricsSink metrics(args.at("--metrics"));
        metrics.append({args.at("--run-id"), work.id, work.model_id, resource, backend.id(),
                        result.message, false, start, end, 0, 0, false,
                        result.peak_process_rss_bytes});
        std::cout << "backend=geniex_llama_cpp requested_device=" << device
                  << " requested_device_id=" << (device_id.empty() ? "auto" : device_id)
                  << " actual_device=inspect_bench_json_and_logs status=" << result.message
                  << " elapsed_ms=" << elapsed << " model_residency=per_process\n";
        return result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
