#include "hexagonflow/metrics.hpp"
#include "hexagonflow/process_backend.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::uint64_t wall_us() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += character; break;
        }
    }
    return result;
}

unsigned long number(const std::map<std::string, std::string>& args,
                     const std::string& key, unsigned long fallback,
                     unsigned long maximum) {
    const auto found = args.find(key);
    const auto value = found == args.end() ? fallback : std::stoul(found->second);
    if (value == 0UL || value > maximum) {
        throw std::invalid_argument("invalid " + key);
    }
    return value;
}

std::uint64_t service_estimate(const std::map<std::string, std::string>& args,
                               const std::string& key) {
    std::size_t consumed = 0;
    const auto value = std::stoull(args.at(key), &consumed);
    if (consumed != args.at(key).size() || value == 0U) {
        throw std::invalid_argument("invalid positive microsecond estimate: " + key);
    }
    return value;
}

hexagonflow::ProcessBackend make_backend(
    const std::map<std::string, std::string>& args,
    const std::string& device_id,
    const std::string& suffix,
    unsigned long prompt_tokens,
    unsigned long output_tokens,
    unsigned long threads,
    unsigned long repetitions,
    unsigned long timeout_seconds) {
    const std::string backend_id = "geniex-process-" + device_id;
    const auto context = std::max(256UL, prompt_tokens + output_tokens + 16UL);
    std::vector<std::string> engine_args{
        "--plugin", "llama_cpp", "--device", "npu", "--device-id", device_id,
        "-m", args.at("--model"), "-p", std::to_string(prompt_tokens),
        "-n", std::to_string(output_tokens), "-c", std::to_string(context),
        "-r", std::to_string(repetitions), "--warmup", "1",
        "-t", std::to_string(threads), "--seed", "42", "--log", "error",
        "--output-json", (std::filesystem::path(args.at("--output-dir")) /
                           ("bench-" + suffix + ".json")).string(),
        "--cell-id", args.at("--run-id") + "-" + suffix};
    return hexagonflow::ProcessBackend({
        args.at("--bench"), std::move(engine_args),
        (std::filesystem::path(args.at("--output-dir")) /
         ("engine-" + suffix + ".log")).string(),
        std::chrono::seconds(timeout_seconds), backend_id,
        hexagonflow::BackendFamily::GenieX});
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) throw std::invalid_argument("expected --option value");
            const std::string key(argv[index]);
            if (key != "--bench" && key != "--model" && key != "--output-dir" &&
                key != "--run-id" && key != "--device-a" && key != "--device-b" &&
                key != "--service-us-a" && key != "--service-us-b" &&
                key != "--prompt-tokens" && key != "--gen-tokens" &&
                key != "--threads" && key != "--repetitions" && key != "--timeout-s") {
                throw std::invalid_argument("unknown option: " + key);
            }
            if (!args.emplace(key, argv[index + 1]).second) {
                throw std::invalid_argument("duplicate option");
            }
        }
        for (const auto* key : {"--bench", "--model", "--output-dir", "--run-id",
                                "--device-a", "--device-b", "--service-us-a",
                                "--service-us-b"}) {
            if (!args.contains(key) || args.at(key).empty()) {
                throw std::invalid_argument(std::string("missing ") + key);
            }
        }
        if (args.at("--device-a") == args.at("--device-b")) {
            throw std::invalid_argument("candidate device IDs must be distinct");
        }

        const auto prompt = number(args, "--prompt-tokens", 128, 4096);
        const auto tokens = number(args, "--gen-tokens", 64, 512);
        const auto threads = number(args, "--threads", 4, 64);
        const auto repetitions = number(args, "--repetitions", 5, 20);
        const auto timeout = number(args, "--timeout-s", 180, 900);
        const auto service_a = service_estimate(args, "--service-us-a");
        const auto service_b = service_estimate(args, "--service-us-b");
        const auto model_size = std::filesystem::file_size(args.at("--model"));
        std::filesystem::create_directories(args.at("--output-dir"));

        auto backend_a = make_backend(args, args.at("--device-a"), "a", prompt, tokens,
                                      threads, repetitions, timeout);
        auto backend_b = make_backend(args, args.at("--device-b"), "b", prompt, tokens,
                                      threads, repetitions, timeout);
        hexagonflow::ResourceManager resources({
            {args.at("--device-a"), backend_a.id(), backend_a.family(), 1U, model_size},
            {args.at("--device-b"), backend_b.id(), backend_b.family(), 1U, model_size},
        });
        hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::PredictedLatency,
                                         resources);
        hexagonflow::Workload task;
        task.id = "routed-llm-request";
        task.model_id = "geniex-llama-model";
        task.kind = hexagonflow::WorkloadKind::LlmDecode;
        task.model_memory_bytes = model_size;
        task.requested_output_tokens = tokens;
        task.allowed_backend_ids = {backend_a.id(), backend_b.id()};
        task.service_estimates_by_backend_us = {
            {backend_a.id(), service_a}, {backend_b.id(), service_b},
        };
        if (!scheduler.submit(task)) throw std::runtime_error("task submission failed");
        const auto decisions = scheduler.dispatch_ready();
        if (decisions.size() != 1U || !decisions.front().dispatched) {
            throw std::runtime_error(decisions.empty() ? "scheduler returned no placement"
                                                       : decisions.front().reason);
        }
        const auto& decision = decisions.front();
        hexagonflow::ProcessBackend* selected = nullptr;
        std::string selected_suffix;
        if (decision.resource_id == args.at("--device-a")) {
            selected = &backend_a;
            selected_suffix = "a";
        } else if (decision.resource_id == args.at("--device-b")) {
            selected = &backend_b;
            selected_suffix = "b";
        } else {
            throw std::runtime_error("scheduler selected an unknown resource");
        }

        const auto start_unix = wall_us();
        const auto start = std::chrono::steady_clock::now();
        const auto result = selected->execute(task);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        const auto end_unix = wall_us();
        if (result.success) (void)scheduler.complete(task.id);
        else (void)scheduler.fail(task.id, result.message);

        const auto output_dir = std::filesystem::path(args.at("--output-dir"));
        hexagonflow::JsonlMetricsSink metrics((output_dir / "metrics.jsonl").string());
        metrics.append({args.at("--run-id"), task.id, task.model_id,
                        decision.resource_id, selected->id(), result.message, false,
                        start_unix, end_unix, 0, 0, false,
                        result.peak_process_rss_bytes});
        std::ofstream output(output_dir / "result.json", std::ios::out | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create result.json");
        output << "{\n"
               << "  \"run_id\": \"" << json_escape(args.at("--run-id")) << "\",\n"
               << "  \"simulated\": false,\n"
               << "  \"policy\": \"predicted_latency\",\n"
               << "  \"selected_resource_id\": \"" << json_escape(decision.resource_id) << "\",\n"
               << "  \"selected_backend_id\": \"" << json_escape(selected->id()) << "\",\n"
               << "  \"profile_service_estimate_us\": "
               << (selected_suffix == "a" ? service_a : service_b) << ",\n"
               << "  \"wrapper_elapsed_us\": " << elapsed << ",\n"
               << "  \"request_status\": \"" << json_escape(result.message) << "\",\n"
               << "  \"runtime_assignment_evidence\": \"inspect bench JSON and engine log\"\n"
               << "}\n";
        output.flush();
        if (!output) throw std::runtime_error("failed writing result.json");
        std::cout << "policy=predicted_latency selected_resource=" << decision.resource_id
                  << " profile_service_estimate_us="
                  << (selected_suffix == "a" ? service_a : service_b)
                  << " status=" << result.message << " wrapper_elapsed_us=" << elapsed << '\n';
        return result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
