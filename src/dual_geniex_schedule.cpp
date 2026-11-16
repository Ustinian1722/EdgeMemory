#include "hexagonflow/metrics.hpp"
#include "hexagonflow/process_backend.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t unix_us() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string json_escape(const std::string& input) {
    std::string escaped;
    for (const char raw : input) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character < 0x20U ? '?' : static_cast<char>(character);
        }
    }
    return escaped;
}

bool safe_identifier(const std::string& value) {
    if (value.empty()) return false;
    return std::all_of(value.begin(), value.end(), [](char raw) {
        const auto c = static_cast<unsigned char>(raw);
        const bool ascii_letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        const bool digit = c >= '0' && c <= '9';
        return ascii_letter || digit || c == '_' || c == '-' || c == '.';
    });
}

unsigned long number(const std::map<std::string, std::string>& args,
                     const std::string& key,
                     unsigned long fallback,
                     unsigned long minimum,
                     unsigned long maximum) {
    const auto found = args.find(key);
    const auto value = found == args.end() ? fallback : std::stoul(found->second);
    if (value < minimum || value > maximum) throw std::invalid_argument("out-of-range " + key);
    return value;
}

struct Observation {
    hexagonflow::BackendResult result;
    std::uint64_t start_unix_us{0};
    std::uint64_t end_unix_us{0};
    std::int64_t start_steady_us{0};
    std::int64_t end_steady_us{0};
};

Observation execute(hexagonflow::ProcessBackend& backend,
                    const hexagonflow::Workload& workload) {
    Observation observation;
    const auto start = Clock::now();
    observation.start_unix_us = unix_us();
    observation.start_steady_us = std::chrono::duration_cast<std::chrono::microseconds>(
        start.time_since_epoch()).count();
    observation.result = backend.execute(workload);
    const auto end = Clock::now();
    observation.end_unix_us = unix_us();
    observation.end_steady_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end.time_since_epoch()).count();
    return observation;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
    (void)umask(0077);
#endif
    try {
        std::map<std::string, std::string> args;
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) throw std::invalid_argument("expected --option value");
            const std::string key(argv[index]);
            static const std::vector<std::string> allowed{
                "--geniex-bench", "--model", "--model-id", "--run-id", "--output-dir", "--device-a",
                "--device-b", "--prompt-tokens", "--gen-tokens", "--repetitions",
                "--threads", "--timeout-s", "--condition", "--geniex-log-level",
                "--interference-domain-id"};
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
                throw std::invalid_argument("unknown option: " + key);
            }
            if (!args.emplace(key, argv[index + 1]).second) {
                throw std::invalid_argument("duplicate option: " + key);
            }
        }
        for (const auto* key : {"--geniex-bench", "--model", "--run-id", "--output-dir"}) {
            if (!args.contains(key) || args.at(key).empty()) {
                throw std::invalid_argument(std::string("missing ") + key);
            }
        }
        for (const auto* key : {"--device-a", "--device-b"}) {
            if (!args.contains(key) || args.at(key).empty()) {
                throw std::invalid_argument(std::string("missing ") + key);
            }
        }
        const auto device_a = args.at("--device-a");
        const auto device_b = args.at("--device-b");
        const auto interference_domain = args.contains("--interference-domain-id")
                                             ? args.at("--interference-domain-id")
                                             : std::string{};
        if (args.contains("--interference-domain-id") &&
            !safe_identifier(interference_domain)) {
            throw std::invalid_argument("invalid --interference-domain-id");
        }
        if (device_a.empty() || device_b.empty() || device_a == device_b ||
            device_a.find(',') != std::string::npos || device_b.find(',') != std::string::npos) {
            throw std::invalid_argument("device-a and device-b must be distinct single device IDs");
        }
        const auto condition = args.contains("--condition") ? args.at("--condition") : "concurrent";
        if (condition != "concurrent" && condition != "isolated") {
            throw std::invalid_argument("--condition must be concurrent or isolated");
        }
        const auto prompt_tokens = number(args, "--prompt-tokens", 128, 1, 4096);
        const auto gen_tokens = number(args, "--gen-tokens", 64, 1, 256);
        const auto repetitions = number(args, "--repetitions", 5, 1, 20);
        const auto threads = number(args, "--threads", 4, 1, 64);
        const auto timeout_s = number(args, "--timeout-s", 300, 1, 900);
        const auto log_level = args.contains("--geniex-log-level")
                                   ? args.at("--geniex-log-level") : "error";
        if (log_level != "none" && log_level != "error" && log_level != "warn" &&
            log_level != "info" && log_level != "debug" && log_level != "trace") {
            throw std::invalid_argument("invalid --geniex-log-level");
        }

        const std::filesystem::path output_dir(args.at("--output-dir"));
        std::filesystem::create_directories(output_dir);
        const auto model_bytes = std::filesystem::file_size(args.at("--model"));
        const auto make_backend = [&](const std::string& device, const std::string& suffix) {
            const std::string backend_id = "geniex-device-" + suffix;
            return hexagonflow::ProcessBackend({
                args.at("--geniex-bench"),
                {"--plugin", "llama_cpp", "--device", "npu", "--device-id", device,
                 "-m", args.at("--model"), "-p", std::to_string(prompt_tokens),
                 "-n", std::to_string(gen_tokens), "-c",
                 std::to_string(std::max(256UL, prompt_tokens + gen_tokens + 16UL)),
                 "-r", std::to_string(repetitions), "--warmup", "1", "-t", std::to_string(threads),
                 "--seed", "42", "--log", log_level, "--output-json",
                 (output_dir / ("llm-" + suffix + ".json")).string(),
                 "--cell-id", args.at("--run-id") + "-" + suffix},
                (output_dir / ("llm-" + suffix + ".log")).string(),
                std::chrono::seconds(timeout_s), backend_id, hexagonflow::BackendFamily::GenieX});
        };
        auto backend_a = make_backend(device_a, "a");
        auto backend_b = make_backend(device_b, "b");
        hexagonflow::ResourceManager resources({
            {"resource-a", backend_a.id(), backend_a.family(), 1U, model_bytes,
             interference_domain},
            {"resource-b", backend_b.id(), backend_b.family(), 1U, model_bytes,
             interference_domain}});
        hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::Fifo, resources);
        hexagonflow::Workload task_a;
        task_a.id = "llm-request-a";
        task_a.model_id = args.contains("--model-id") ? args.at("--model-id")
                                                       : "smollm2-q4_0";
        if (task_a.model_id.empty()) throw std::invalid_argument("--model-id cannot be empty");
        task_a.kind = hexagonflow::WorkloadKind::LlmDecode;
        task_a.input_units = prompt_tokens;
        task_a.model_memory_bytes = model_bytes;
        task_a.requested_output_tokens = gen_tokens;
        task_a.allowed_backend_ids = {backend_a.id()};
        auto task_b = task_a;
        task_b.id = "llm-request-b";
        task_b.allowed_backend_ids = {backend_b.id()};
        if (!scheduler.submit(task_a) || !scheduler.submit(task_b)) {
            throw std::runtime_error("request submission failed");
        }
        const auto decisions = scheduler.dispatch_ready();
        if (decisions.size() != 2 ||
            std::any_of(decisions.begin(), decisions.end(), [](const auto& decision) {
                return !decision.dispatched;
            })) {
            throw std::runtime_error("scheduler could not reserve both independent resources");
        }

        Observation result_a;
        Observation result_b;
        if (condition == "concurrent") {
            auto future_a = std::async(std::launch::async, [&] { return execute(backend_a, task_a); });
            auto future_b = std::async(std::launch::async, [&] { return execute(backend_b, task_b); });
            result_a = future_a.get();
            result_b = future_b.get();
        } else {
            result_a = execute(backend_a, task_a);
            result_b = execute(backend_b, task_b);
        }
        if (result_a.result.success) (void)scheduler.complete(task_a.id);
        else (void)scheduler.fail(task_a.id, result_a.result.message);
        if (result_b.result.success) (void)scheduler.complete(task_b.id);
        else (void)scheduler.fail(task_b.id, result_b.result.message);

        const auto overlap_start = std::max(result_a.start_steady_us, result_b.start_steady_us);
        const auto overlap_end = std::min(result_a.end_steady_us, result_b.end_steady_us);
        const auto overlap_us = std::max<std::int64_t>(0, overlap_end - overlap_start);
        const auto metrics_path = (output_dir / "metrics.jsonl").string();
        hexagonflow::JsonlMetricsSink metrics(metrics_path);
        metrics.append({args.at("--run-id"), task_a.id, task_a.model_id, device_a,
                        backend_a.id(), result_a.result.message, false,
                        result_a.start_unix_us, result_a.end_unix_us, 0, 0, false,
                        result_a.result.peak_process_rss_bytes});
        metrics.append({args.at("--run-id"), task_b.id, task_b.model_id, device_b,
                        backend_b.id(), result_b.result.message, false,
                        result_b.start_unix_us, result_b.end_unix_us, 0, 0, false,
                        result_b.result.peak_process_rss_bytes});

        std::ofstream result_file(output_dir / "result.json", std::ios::out | std::ios::trunc);
        if (!result_file) throw std::runtime_error("cannot create result.json");
        result_file << "{\n"
                    << "  \"run_id\": \"" << json_escape(args.at("--run-id")) << "\",\n"
                    << "  \"condition\": \"" << condition << "\",\n"
                    << "  \"simulated\": false,\n"
                    << "  \"model_id\": \"" << json_escape(task_a.model_id) << "\",\n"
                    << "  \"input_units\": " << task_a.input_units << ",\n"
                    << "  \"output_units\": " << task_a.requested_output_tokens << ",\n"
                    << "  \"interference_domain_id\": \""
                    << json_escape(interference_domain) << "\",\n"
                    << "  \"device_a\": \"" << json_escape(device_a) << "\",\n"
                    << "  \"device_b\": \"" << json_escape(device_b) << "\",\n"
                    << "  \"request_a_status\": \"" << json_escape(result_a.result.message) << "\",\n"
                    << "  \"request_b_status\": \"" << json_escape(result_b.result.message) << "\",\n"
                    << "  \"request_a_wrapper_elapsed_ms\": "
                    << (result_a.end_steady_us - result_a.start_steady_us) / 1000.0 << ",\n"
                    << "  \"request_b_wrapper_elapsed_ms\": "
                    << (result_b.end_steady_us - result_b.start_steady_us) / 1000.0 << ",\n"
                    << "  \"wrapper_interval_overlap_ms\": " << overlap_us / 1000.0 << ",\n"
                    << "  \"model_residency\": \"fresh_process_per_request\",\n"
                    << "  \"memory_admission_basis\": \"on_disk_model_bytes_not_peak_rss\",\n"
                    << "  \"device_domain_verification\": \"inspect_each_engine_trace\"\n"
                    << "}\n";
        result_file.flush();
        if (!result_file) throw std::runtime_error("failed writing result.json");
        std::cout << "run_id=" << args.at("--run-id") << " condition=" << condition
                  << " simulated=false request_a=" << result_a.result.message
                  << " request_b=" << result_b.result.message
                  << " wrapper_interval_overlap_ms=" << overlap_us / 1000.0 << '\n';
        return result_a.result.success && result_b.result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
