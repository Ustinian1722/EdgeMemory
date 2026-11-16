#include "hexagonflow/metrics.hpp"
#include "hexagonflow/process_backend.hpp"
#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
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
            default:
                escaped += character < 0x20U ? '?' : static_cast<char>(character);
        }
    }
    return escaped;
}

unsigned long number(const std::map<std::string, std::string>& args,
                     const std::string& key,
                     unsigned long fallback,
                     unsigned long minimum,
                     unsigned long maximum) {
    const auto found = args.find(key);
    const auto value = found == args.end() ? fallback : std::stoul(found->second);
    if (value < minimum || value > maximum) {
        throw std::invalid_argument("out-of-range " + key);
    }
    return value;
}

struct Observation {
    hexagonflow::BackendResult result;
    std::uint64_t start_unix_us{0};
    std::uint64_t end_unix_us{0};
    std::int64_t start_steady_us{0};
    std::int64_t end_steady_us{0};
};

template <typename Backend>
Observation execute(Backend& backend, const hexagonflow::Workload& workload) {
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

Observation execute_repeated(hexagonflow::ProcessBackend& backend,
                             const hexagonflow::Workload& workload,
                             unsigned long repeats) {
    Observation observation;
    const auto start = Clock::now();
    observation.start_unix_us = unix_us();
    observation.start_steady_us = std::chrono::duration_cast<std::chrono::microseconds>(
        start.time_since_epoch()).count();
    for (unsigned long index = 0; index < repeats; ++index) {
        observation.result = backend.execute(workload);
        if (!observation.result.success) break;
    }
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
                "--geniex-bench", "--llm-model", "--embedding-engine", "--embedding-model",
                "--run-id", "--output-dir", "--device-id", "--prompt-tokens", "--gen-tokens",
                "--repetitions", "--threads", "--embedding-threads", "--embedding-items",
                "--embedding-repeats", "--condition", "--timeout-s", "--geniex-log-level"};
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
                throw std::invalid_argument("unknown option: " + key);
            }
            if (!args.emplace(key, argv[index + 1]).second) {
                throw std::invalid_argument("duplicate option: " + key);
            }
        }
        for (const auto* key : {"--geniex-bench", "--llm-model", "--embedding-engine",
                                "--embedding-model", "--run-id", "--output-dir", "--device-id"}) {
            if (!args.contains(key) || args.at(key).empty()) {
                throw std::invalid_argument(std::string("missing ") + key);
            }
        }

        const auto prompt_tokens = number(args, "--prompt-tokens", 128, 1, 4096);
        const auto gen_tokens = number(args, "--gen-tokens", 64, 1, 256);
        const auto repetitions = number(args, "--repetitions", 3, 1, 20);
        const auto llm_threads = number(args, "--threads", 4, 1, 64);
        const auto embedding_threads = number(args, "--embedding-threads", 2, 1, 64);
        const auto embedding_items = number(args, "--embedding-items", 48, 3, 100);
        const auto embedding_repeats = number(args, "--embedding-repeats", 1, 1, 12);
        const auto timeout_s = number(args, "--timeout-s", 300, 1, 900);
        const auto condition = args.contains("--condition") ? args.at("--condition") : "concurrent";
        if (condition != "concurrent" && condition != "isolated") {
            throw std::invalid_argument("--condition must be concurrent or isolated");
        }
        const auto log_level = args.contains("--geniex-log-level")
                                   ? args.at("--geniex-log-level") : "error";
        if (log_level != "none" && log_level != "error" && log_level != "warn" &&
            log_level != "info" && log_level != "debug" && log_level != "trace") {
            throw std::invalid_argument("invalid --geniex-log-level");
        }

        const std::filesystem::path output_dir(args.at("--output-dir"));
        std::filesystem::create_directories(output_dir);
        const auto llm_model_bytes = std::filesystem::file_size(args.at("--llm-model"));
        const auto embedding_model_bytes = std::filesystem::file_size(args.at("--embedding-model"));
        const auto llm_json = (output_dir / "llm-benchmark.json").string();
        const auto llm_log = (output_dir / "llm-process.log").string();
        const auto embedding_log = (output_dir / "embedding-process.log").string();
        const auto metrics_path = (output_dir / "metrics.jsonl").string();
        const auto result_path = (output_dir / "result.json").string();

        hexagonflow::ProcessBackend llm_backend({
            args.at("--geniex-bench"),
            {"--plugin", "llama_cpp", "--device", "npu", "--device-id", args.at("--device-id"),
             "-m", args.at("--llm-model"), "-p", std::to_string(prompt_tokens),
             "-n", std::to_string(gen_tokens), "-c",
             std::to_string(std::max(256UL, prompt_tokens + gen_tokens + 16UL)),
             "-r", std::to_string(repetitions), "--warmup", "1", "-t", std::to_string(llm_threads),
             "--seed", "42", "--log", log_level, "--output-json", llm_json,
             "--cell-id", args.at("--run-id")},
            llm_log, std::chrono::seconds(timeout_s), "geniex-llm-npu", hexagonflow::BackendFamily::GenieX});

        static const std::vector<std::string> prompts{
            "A cat sits on a mat.", "A kitten is sitting on the mat.",
            "A rocket launches into outer space."};
        std::ostringstream batch;
        for (unsigned long index = 0; index < embedding_items; ++index) {
            if (index != 0) batch << " <#sep#> ";
            batch << prompts[static_cast<std::size_t>(index % prompts.size())];
        }
        std::remove(embedding_log.c_str());
        hexagonflow::ProcessBackend embedding_backend({
            args.at("--embedding-engine"),
            {"-m", args.at("--embedding-model"), "-p", batch.str(),
             "--pooling", "mean", "--embd-separator", "<#sep#>",
             "--embd-normalize", "2", "--embd-output-format", "json+",
             "-ngl", "0", "-t", std::to_string(embedding_threads), "-c", "512"},
            embedding_log, std::chrono::seconds(timeout_s), "llama-embedding-cpu",
            hexagonflow::BackendFamily::Cpu, true});

        hexagonflow::ResourceManager resources({
            {"npu-requested", llm_backend.id(), llm_backend.family(), 1U, llm_model_bytes},
            {"cpu", embedding_backend.id(), embedding_backend.family(), 1U, embedding_model_bytes}});
        hexagonflow::Scheduler scheduler(hexagonflow::SchedulingPolicy::Fifo, resources);
        hexagonflow::Workload llm;
        llm.id = "llm-request";
        llm.model_id = "smollm2-q4_0";
        llm.kind = hexagonflow::WorkloadKind::LlmDecode;
        llm.model_memory_bytes = llm_model_bytes;  // On-disk admission estimate, not runtime RSS.
        llm.requested_output_tokens = gen_tokens;
        llm.allowed_backend_ids = {llm_backend.id()};
        hexagonflow::Workload embedding;
        embedding.id = "embedding-request";
        embedding.model_id = "all-minilm-l6-q8_0";
        embedding.kind = hexagonflow::WorkloadKind::Embedding;
        embedding.model_memory_bytes = embedding_model_bytes;  // On-disk admission estimate.
        embedding.allowed_backend_ids = {embedding_backend.id()};
        if (!scheduler.submit(llm) || !scheduler.submit(embedding)) {
            throw std::runtime_error("request submission failed");
        }
        const auto decisions = scheduler.dispatch_ready();
        if (decisions.size() != 2 ||
            std::any_of(decisions.begin(), decisions.end(), [](const auto& decision) {
                return !decision.dispatched;
            })) {
            throw std::runtime_error("scheduler could not reserve both backend resources");
        }

        Observation llm_result;
        Observation embedding_result;
        if (condition == "concurrent") {
            auto llm_future = std::async(std::launch::async,
                                         [&] { return execute(llm_backend, llm); });
            auto embedding_future = std::async(std::launch::async, [&] {
                return execute_repeated(embedding_backend, embedding, embedding_repeats);
            });
            llm_result = llm_future.get();
            embedding_result = embedding_future.get();
        } else {
            llm_result = execute(llm_backend, llm);
            embedding_result = execute_repeated(embedding_backend, embedding, embedding_repeats);
        }
        if (llm_result.result.success) (void)scheduler.complete(llm.id);
        else (void)scheduler.fail(llm.id, llm_result.result.message);
        if (embedding_result.result.success) (void)scheduler.complete(embedding.id);
        else (void)scheduler.fail(embedding.id, embedding_result.result.message);

        const auto overlap_start = std::max(llm_result.start_steady_us,
                                            embedding_result.start_steady_us);
        const auto overlap_end = std::min(llm_result.end_steady_us,
                                          embedding_result.end_steady_us);
        const auto overlap_us = std::max<std::int64_t>(0, overlap_end - overlap_start);

        hexagonflow::JsonlMetricsSink metrics(metrics_path);
        metrics.append({args.at("--run-id"), llm.id, llm.model_id, "npu-requested",
                        llm_backend.id(), llm_result.result.message, false,
                        llm_result.start_unix_us, llm_result.end_unix_us, 0, 0, false,
                        llm_result.result.peak_process_rss_bytes});
        metrics.append({args.at("--run-id"), embedding.id, embedding.model_id, "cpu",
                        embedding_backend.id(), embedding_result.result.message, false,
                        embedding_result.start_unix_us, embedding_result.end_unix_us, 0, 0, false,
                        embedding_result.result.peak_process_rss_bytes});

        std::ofstream result_file(result_path, std::ios::out | std::ios::trunc);
        if (!result_file) throw std::runtime_error("cannot create result.json");
        result_file << "{\n"
                    << "  \"run_id\": \"" << json_escape(args.at("--run-id")) << "\",\n"
                    << "  \"condition\": \"" << condition << "\",\n"
                    << "  \"embedding_repeats\": " << embedding_repeats << ",\n"
                    << "  \"simulated\": false,\n"
                    << "  \"requested_device_id\": \"" << json_escape(args.at("--device-id")) << "\",\n"
                    << "  \"geniex_log_level\": \"" << json_escape(log_level) << "\",\n"
                    << "  \"llm_status\": \"" << json_escape(llm_result.result.message) << "\",\n"
                    << "  \"embedding_status\": \"" << json_escape(embedding_result.result.message) << "\",\n"
                    << "  \"llm_wrapper_elapsed_ms\": "
                    << (llm_result.end_steady_us - llm_result.start_steady_us) / 1000.0 << ",\n"
                    << "  \"embedding_wrapper_elapsed_ms\": "
                    << (embedding_result.end_steady_us - embedding_result.start_steady_us) / 1000.0 << ",\n"
                    << "  \"wrapper_interval_overlap_ms\": " << overlap_us / 1000.0 << ",\n"
                    << "  \"placement_verification\": \"inspect_llm_benchmark_json_and_runtime_trace\",\n"
                    << "  \"memory_admission_basis\": \"on_disk_model_bytes_not_peak_rss\",\n"
                    << "  \"residency\": \"fresh_process_per_request\",\n"
                    << "  \"token_counts_known\": false\n"
                    << "}\n";
        result_file.flush();
        if (!result_file) throw std::runtime_error("failed writing result.json");

        std::cout << "run_id=" << args.at("--run-id")
                  << " condition=" << condition
                  << " simulated=false llm_status=" << llm_result.result.message
                  << " embedding_status=" << embedding_result.result.message
                  << " wrapper_interval_overlap_ms=" << overlap_us / 1000.0
                  << " placement=verify_json_and_trace\n";
        return llm_result.result.success && embedding_result.result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
