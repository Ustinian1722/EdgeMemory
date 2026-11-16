#include "hexagonflow/geniex_llm_backend.hpp"
#include "hexagonflow/persistent_worker.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string option(int argc, char** argv, const std::string& name,
                   const std::string& fallback = {}) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

std::int32_t integer_option(int argc, char** argv, const std::string& name,
                            std::int32_t fallback) {
    const auto text = option(argc, argv, name);
    if (text.empty()) return fallback;
    std::int32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("invalid integer option");
    }
    return value;
}

std::string hex_encode(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

bool hex_value(char character, unsigned char& value) {
    if (character >= '0' && character <= '9') value = static_cast<unsigned char>(character - '0');
    else if (character >= 'a' && character <= 'f') value = static_cast<unsigned char>(character - 'a' + 10);
    else if (character >= 'A' && character <= 'F') value = static_cast<unsigned char>(character - 'A' + 10);
    else return false;
    return true;
}

std::string hex_decode(std::string_view value, std::size_t max_bytes) {
    if (value.size() % 2U != 0U || value.size() / 2U > max_bytes) {
        throw std::invalid_argument("invalid prompt field length");
    }
    std::string output;
    output.reserve(value.size() / 2U);
    for (std::size_t index = 0; index < value.size(); index += 2U) {
        unsigned char high = 0;
        unsigned char low = 0;
        if (!hex_value(value[index], high) || !hex_value(value[index + 1U], low)) {
            throw std::invalid_argument("invalid prompt encoding");
        }
        const auto byte = static_cast<unsigned char>((high << 4U) | low);
        if (byte == 0U) throw std::invalid_argument("NUL is not allowed in prompt");
        output.push_back(static_cast<char>(byte));
    }
    return output;
}

std::vector<std::string_view> split_tabs(const std::string& line) {
    std::vector<std::string_view> fields;
    std::size_t begin = 0;
    for (;;) {
        const auto end = line.find('\t', begin);
        fields.emplace_back(line.data() + begin,
                            (end == std::string::npos ? line.size() : end) - begin);
        if (end == std::string::npos) break;
        begin = end + 1U;
    }
    return fields;
}

bool valid_request_id(std::string_view id) {
    if (id.empty() || id.size() > 64U) return false;
    for (const unsigned char character : id) {
        if (!((character >= 'a' && character <= 'z') ||
              (character >= 'A' && character <= 'Z') ||
              (character >= '0' && character <= '9') || character == '-' ||
              character == '_' || character == '.')) return false;
    }
    return true;
}

void print_error(std::string_view request_id, std::string_view reason) {
    std::cout << "ERROR\t" << request_id << '\t' << hex_encode(reason) << '\n' << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        hexagonflow::GenieXLLMBackendConfig config;
        config.model_path = option(argc, argv, "--model");
        config.model_id = option(argc, argv, "--model-id");
        config.backend_id = option(argc, argv, "--backend-id", "geniex-llm-service");
        config.device_id = option(argc, argv, "--device-id");
        config.library_path = option(argc, argv, "--library");
        config.plugin_directory = option(argc, argv, "--plugin-directory");
        config.context_length = integer_option(argc, argv, "--context", 2048);
        config.threads = integer_option(argc, argv, "--threads", 4);
        config.batch_size = integer_option(argc, argv, "--batch", 2048);
        config.physical_batch_size = integer_option(argc, argv, "--ubatch", 128);
        config.default_max_tokens = integer_option(argc, argv, "--max-tokens", 192);
        if (config.model_path.empty() || config.model_id.empty() || config.device_id.empty() ||
            config.context_length < 256 || config.threads < 1 || config.batch_size < 1 ||
            config.physical_batch_size < 1 || config.default_max_tokens < 1) {
            throw std::invalid_argument("missing or invalid LLM service configuration");
        }

        const auto estimated_bytes = static_cast<std::uint64_t>(
            std::filesystem::file_size(config.model_path));
        struct RuntimeInfo {
            std::mutex mutex;
            std::string runtime;
            std::string plugin;
        };
        const auto info = std::make_shared<RuntimeInfo>();
        hexagonflow::PersistentWorker worker(
            {"geniex-llm-service", config.model_id, estimated_bytes, 4},
            [config, info] {
                auto backend = std::make_unique<hexagonflow::GenieXLLMBackend>(config);
                std::lock_guard lock(info->mutex);
                info->runtime = backend->runtime_version();
                info->plugin = backend->plugin_version();
                return backend;
            });
        if (!worker.wait_until_ready(std::chrono::minutes(4))) {
            print_error("startup", "backend-initialization-failed");
            worker.stop(hexagonflow::WorkerStopMode::CancelPending);
            return 2;
        }
        std::string runtime_version;
        std::string plugin_version;
        {
            std::lock_guard lock(info->mutex);
            runtime_version = info->runtime;
            plugin_version = info->plugin;
        }
        std::cout << "READY\t1\t" << hex_encode(config.model_id) << '\t'
                  << hex_encode(config.device_id) << '\t' << hex_encode(runtime_version) << '\t'
                  << hex_encode(plugin_version) << '\t' << estimated_bytes << '\n' << std::flush;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") break;
            if (line.size() > 20000U) {
                print_error("invalid", "request-line-too-large");
                continue;
            }
            const auto fields = split_tabs(line);
            std::string_view request_id = fields.size() > 1U ? fields[1] : "invalid";
            try {
                if (fields.size() != 4U || fields[0] != "REQ" || !valid_request_id(fields[1])) {
                    throw std::invalid_argument("invalid request envelope");
                }
                std::int32_t max_tokens = 0;
                const auto parsed = std::from_chars(fields[2].data(),
                    fields[2].data() + fields[2].size(), max_tokens);
                if (parsed.ec != std::errc{} || parsed.ptr != fields[2].data() + fields[2].size() ||
                    max_tokens < 1 || max_tokens > 4096) {
                    throw std::invalid_argument("invalid output token limit");
                }
                const auto prompt = hex_decode(fields[3], 8192U);
                if (prompt.empty()) throw std::invalid_argument("empty prompt");
                hexagonflow::Workload request;
                request.id = std::string(request_id);
                request.model_id = config.model_id;
                request.kind = hexagonflow::WorkloadKind::LlmDecode;
                request.prompt_utf8 = prompt;
                request.requested_output_tokens = static_cast<std::uint64_t>(max_tokens);
                const auto result = worker.submit(std::move(request)).get();
                std::cout << std::setprecision(12) << "RES\t" << request_id << '\t'
                          << (result.success ? 1 : 0) << '\t' << (result.simulated ? 1 : 0) << '\t'
                          << result.output_tokens << '\t' << result.queue_wait_us << '\t'
                          << result.worker_elapsed_us << '\t' << result.ttft_us << '\t'
                          << result.media_us << '\t' << result.prefill_us << '\t' << result.decode_us << '\t'
                          << result.prefill_tokens_per_second << '\t' << result.decode_tokens_per_second << '\t'
                          << hex_encode(result.output_text) << '\t' << hex_encode(result.message)
                          << '\n' << std::flush;
            } catch (const std::exception& error) {
                print_error(valid_request_id(request_id) ? request_id : "invalid", error.what());
            }
        }
        worker.stop(hexagonflow::WorkerStopMode::Drain);
        std::cout << "BYE\n" << std::flush;
        return 0;
    } catch (const std::exception&) {
        print_error("startup", "invalid-service-configuration");
        return 2;
    }
}
