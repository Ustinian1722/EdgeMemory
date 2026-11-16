#include "hexagonflow/geniex_vlm_backend.hpp"
#include "hexagonflow/persistent_worker.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
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
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("invalid integer option");
    }
    return value;
}

std::string hex_encode(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

bool hex_value(char character, unsigned char& value) {
    if (character >= '0' && character <= '9') {
        value = static_cast<unsigned char>(character - '0');
        return true;
    }
    if (character >= 'a' && character <= 'f') {
        value = static_cast<unsigned char>(character - 'a' + 10);
        return true;
    }
    if (character >= 'A' && character <= 'F') {
        value = static_cast<unsigned char>(character - 'A' + 10);
        return true;
    }
    return false;
}

std::string hex_decode(std::string_view value, std::size_t max_bytes) {
    if (value.size() % 2U != 0U || value.size() / 2U > max_bytes) {
        throw std::invalid_argument("invalid protocol field length");
    }
    std::string output;
    output.reserve(value.size() / 2U);
    for (std::size_t index = 0; index < value.size(); index += 2U) {
        unsigned char high = 0;
        unsigned char low = 0;
        if (!hex_value(value[index], high) || !hex_value(value[index + 1U], low)) {
            throw std::invalid_argument("invalid protocol encoding");
        }
        const auto byte = static_cast<unsigned char>((high << 4U) | low);
        if (byte == 0U) throw std::invalid_argument("NUL is not allowed in protocol fields");
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
              (character >= '0' && character <= '9') ||
              character == '-' || character == '_' || character == '.')) {
            return false;
        }
    }
    return true;
}

void print_error(std::string_view request_id, std::string_view reason) {
    std::cout << "ERROR\t" << request_id << '\t' << hex_encode(reason) << '\n' << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        hexagonflow::GenieXVlmBackendConfig config;
        config.model_path = option(argc, argv, "--model");
        config.mmproj_path = option(argc, argv, "--mmproj");
        config.model_id = option(argc, argv, "--model-id");
        config.backend_id = option(argc, argv, "--backend-id", "geniex-vlm-service");
        config.device_id = option(argc, argv, "--device-id");
        config.vision_device_id = option(argc, argv, "--vit-device-id");
        config.library_path = option(argc, argv, "--library");
        config.plugin_directory = option(argc, argv, "--plugin-directory");
        config.context_length = integer_option(argc, argv, "--context", 2048);
        config.threads = integer_option(argc, argv, "--threads", 4);
        config.default_max_tokens = integer_option(argc, argv, "--max-tokens", 96);
        if (config.model_path.empty() || config.mmproj_path.empty() || config.model_id.empty() ||
            config.device_id.empty() || config.context_length < 1 || config.threads < 1 ||
            config.default_max_tokens < 1) {
            throw std::invalid_argument("missing or invalid service configuration");
        }

        const auto estimated_bytes = static_cast<std::uint64_t>(
            std::filesystem::file_size(config.model_path) +
            std::filesystem::file_size(config.mmproj_path));
        struct RuntimeInfo {
            std::mutex mutex;
            std::string runtime;
            std::string plugin;
        };
        const auto info = std::make_shared<RuntimeInfo>();
        hexagonflow::PersistentWorker worker(
            {"geniex-vlm-service", config.model_id, estimated_bytes, 4},
            [config, info] {
                auto backend = std::make_unique<hexagonflow::GenieXVlmBackend>(config);
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
                  << hex_encode(config.device_id) << '\t'
                  << hex_encode(runtime_version) << '\t'
                  << hex_encode(plugin_version) << '\t' << estimated_bytes << '\n'
                  << std::flush;

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
                if (fields.size() != 5U || fields[0] != "REQ" ||
                    !valid_request_id(fields[1])) {
                    throw std::invalid_argument("invalid request envelope");
                }
                std::int32_t max_tokens = 0;
                const auto token_text = fields[2];
                const auto parsed = std::from_chars(
                    token_text.data(), token_text.data() + token_text.size(), max_tokens);
                if (parsed.ec != std::errc{} || parsed.ptr != token_text.data() + token_text.size() ||
                    max_tokens < 1 || max_tokens > 4096) {
                    throw std::invalid_argument("invalid output token limit");
                }
                const auto prompt = hex_decode(fields[3], 8192U);
                const auto image = hex_decode(fields[4], 4096U);
                if (prompt.empty() || image.empty()) {
                    throw std::invalid_argument("prompt and image path are required");
                }

                hexagonflow::Workload request;
                request.id = std::string(request_id);
                request.model_id = config.model_id;
                request.kind = hexagonflow::WorkloadKind::Custom;
                request.prompt_utf8 = prompt;
                request.media_paths = {image};
                request.requested_output_tokens = static_cast<std::uint64_t>(max_tokens);
                const auto result = worker.submit(std::move(request)).get();
                std::cout << std::setprecision(12)
                          << "RES\t" << request_id << '\t'
                          << (result.success ? 1 : 0) << '\t'
                          << (result.simulated ? 1 : 0) << '\t'
                          << result.output_tokens << '\t'
                          << result.queue_wait_us << '\t' << result.worker_elapsed_us << '\t'
                          << result.ttft_us << '\t' << result.media_us << '\t'
                          << result.prefill_us << '\t' << result.decode_us << '\t'
                          << result.prefill_tokens_per_second << '\t'
                          << result.decode_tokens_per_second << '\t'
                          << hex_encode(result.output_text) << '\t'
                          << hex_encode(result.message) << '\n' << std::flush;
            } catch (const std::exception& error) {
                print_error(valid_request_id(request_id) ? request_id : "invalid",
                            error.what());
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
