#include "hexagonflow/geniex_vlm_backend.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& value) {
    std::string output;
    output.reserve(value.size() + 8U);
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
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

std::string option(int argc, char** argv, const std::string& name,
                   const std::string& fallback = {}) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

int integer_option(int argc, char** argv, const std::string& name, int fallback) {
    const auto value = option(argc, argv, name);
    return value.empty() ? fallback : std::stoi(value);
}

void require_value(const std::string& name, const std::string& value) {
    if (value.empty()) throw std::invalid_argument("missing " + name);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        hexagonflow::GenieXVlmBackendConfig config;
        config.model_path = option(argc, argv, "--model");
        config.mmproj_path = option(argc, argv, "--mmproj");
        config.model_id = option(argc, argv, "--model-id");
        config.device_id = option(argc, argv, "--device-id");
        config.context_length = integer_option(argc, argv, "--context", 2048);
        config.threads = integer_option(argc, argv, "--threads", 4);
        config.default_max_tokens = integer_option(argc, argv, "--max-tokens", 64);
        const int repetitions = integer_option(argc, argv, "--repetitions", 2);
        const std::string image = option(argc, argv, "--image");
        const std::string prompt = option(argc, argv, "--prompt",
            "Describe the visible objects and actions in one concise sentence. Do not identify people or infer emotions.");
        require_value("--model", config.model_path);
        require_value("--mmproj", config.mmproj_path);
        require_value("--image", image);
        require_value("--model-id", config.model_id);
        require_value("--device-id", config.device_id);
        if (repetitions < 1 || repetitions > 4) throw std::invalid_argument("repetitions must be in 1..4");

        hexagonflow::GenieXVlmBackend backend(config);
        std::cout << "{\"event\":\"model_ready\",\"runtime_version\":\""
                  << json_escape(backend.runtime_version()) << "\",\"plugin_version\":\""
                  << json_escape(backend.plugin_version()) << "\",\"backend_id\":\""
                  << json_escape(backend.id()) << "\"}\n" << std::flush;
        for (int index = 0; index < repetitions; ++index) {
            hexagonflow::Workload workload;
            workload.id = "vlm-smoke-" + std::to_string(index + 1);
            workload.model_id = config.model_id;
            workload.kind = hexagonflow::WorkloadKind::Custom;
            workload.prompt_utf8 = prompt;
            workload.media_paths = {image};
            workload.requested_output_tokens = static_cast<std::uint64_t>(config.default_max_tokens);
            const auto result = backend.execute(workload);
            std::cout << "{\"event\":\"inference\",\"request_index\":" << (index + 1)
                      << ",\"success\":" << (result.success ? "true" : "false")
                      << ",\"simulated\":" << (result.simulated ? "true" : "false")
                      << ",\"output_tokens\":" << result.output_tokens
                      << ",\"ttft_us\":" << result.ttft_us
                      << ",\"media_us\":" << result.media_us
                      << ",\"prefill_us\":" << result.prefill_us
                      << ",\"decode_us\":" << result.decode_us
                      << ",\"prefill_tokens_per_second\":" << result.prefill_tokens_per_second
                      << ",\"decode_tokens_per_second\":" << result.decode_tokens_per_second
                      << ",\"message\":\"" << json_escape(result.message)
                      << "\",\"text\":\"" << json_escape(result.output_text) << "\"}\n"
                      << std::flush;
            if (!result.success) return 3;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "geniex-vlm-smoke: " << error.what() << '\n';
        return 2;
    }
}
