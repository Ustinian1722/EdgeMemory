#include "hexagonflow/metrics.hpp"

#include <stdexcept>

namespace hexagonflow {
namespace {

std::string json_escape(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size() + 2U);
    for (const char raw_character : input) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (character < 0x20U) {
                    escaped += '?';
                } else {
                    escaped += static_cast<char>(character);
                }
        }
    }
    return escaped;
}

}  // namespace

JsonlMetricsSink::JsonlMetricsSink(const std::string& path)
    : stream_(path, std::ios::out | std::ios::app) {
    if (!stream_) {
        throw std::runtime_error("cannot open metrics file: " + path);
    }
}

void JsonlMetricsSink::append(const ExecutionRecord& record) {
    stream_ << "{"
            << "\"run_id\":\"" << json_escape(record.run_id) << "\","
            << "\"task_id\":\"" << json_escape(record.task_id) << "\","
            << "\"model_id\":\"" << json_escape(record.model_id) << "\","
            << "\"resource_id\":\"" << json_escape(record.resource_id)
            << "\","
            << "\"backend_id\":\"" << json_escape(record.backend_id)
            << "\","
            << "\"status\":\"" << json_escape(record.status) << "\","
            << "\"simulated\":" << (record.simulated ? "true" : "false")
            << ","
            << "\"start_unix_us\":" << record.start_unix_us << ","
            << "\"end_unix_us\":" << record.end_unix_us << ","
            << "\"input_tokens\":" << record.input_tokens << ","
            << "\"output_tokens\":" << record.output_tokens << ","
            << "\"token_counts_known\":"
            << (record.token_counts_known ? "true" : "false") << ","
            << "\"peak_process_rss_bytes\":";
    if (record.peak_process_rss_bytes.has_value()) {
        stream_ << *record.peak_process_rss_bytes;
    } else {
        stream_ << "null";
    }
    stream_ << ",\"peak_process_rss_scope\":";
    if (record.peak_process_rss_bytes.has_value()) {
        stream_ << "\"direct_child\"";
    } else {
        stream_ << "null";
    }
    stream_ << "}" << '\n';
    stream_.flush();
    if (!stream_) {
        throw std::runtime_error("failed writing metrics record");
    }
}

}  // namespace hexagonflow
