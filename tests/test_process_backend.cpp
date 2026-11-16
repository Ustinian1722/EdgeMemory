#include "hexagonflow/metrics.hpp"
#include "hexagonflow/process_backend.hpp"
#include "test_framework.hpp"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <thread>

#ifndef _WIN32
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifndef _WIN32
HF_TEST(process_backend_reports_success_and_failure) {
    hexagonflow::Workload task;
    hexagonflow::ProcessBackend success({"/bin/true", {}, "process-test.log",
                                        std::chrono::seconds(2)});
    auto result = success.execute(task);
    HF_REQUIRE(result.success);
    HF_REQUIRE(!result.simulated);
    hexagonflow::ProcessBackend failure({"/bin/false", {}, "process-test.log",
                                        std::chrono::seconds(2)});
    result = failure.execute(task);
    HF_REQUIRE(!result.success);
    HF_REQUIRE(result.message == "process-exit-1");
#ifdef __linux__
    HF_REQUIRE(result.peak_process_rss_bytes.has_value());
    HF_REQUIRE(*result.peak_process_rss_bytes > 0);
#endif
    std::remove("process-test.log");
}

HF_TEST(process_backend_supports_bounded_append_output) {
    hexagonflow::Workload task;
    std::remove("process-append.log");
    hexagonflow::ProcessBackend backend({
        "/bin/echo", {"repeat"}, "process-append.log", std::chrono::seconds(2),
        "append-test", hexagonflow::BackendFamily::Cpu, true});
    HF_REQUIRE(backend.execute(task).success);
    HF_REQUIRE(backend.execute(task).success);
    std::ifstream output("process-append.log");
    const std::string contents((std::istreambuf_iterator<char>(output)),
                               std::istreambuf_iterator<char>());
    HF_REQUIRE(contents == "repeat\nrepeat\n");
    std::remove("process-append.log");
}

HF_TEST(process_backend_reports_peak_child_rss_in_metrics) {
    hexagonflow::Workload task;
    hexagonflow::ProcessBackend backend({"/bin/sleep", {"0.1"},
                                        "process-rss.log", std::chrono::seconds(2)});
    const auto result = backend.execute(task);
    HF_REQUIRE(result.success);
#ifdef __linux__
    HF_REQUIRE(result.peak_process_rss_bytes.has_value());
    HF_REQUIRE(*result.peak_process_rss_bytes >= 64U * 1024U);
#else
    HF_REQUIRE(!result.peak_process_rss_bytes.has_value());
#endif

    const std::string metrics_path = "process-rss-metrics.jsonl";
    std::remove(metrics_path.c_str());
    {
        hexagonflow::JsonlMetricsSink metrics(metrics_path);
        metrics.append({"rss-run", "rss-task", "rss-model", "cpu", backend.id(),
                        result.message, false, 1U, 2U, 0U, 0U, false,
                        result.peak_process_rss_bytes});
    }
    std::ifstream input(metrics_path);
    const std::string line((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    HF_REQUIRE(line.find("\"peak_process_rss_bytes\":") != std::string::npos);
    if (result.peak_process_rss_bytes.has_value()) {
        HF_REQUIRE(line.find(std::to_string(*result.peak_process_rss_bytes)) !=
                   std::string::npos);
        HF_REQUIRE(line.find("\"peak_process_rss_scope\":\"direct_child\"") !=
                   std::string::npos);
    } else {
        HF_REQUIRE(line.find("\"peak_process_rss_bytes\":null") != std::string::npos);
        HF_REQUIRE(line.find("\"peak_process_rss_scope\":null") != std::string::npos);
    }
    std::remove(metrics_path.c_str());
    std::remove("process-rss.log");
}

HF_TEST(process_backend_enforces_timeout) {
    hexagonflow::Workload task;
    hexagonflow::ProcessBackend backend({"/bin/sleep", {"5"}, "process-timeout.log",
                                        std::chrono::milliseconds(80)});
    const auto result = backend.execute(task);
    HF_REQUIRE(!result.success);
    HF_REQUIRE(result.message == "process-timeout");
#ifdef __linux__
    HF_REQUIRE(result.peak_process_rss_bytes.has_value());
    HF_REQUIRE(*result.peak_process_rss_bytes > 0);
#endif
    std::remove("process-timeout.log");
}

HF_TEST(process_backend_timeout_terminates_spawned_process_group) {
    hexagonflow::Workload task;
    std::remove("process-child.pid");
    hexagonflow::ProcessBackend backend({
        "/bin/sh",
        {"-c", "sleep 10 & echo $! > process-child.pid; wait"},
        // Allow the shell to write its child PID before the watchdog fires.
        // Very short deadlines made this assertion race on slower WSL/D: runs.
        "process-group-timeout.log", std::chrono::milliseconds(500)});
    const auto result = backend.execute(task);
    HF_REQUIRE(!result.success);
    HF_REQUIRE(result.message == "process-timeout");

    long child = -1;
    {
        std::ifstream pid_file("process-child.pid");
        pid_file >> child;
    }
    HF_REQUIRE(child > 0);
    bool stopped = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (kill(static_cast<pid_t>(child), 0) != 0 && errno == ESRCH) {
            stopped = true;
            break;
        }
        std::ifstream stat_file("/proc/" + std::to_string(child) + "/stat");
        std::string stat_line;
        if (std::getline(stat_file, stat_line)) {
            const auto close = stat_line.rfind(')');
            if (close != std::string::npos && close + 2 < stat_line.size() &&
                stat_line[close + 2] == 'Z') {
                stopped = true;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    HF_REQUIRE(stopped);
    std::remove("process-child.pid");
    std::remove("process-group-timeout.log");
}
#endif
