#include "hexagonflow/process_backend.hpp"

#include <cerrno>
#include <chrono>
#include <limits>
#include <optional>
#include <thread>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

namespace hexagonflow {
namespace {

#ifndef _WIN32
struct WaitResult {
    pid_t pid{-1};
    std::optional<std::uint64_t> peak_rss_bytes;
};

WaitResult wait_for_child(pid_t pid, int* status, int options) {
#ifdef __linux__
    struct rusage usage {};
    const auto waited = wait4(pid, status, options, &usage);
    if (waited != pid || usage.ru_maxrss <= 0) return {waited, std::nullopt};
    const auto peak_kib = static_cast<std::uint64_t>(usage.ru_maxrss);
    if (peak_kib > std::numeric_limits<std::uint64_t>::max() / 1024U) {
        return {waited, std::nullopt};
    }
    return {waited, peak_kib * 1024U};
#else
    return {waitpid(pid, status, options), std::nullopt};
#endif
}

BackendResult process_result(bool success, std::string message,
                             std::optional<std::uint64_t> peak_rss_bytes) {
    BackendResult result{success, false, 0, std::move(message)};
    result.peak_process_rss_bytes = peak_rss_bytes;
    return result;
}
#endif

}  // namespace

ProcessBackend::ProcessBackend(ProcessBackendOptions options)
    : options_(std::move(options)) {}

std::string ProcessBackend::id() const { return options_.backend_id; }
BackendFamily ProcessBackend::family() const { return options_.backend_family; }

BackendResult ProcessBackend::execute(const Workload& workload) {
    (void)workload;
    if (options_.executable.empty() || options_.output_path.empty() || options_.backend_id.empty() ||
        options_.timeout.count() <= 0) {
        return {false, false, 0, "invalid-process-configuration"};
    }
#ifdef _WIN32
    return {false, false, 0, "process-backend-requires-posix"};
#else
    std::vector<std::string> arguments{options_.executable};
    arguments.insert(arguments.end(), options_.arguments.begin(), options_.arguments.end());
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1U);
    for (auto& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);

    posix_spawn_file_actions_t file_actions;
    int error = posix_spawn_file_actions_init(&file_actions);
    if (error != 0) return {false, false, 0, "spawn-file-actions-init-failed"};
    const int output_flags = O_WRONLY | O_CREAT |
                             (options_.append_output ? O_APPEND : O_TRUNC);
    error = posix_spawn_file_actions_addopen(
        &file_actions, STDOUT_FILENO, options_.output_path.c_str(),
        output_flags, 0600);
    if (error == 0) {
        error = posix_spawn_file_actions_adddup2(
            &file_actions, STDOUT_FILENO, STDERR_FILENO);
    }
    if (error != 0) {
        posix_spawn_file_actions_destroy(&file_actions);
        return {false, false, 0, "spawn-file-actions-config-failed"};
    }

    posix_spawnattr_t attributes;
    error = posix_spawnattr_init(&attributes);
    if (error != 0) {
        posix_spawn_file_actions_destroy(&file_actions);
        return {false, false, 0, "spawn-attributes-init-failed"};
    }
    error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    if (error == 0) error = posix_spawnattr_setpgroup(&attributes, 0);
    if (error != 0) {
        posix_spawnattr_destroy(&attributes);
        posix_spawn_file_actions_destroy(&file_actions);
        return {false, false, 0, "spawn-process-group-config-failed"};
    }

    pid_t pid = -1;
    error = posix_spawnp(&pid, argv[0], &file_actions, &attributes,
                         argv.data(), environ);
    posix_spawnattr_destroy(&attributes);
    posix_spawn_file_actions_destroy(&file_actions);
    if (error != 0) return {false, false, 0, "process-spawn-failed"};

    const auto deadline = std::chrono::steady_clock::now() + options_.timeout;
    int status = 0;
    std::optional<std::uint64_t> peak_rss_bytes;
    for (;;) {
        const auto result = wait_for_child(pid, &status, WNOHANG);
        if (result.pid == pid) {
            peak_rss_bytes = result.peak_rss_bytes;
            break;
        }
        if (result.pid < 0) {
            if (errno == EINTR) continue;
            (void)kill(-pid, SIGKILL);
            auto reaped = wait_for_child(pid, &status, 0);
            while (reaped.pid < 0 && errno == EINTR) {
                reaped = wait_for_child(pid, &status, 0);
            }
            return process_result(false, "waitpid-failed", reaped.peak_rss_bytes);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            (void)kill(-pid, SIGTERM);
            const auto grace_deadline = std::chrono::steady_clock::now() +
                                        std::chrono::milliseconds(200);
            while (std::chrono::steady_clock::now() < grace_deadline) {
                const auto waited = wait_for_child(pid, &status, WNOHANG);
                if (waited.pid == pid) {
                    (void)kill(-pid, SIGKILL);
                    return process_result(false, "process-timeout",
                                          waited.peak_rss_bytes);
                }
                if (waited.pid < 0 && errno != EINTR) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            (void)kill(-pid, SIGKILL);
            auto reaped = wait_for_child(pid, &status, 0);
            while (reaped.pid < 0 && errno == EINTR) {
                reaped = wait_for_child(pid, &status, 0);
            }
            return process_result(false, "process-timeout", reaped.peak_rss_bytes);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return process_result(true, "process-completed", peak_rss_bytes);
    }
    if (WIFEXITED(status)) {
        return process_result(false,
                              "process-exit-" + std::to_string(WEXITSTATUS(status)),
                              peak_rss_bytes);
    }
    return process_result(false, "process-terminated-by-signal", peak_rss_bytes);
#endif
}

}  // namespace hexagonflow
