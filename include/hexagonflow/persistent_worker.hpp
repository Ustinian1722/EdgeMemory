#pragma once

#include "hexagonflow/backend.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace hexagonflow {

enum class WorkerState {
    Starting,
    Ready,
    Draining,
    Stopped,
    Faulted,
};

enum class WorkerStopMode {
    Drain,
    CancelPending,
};

struct PersistentWorkerOptions {
    std::string worker_id;
    std::string resident_model_id;
    // Scheduler estimate only. It is not an RSS or HTP-memory measurement.
    std::uint64_t estimated_resident_bytes{0};
    std::size_t max_pending_requests{8};
};

struct PersistentWorkerSnapshot {
    std::string worker_id;
    std::string resident_model_id;
    WorkerState state{WorkerState::Starting};
    std::size_t queued_requests{0};
    bool request_in_flight{false};
    std::uint64_t accepted_requests{0};
    std::uint64_t completed_requests{0};
    std::uint64_t failed_requests{0};
    std::uint64_t cancelled_requests{0};
    std::uint64_t estimated_resident_bytes{0};
    std::string startup_error;
};

/**
 * Serializes work onto one long-lived backend instance.
 *
 * A factory runs once on the worker thread, so runtimes with thread-affine
 * contexts can initialize and destroy their model on the same thread. The
 * bounded queue provides explicit backpressure; the worker itself never
 * claims that its scheduler memory estimate is measured device memory.
 */
class PersistentWorker {
public:
    using BackendFactory = std::function<std::unique_ptr<InferenceBackend>()>;

    PersistentWorker(PersistentWorkerOptions options, BackendFactory factory);
    ~PersistentWorker();

    PersistentWorker(const PersistentWorker&) = delete;
    PersistentWorker& operator=(const PersistentWorker&) = delete;

    bool wait_until_ready(std::chrono::milliseconds timeout);
    std::future<BackendResult> submit(
        Workload workload,
        std::chrono::milliseconds admission_timeout = std::chrono::milliseconds::zero());
    void stop(WorkerStopMode mode = WorkerStopMode::Drain);

    [[nodiscard]] PersistentWorkerSnapshot snapshot() const;

private:
    struct Request {
        Workload workload;
        std::chrono::steady_clock::time_point submitted_at;
        std::promise<BackendResult> completion;
    };

    void run();
    void cancel_queued_locked(const std::string& reason);

    PersistentWorkerOptions options_;
    BackendFactory factory_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::queue<std::unique_ptr<Request>> queue_;
    std::unique_ptr<InferenceBackend> backend_;
    std::thread thread_;
    PersistentWorkerSnapshot snapshot_;
    bool stop_requested_{false};
    bool drain_requested_{true};
};

std::string to_string(WorkerState state);

}  // namespace hexagonflow
