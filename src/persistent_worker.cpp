#include "hexagonflow/persistent_worker.hpp"

#include <exception>
#include <stdexcept>
#include <utility>

namespace hexagonflow {
namespace {

std::uint64_t elapsed_us(const std::chrono::steady_clock::time_point start,
                         const std::chrono::steady_clock::time_point end) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0U;
}

BackendResult failure(std::string reason) {
    BackendResult result;
    result.success = false;
    result.message = std::move(reason);
    return result;
}

}  // namespace

PersistentWorker::PersistentWorker(PersistentWorkerOptions options, BackendFactory factory)
    : options_(std::move(options)), factory_(std::move(factory)) {
    if (options_.worker_id.empty()) {
        throw std::invalid_argument("persistent worker requires a worker id");
    }
    if (options_.max_pending_requests == 0) {
        throw std::invalid_argument("persistent worker queue capacity must be positive");
    }
    if (!factory_) {
        throw std::invalid_argument("persistent worker requires a backend factory");
    }
    snapshot_.worker_id = options_.worker_id;
    snapshot_.resident_model_id = options_.resident_model_id;
    snapshot_.estimated_resident_bytes = options_.estimated_resident_bytes;
    thread_ = std::thread(&PersistentWorker::run, this);
}

PersistentWorker::~PersistentWorker() {
    stop(WorkerStopMode::CancelPending);
}

bool PersistentWorker::wait_until_ready(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, timeout, [this] {
        return snapshot_.state != WorkerState::Starting;
    });
    return snapshot_.state == WorkerState::Ready;
}

std::future<BackendResult> PersistentWorker::submit(
    Workload workload, std::chrono::milliseconds admission_timeout) {
    std::unique_lock lock(mutex_);
    const auto can_start = [this] { return snapshot_.state != WorkerState::Starting; };
    if (snapshot_.state == WorkerState::Starting && admission_timeout > std::chrono::milliseconds::zero()) {
        changed_.wait_for(lock, admission_timeout, can_start);
    }
    if (snapshot_.state == WorkerState::Starting) {
        throw std::runtime_error("worker-not-ready");
    }
    if (snapshot_.state != WorkerState::Ready) {
        throw std::runtime_error(snapshot_.state == WorkerState::Faulted
                                     ? "worker-faulted" : "worker-not-accepting-requests");
    }
    if (queue_.size() >= options_.max_pending_requests) {
        if (admission_timeout <= std::chrono::milliseconds::zero() ||
            !changed_.wait_for(lock, admission_timeout, [this] {
                return queue_.size() < options_.max_pending_requests ||
                       snapshot_.state != WorkerState::Ready;
            })) {
            throw std::runtime_error("worker-queue-full");
        }
        if (snapshot_.state != WorkerState::Ready) {
            throw std::runtime_error("worker-not-accepting-requests");
        }
    }

    auto request = std::make_unique<Request>();
    request->workload = std::move(workload);
    request->submitted_at = std::chrono::steady_clock::now();
    auto future = request->completion.get_future();
    queue_.push(std::move(request));
    ++snapshot_.accepted_requests;
    changed_.notify_all();
    return future;
}

void PersistentWorker::cancel_queued_locked(const std::string& reason) {
    while (!queue_.empty()) {
        auto request = std::move(queue_.front());
        queue_.pop();
        try {
            request->completion.set_value(failure(reason));
        } catch (const std::future_error&) {
            // The request future is already resolved; continue cancelling others.
        }
        ++snapshot_.cancelled_requests;
    }
}

void PersistentWorker::stop(WorkerStopMode mode) {
    {
        std::lock_guard lock(mutex_);
        if (snapshot_.state == WorkerState::Stopped && !thread_.joinable()) {
            return;
        }
        stop_requested_ = true;
        drain_requested_ = mode == WorkerStopMode::Drain;
        if (!drain_requested_) {
            cancel_queued_locked("worker-request-cancelled");
        }
        if (snapshot_.state == WorkerState::Ready) {
            snapshot_.state = WorkerState::Draining;
        }
        changed_.notify_all();
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

PersistentWorkerSnapshot PersistentWorker::snapshot() const {
    std::lock_guard lock(mutex_);
    auto copy = snapshot_;
    copy.queued_requests = queue_.size();
    return copy;
}

void PersistentWorker::run() {
    try {
        backend_ = factory_();
        if (!backend_) {
            throw std::runtime_error("backend factory returned null");
        }
    } catch (const std::exception&) {
        std::lock_guard lock(mutex_);
        snapshot_.state = WorkerState::Faulted;
        snapshot_.startup_error = "backend-initialization-failed";
        cancel_queued_locked("worker-backend-initialization-failed");
        changed_.notify_all();
        return;
    } catch (...) {
        std::lock_guard lock(mutex_);
        snapshot_.state = WorkerState::Faulted;
        snapshot_.startup_error = "backend-initialization-failed";
        cancel_queued_locked("worker-backend-initialization-failed");
        changed_.notify_all();
        return;
    }

    {
        std::lock_guard lock(mutex_);
        snapshot_.state = stop_requested_ ? WorkerState::Draining : WorkerState::Ready;
        changed_.notify_all();
    }

    while (true) {
        std::unique_ptr<Request> request;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [this] {
                return stop_requested_ || !queue_.empty();
            });
            if (stop_requested_ && (!drain_requested_ || queue_.empty())) {
                break;
            }
            request = std::move(queue_.front());
            queue_.pop();
            snapshot_.request_in_flight = true;
            changed_.notify_all();
        }

        const auto started = std::chrono::steady_clock::now();
        BackendResult result;
        try {
            result = backend_->execute(request->workload);
        } catch (const std::exception&) {
            result = failure("backend-exception");
        } catch (...) {
            result = failure("backend-unknown-exception");
        }
        const auto finished = std::chrono::steady_clock::now();
        result.queue_wait_us = elapsed_us(request->submitted_at, started);
        result.worker_elapsed_us = elapsed_us(started, finished);

        {
            std::lock_guard lock(mutex_);
            snapshot_.request_in_flight = false;
            if (result.success) {
                ++snapshot_.completed_requests;
            } else {
                ++snapshot_.failed_requests;
            }
            try {
                request->completion.set_value(std::move(result));
            } catch (const std::future_error&) {
                // A caller may have abandoned its future; request cleanup still proceeds.
            }
            changed_.notify_all();
        }
    }

    backend_.reset();
    {
        std::lock_guard lock(mutex_);
        snapshot_.request_in_flight = false;
        snapshot_.state = WorkerState::Stopped;
        changed_.notify_all();
    }
}

std::string to_string(WorkerState state) {
    switch (state) {
        case WorkerState::Starting: return "starting";
        case WorkerState::Ready: return "ready";
        case WorkerState::Draining: return "draining";
        case WorkerState::Stopped: return "stopped";
        case WorkerState::Faulted: return "faulted";
    }
    return "unknown";
}

}  // namespace hexagonflow
