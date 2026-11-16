#include "hexagonflow/persistent_worker.hpp"
#include "test_framework.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

struct BackendState {
    std::atomic<int> constructions{0};
    std::atomic<int> executions{0};
    std::mutex mutex;
    std::condition_variable changed;
    bool block{false};
    bool release{false};
    bool started{false};
};

class CountingBackend final : public hexagonflow::InferenceBackend {
public:
    explicit CountingBackend(std::shared_ptr<BackendState> state)
        : state_(std::move(state)) {
        ++state_->constructions;
    }

    std::string id() const override { return "persistent-counting"; }
    hexagonflow::BackendFamily family() const override { return hexagonflow::BackendFamily::Cpu; }

    hexagonflow::BackendResult execute(const hexagonflow::Workload& workload) override {
        ++state_->executions;
        {
            std::unique_lock lock(state_->mutex);
            state_->started = true;
            state_->changed.notify_all();
            state_->changed.wait(lock, [this] { return !state_->block || state_->release; });
        }
        return {true, false, workload.requested_output_tokens, "fixture execution"};
    }

private:
    std::shared_ptr<BackendState> state_;
};

bool wait_for_started(const std::shared_ptr<BackendState>& state) {
    std::unique_lock lock(state->mutex);
    return state->changed.wait_for(lock, std::chrono::seconds(2), [&] { return state->started; });
}

void release_backend(const std::shared_ptr<BackendState>& state) {
    std::lock_guard lock(state->mutex);
    state->release = true;
    state->block = false;
    state->changed.notify_all();
}

hexagonflow::PersistentWorkerOptions options(std::size_t capacity = 4) {
    return {"worker-0", "fixture-model", 4096U, capacity};
}

}  // namespace

HF_TEST(persistent_worker_reuses_one_backend_and_drains_on_shutdown) {
    auto state = std::make_shared<BackendState>();
    hexagonflow::PersistentWorker worker(options(), [state] {
        return std::make_unique<CountingBackend>(state);
    });
    HF_REQUIRE(worker.wait_until_ready(std::chrono::seconds(2)));

    hexagonflow::Workload first;
    first.id = "first";
    first.requested_output_tokens = 7;
    auto first_result = worker.submit(std::move(first));
    hexagonflow::Workload second;
    second.id = "second";
    second.requested_output_tokens = 9;
    auto second_result = worker.submit(std::move(second));
    HF_REQUIRE(first_result.get().success);
    HF_REQUIRE(second_result.get().output_tokens == 9U);
    worker.stop(hexagonflow::WorkerStopMode::Drain);

    const auto snapshot = worker.snapshot();
    HF_REQUIRE(snapshot.state == hexagonflow::WorkerState::Stopped);
    HF_REQUIRE(snapshot.accepted_requests == 2U);
    HF_REQUIRE(snapshot.completed_requests == 2U);
    HF_REQUIRE(snapshot.resident_model_id == "fixture-model");
    HF_REQUIRE(snapshot.estimated_resident_bytes == 4096U);
    HF_REQUIRE(state->constructions.load() == 1);
    HF_REQUIRE(state->executions.load() == 2);
}

HF_TEST(persistent_worker_applies_bounded_queue_backpressure) {
    auto state = std::make_shared<BackendState>();
    state->block = true;
    hexagonflow::PersistentWorker worker(options(1), [state] {
        return std::make_unique<CountingBackend>(state);
    });
    HF_REQUIRE(worker.wait_until_ready(std::chrono::seconds(2)));

    auto first = worker.submit(hexagonflow::Workload{});
    HF_REQUIRE(wait_for_started(state));
    auto second = worker.submit(hexagonflow::Workload{});
    bool rejected = false;
    try {
        (void)worker.submit(hexagonflow::Workload{});
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()) == "worker-queue-full";
    }
    HF_REQUIRE(rejected);
    release_backend(state);
    HF_REQUIRE(first.get().success);
    HF_REQUIRE(second.get().success);
    worker.stop();
}

HF_TEST(persistent_worker_cancels_pending_requests_and_finishes_active_work) {
    auto state = std::make_shared<BackendState>();
    state->block = true;
    hexagonflow::PersistentWorker worker(options(2), [state] {
        return std::make_unique<CountingBackend>(state);
    });
    HF_REQUIRE(worker.wait_until_ready(std::chrono::seconds(2)));
    auto active = worker.submit(hexagonflow::Workload{});
    HF_REQUIRE(wait_for_started(state));
    auto queued = worker.submit(hexagonflow::Workload{});

    std::thread stopper([&worker] {
        worker.stop(hexagonflow::WorkerStopMode::CancelPending);
    });
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (worker.snapshot().cancelled_requests == 1U) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    release_backend(state);
    stopper.join();

    HF_REQUIRE(active.get().success);
    const auto cancelled = queued.get();
    HF_REQUIRE(!cancelled.success);
    HF_REQUIRE(cancelled.message == "worker-request-cancelled");
    HF_REQUIRE(worker.snapshot().state == hexagonflow::WorkerState::Stopped);
}

HF_TEST(persistent_worker_surfaces_backend_startup_failure) {
    hexagonflow::PersistentWorker worker(options(), []() -> std::unique_ptr<hexagonflow::InferenceBackend> {
        throw std::runtime_error("fixture startup failure");
    });
    HF_REQUIRE(!worker.wait_until_ready(std::chrono::seconds(2)));
    const auto snapshot = worker.snapshot();
    HF_REQUIRE(snapshot.state == hexagonflow::WorkerState::Faulted);
    HF_REQUIRE(snapshot.startup_error == "backend-initialization-failed");
    worker.stop();
}
