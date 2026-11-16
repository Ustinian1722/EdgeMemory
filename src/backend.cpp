#include "hexagonflow/backend.hpp"

#include <utility>

namespace hexagonflow {

MockBackend::MockBackend(std::string backend_id)
    : backend_id_(std::move(backend_id)) {}

std::string MockBackend::id() const {
    return backend_id_;
}

BackendFamily MockBackend::family() const {
    return BackendFamily::Mock;
}

BackendResult MockBackend::execute(const Workload& workload) {
    if (!failing_model_id_.empty() && workload.model_id == failing_model_id_) {
        return {false, true, 0, "mock failure injected for lifecycle test"};
    }
    return {true, true, workload.requested_output_tokens,
            "mock completion; not an inference result"};
}

void MockBackend::fail_model(std::string model_id) {
    failing_model_id_ = std::move(model_id);
}

void MockBackend::clear_failure() {
    failing_model_id_.clear();
}

}  // namespace hexagonflow
