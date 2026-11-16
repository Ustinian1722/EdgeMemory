#include "hexagonflow/backend.hpp"
#include "test_framework.hpp"

HF_TEST(mock_backend_marks_results_as_simulated) {
    hexagonflow::MockBackend backend;
    hexagonflow::Workload workload;
    workload.id = "task-1";
    workload.model_id = "demo-model";
    workload.requested_output_tokens = 7;

    const auto result = backend.execute(workload);
    HF_REQUIRE(result.success);
    HF_REQUIRE(result.simulated);
    HF_REQUIRE(result.output_tokens == 7U);
}

HF_TEST(mock_backend_failure_is_explicit) {
    hexagonflow::MockBackend backend;
    backend.fail_model("broken-model");
    hexagonflow::Workload workload;
    workload.id = "task-1";
    workload.model_id = "broken-model";

    const auto result = backend.execute(workload);
    HF_REQUIRE(!result.success);
    HF_REQUIRE(result.simulated);
    HF_REQUIRE(!result.message.empty());
    backend.clear_failure();
    HF_REQUIRE(backend.execute(workload).success);
}
