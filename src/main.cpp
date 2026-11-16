#include "hexagonflow/resource_manager.hpp"
#include "hexagonflow/scheduler.hpp"
#include "hexagonflow/types.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using hexagonflow::BackendFamily;
using hexagonflow::ResourceSpec;
using hexagonflow::ResourceManager;
using hexagonflow::Scheduler;
using hexagonflow::SchedulingPolicy;
using hexagonflow::Workload;
using hexagonflow::WorkloadKind;

void print_usage() {
    std::cout << "HexagonFlow local scheduler prototype\n"
              << "Usage:\n"
              << "  hexagonflow simulate [--policy fifo|priority|edf|static]"
                 " [--tasks N]\n"
              << "  hexagonflow policies\n"
              << "  hexagonflow version\n\n"
              << "Simulation output is synthetic and is not IQ-9075 "
                 "performance data.\n";
}

SchedulingPolicy parse_policy(const std::string& name) {
    if (name == "fifo") return SchedulingPolicy::Fifo;
    if (name == "priority") return SchedulingPolicy::Priority;
    if (name == "edf") return SchedulingPolicy::EarliestDeadlineFirst;
    if (name == "static") return SchedulingPolicy::StaticAllocation;
    throw std::invalid_argument("unknown policy: " + name);
}

int run_simulation(SchedulingPolicy policy, std::uint64_t task_count) {
    constexpr std::uint64_t memory_bytes = 512ULL * 1024ULL * 1024ULL;
    std::vector<ResourceSpec> specs{
        {"sim-resource-a", "mock", BackendFamily::Mock, 1U, memory_bytes},
        {"sim-resource-b", "mock", BackendFamily::Mock, 1U, memory_bytes},
    };
    ResourceManager resources(std::move(specs));
    const std::map<std::string, std::string> static_bindings{
        {"demo-llm", "sim-resource-a"},
        {"demo-asr", "sim-resource-b"},
    };
    Scheduler scheduler(policy, resources, static_bindings);

    for (std::uint64_t index = 0; index < task_count; ++index) {
        Workload workload;
        workload.id = "task-" + std::to_string(index);
        const bool llm = index % 2U == 0U;
        workload.model_id = llm ? "demo-llm" : "demo-asr";
        workload.kind = llm ? WorkloadKind::LlmDecode : WorkloadKind::Asr;
        workload.priority = static_cast<std::int32_t>(index % 3U);
        workload.absolute_deadline_us = 1000000ULL + (index * 10000ULL);
        workload.estimated_service_us = 50000ULL + (index * 1000ULL);
        workload.model_memory_bytes = 256ULL * 1024ULL * 1024ULL;
        workload.requested_output_tokens = llm ? 16ULL : 0ULL;
        workload.allowed_backend_ids = {"mock"};
        if (!scheduler.submit(std::move(workload))) {
            std::cerr << "failed to submit simulated workload\n";
            return 2;
        }
    }

    std::cout << "simulation_only=true policy=" << hexagonflow::to_string(policy)
              << " submitted=" << task_count << '\n';
    while (!scheduler.queued_order().empty()) {
        const auto decisions = scheduler.dispatch_ready();
        std::vector<std::string> dispatched;
        for (const auto& decision : decisions) {
            if (decision.dispatched) {
                std::cout << "dispatch " << decision.task_id << " -> "
                          << decision.resource_id << '\n';
                dispatched.push_back(decision.task_id);
            } else {
                std::cout << "queued " << decision.task_id << " reason="
                          << decision.reason << '\n';
            }
        }
        if (dispatched.empty()) {
            std::cerr << "no simulated resource could make progress\n";
            return 3;
        }
        for (const auto& task_id : dispatched) {
            if (!scheduler.complete(task_id)) {
                std::cerr << "failed to complete simulated task " << task_id
                          << '\n';
                return 4;
            }
        }
    }
    std::cout << "completed=" << scheduler.tasks().size()
              << " hardware_measurements=0\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" ||
            std::string(argv[1]) == "help") {
            print_usage();
            return 0;
        }
        const std::string command(argv[1]);
        if (command == "version") {
            std::cout << "HexagonFlow 0.1.0-dev\n";
            return 0;
        }
        if (command == "policies") {
            std::cout << "fifo\npriority\nedf\nstatic\n";
            return 0;
        }
        if (command != "simulate") {
            print_usage();
            return 1;
        }

        auto policy = SchedulingPolicy::Fifo;
        std::uint64_t task_count = 6;
        for (int index = 2; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--policy" && index + 1 < argc) {
                policy = parse_policy(argv[++index]);
            } else if (argument == "--tasks" && index + 1 < argc) {
                task_count = std::stoull(argv[++index]);
                if (task_count == 0U || task_count > 100000U) {
                    throw std::invalid_argument("--tasks must be in 1..100000");
                }
            } else {
                throw std::invalid_argument("unrecognized argument: " + argument);
            }
        }
        return run_simulation(policy, task_count);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
