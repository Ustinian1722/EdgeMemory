cmake_minimum_required(VERSION 3.22)

project(HexagonFlow VERSION 0.1.0 LANGUAGES CXX)

option(HEXAGONFLOW_BUILD_TESTS "Build HexagonFlow unit tests" ON)
option(HEXAGONFLOW_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(HEXAGONFLOW_ENABLE_GENIEX_BACKEND "Build the optional adapter against an external GenieX C SDK" OFF)
set(HEXAGONFLOW_GENIEX_ROOT "" CACHE PATH "External GenieX SDK/runtime root; never vendored into this repository")
set(HEXAGONFLOW_GENIEX_INCLUDE_DIR "" CACHE PATH "External GenieX SDK include directory")

add_library(hexagonflow_core
    src/backend.cpp
    src/interference.cpp
    src/interference_io.cpp
    src/metrics.cpp
    src/persistent_worker.cpp
    src/process_backend.cpp
    src/resource_manager.cpp
    src/scheduler.cpp
    src/types.cpp
)
target_include_directories(hexagonflow_core PUBLIC include)
target_compile_features(hexagonflow_core PUBLIC cxx_std_20)
set_target_properties(hexagonflow_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(hexagonflow_core PRIVATE /W4 /permissive-)
    if(HEXAGONFLOW_ENABLE_WARNINGS_AS_ERRORS)
        target_compile_options(hexagonflow_core PRIVATE /WX)
    endif()
else()
    target_compile_options(hexagonflow_core PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
    if(HEXAGONFLOW_ENABLE_WARNINGS_AS_ERRORS)
        target_compile_options(hexagonflow_core PRIVATE -Werror)
    endif()
endif()

add_executable(hexagonflow src/main.cpp)
target_link_libraries(hexagonflow PRIVATE hexagonflow_core)

# Stable C ABI used by the EdgeMemory Python orchestrator. This keeps task
# admission and resource accounting in the C++ scheduler without duplicating
# its policy in Python.
add_library(hexagonflow_scheduler_c SHARED src/scheduler_c_api.cpp)
target_link_libraries(hexagonflow_scheduler_c PRIVATE hexagonflow_core)
target_include_directories(hexagonflow_scheduler_c PUBLIC include)
target_compile_features(hexagonflow_scheduler_c PUBLIC cxx_std_20)

add_executable(hexagonflow-cpu-run src/llama_schedule.cpp)
target_link_libraries(hexagonflow-cpu-run PRIVATE hexagonflow_core)

add_executable(hexagonflow-geniex-run src/geniex_schedule.cpp)
target_link_libraries(hexagonflow-geniex-run PRIVATE hexagonflow_core)

add_executable(hexagonflow-route-geniex src/route_geniex.cpp)
target_link_libraries(hexagonflow-route-geniex PRIVATE hexagonflow_core)

add_executable(hexagonflow-mixed-run src/mixed_schedule.cpp)
target_link_libraries(hexagonflow-mixed-run PRIVATE hexagonflow_core)

add_executable(hexagonflow-dual-geniex-run src/dual_geniex_schedule.cpp)
target_link_libraries(hexagonflow-dual-geniex-run PRIVATE hexagonflow_core)

add_executable(hexagonflow-interference-profile-check
    examples/interference_profile_import_smoke.cpp)
target_link_libraries(hexagonflow-interference-profile-check PRIVATE hexagonflow_core)

if(HEXAGONFLOW_ENABLE_GENIEX_BACKEND)
    if(NOT HEXAGONFLOW_GENIEX_ROOT)
        message(FATAL_ERROR "Set HEXAGONFLOW_GENIEX_ROOT to a target-matched external GenieX SDK root")
    endif()
    if(NOT HEXAGONFLOW_GENIEX_INCLUDE_DIR)
        find_path(HEXAGONFLOW_GENIEX_INCLUDE_DIR geniex.h
            PATHS "${HEXAGONFLOW_GENIEX_ROOT}/sdk/include" "${HEXAGONFLOW_GENIEX_ROOT}/include"
            NO_DEFAULT_PATH REQUIRED)
    endif()
    add_library(hexagonflow_geniex_backend
        src/geniex_vlm_backend.cpp
        src/geniex_llm_backend.cpp)
    target_include_directories(hexagonflow_geniex_backend PUBLIC include PRIVATE "${HEXAGONFLOW_GENIEX_INCLUDE_DIR}")
    target_compile_features(hexagonflow_geniex_backend PUBLIC cxx_std_20)
    target_link_libraries(hexagonflow_geniex_backend PUBLIC hexagonflow_core PRIVATE ${CMAKE_DL_LIBS})
    add_executable(hexagonflow-geniex-vlm-smoke examples/geniex_vlm_smoke.cpp)
    target_link_libraries(hexagonflow-geniex-vlm-smoke PRIVATE hexagonflow_geniex_backend)
    add_executable(hexagonflow-geniex-vlm-dual-worker-smoke examples/geniex_vlm_dual_worker_smoke.cpp)
    target_link_libraries(hexagonflow-geniex-vlm-dual-worker-smoke PRIVATE hexagonflow_geniex_backend)
    add_executable(hexagonflow-geniex-vlm-service examples/geniex_vlm_service.cpp)
    target_link_libraries(hexagonflow-geniex-vlm-service PRIVATE hexagonflow_geniex_backend)
    add_executable(hexagonflow-geniex-llm-service examples/geniex_llm_service.cpp)
    target_link_libraries(hexagonflow-geniex-llm-service PRIVATE hexagonflow_geniex_backend)
    add_executable(hexagonflow-geniex-vlm-scheduled-smoke examples/geniex_vlm_scheduled_smoke.cpp)
    target_link_libraries(hexagonflow-geniex-vlm-scheduled-smoke PRIVATE hexagonflow_geniex_backend)
endif()

if(HEXAGONFLOW_BUILD_TESTS)
    enable_testing()
    add_executable(hexagonflow_unit_tests
        tests/test_main.cpp
        tests/test_backend.cpp
        tests/test_interference.cpp
        tests/test_interference_io.cpp
        tests/test_process_backend.cpp
        tests/test_persistent_worker.cpp
        tests/test_resource_manager.cpp
        tests/test_scheduler.cpp
    )
    target_link_libraries(hexagonflow_unit_tests PRIVATE hexagonflow_core)
    add_test(NAME hexagonflow_unit_tests COMMAND hexagonflow_unit_tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME hexagonflow_python_tests
            COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p "test_*.py"
        )
        set_tests_properties(hexagonflow_python_tests PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()
endif()
