#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

namespace {
struct ClipSession {
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* output = nullptr;
    int backend_type = -1;
    std::string opencl_device_name;
};

std::string find_opencl_gpu_name() {
    void* library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) return {};
    using GetPlatforms = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
    using GetDevices = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    using GetDeviceInfo = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    auto get_platforms = reinterpret_cast<GetPlatforms>(dlsym(library, "clGetPlatformIDs"));
    auto get_devices = reinterpret_cast<GetDevices>(dlsym(library, "clGetDeviceIDs"));
    auto get_info = reinterpret_cast<GetDeviceInfo>(dlsym(library, "clGetDeviceInfo"));
    std::string result;
    if (get_platforms && get_devices && get_info) {
        cl_uint platform_count = 0;
        if (get_platforms(0, nullptr, &platform_count) == CL_SUCCESS && platform_count > 0) {
            std::vector<cl_platform_id> platforms(platform_count);
            if (get_platforms(platform_count, platforms.data(), nullptr) == CL_SUCCESS) {
                for (auto platform : platforms) {
                    cl_uint device_count = 0;
                    if (get_devices(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) != CL_SUCCESS ||
                        device_count == 0) continue;
                    std::vector<cl_device_id> devices(device_count);
                    if (get_devices(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr) != CL_SUCCESS)
                        continue;
                    std::vector<char> name(512, '\0');
                    if (get_info(devices.front(), CL_DEVICE_NAME, name.size(), name.data(), nullptr) == CL_SUCCESS) {
                        result = name.data();
                        break;
                    }
                }
            }
        }
    }
    dlclose(library);
    return result;
}

void set_error(char* destination, size_t capacity, const char* message) {
    if (!destination || capacity == 0) return;
    const auto count = std::min(capacity - 1, std::strlen(message));
    std::memcpy(destination, message, count);
    destination[count] = '\0';
}
}  // namespace

extern "C" {
void* hexagonflow_mnn_clip_opencl_create(const char* model_path,
                                          char* error, size_t error_capacity) {
    try {
        if (!model_path || !*model_path) {
            set_error(error, error_capacity, "empty_model_path");
            return nullptr;
        }
        auto* raw = MNN::Interpreter::createFromFile(model_path);
        if (!raw) {
            set_error(error, error_capacity, "mnn_model_load_failed");
            return nullptr;
        }
        auto state = std::make_unique<ClipSession>();
        state->interpreter = std::shared_ptr<MNN::Interpreter>(raw, MNN::Interpreter::destroy);
        state->interpreter->setSessionMode(MNN::Interpreter::Session_Backend_Fix);
        MNN::ScheduleConfig schedule;
        schedule.type = MNN_FORWARD_OPENCL;
        schedule.numThread = 2;
        MNN::BackendConfig backend;
        schedule.backendConfig = &backend;
        state->session = state->interpreter->createSession(schedule);
        if (!state->session) {
            set_error(error, error_capacity, "opencl_session_create_failed");
            return nullptr;
        }
        int backends[2] = {-1, -1};
        state->interpreter->getSessionInfo(state->session, MNN::Interpreter::BACKENDS, backends);
        state->backend_type = backends[0];
        if (state->backend_type != MNN_FORWARD_OPENCL) {
            set_error(error, error_capacity, "opencl_backend_not_active");
            return nullptr;
        }
        state->opencl_device_name = find_opencl_gpu_name();
        state->input = state->interpreter->getSessionInput(state->session, nullptr);
        state->output = state->interpreter->getSessionOutput(state->session, nullptr);
        if (!state->input || !state->output ||
            state->input->getType().code != halide_type_float ||
            state->output->getType().code != halide_type_float) {
            set_error(error, error_capacity, "clip_model_tensor_contract_unsupported");
            return nullptr;
        }
        set_error(error, error_capacity, "");
        return state.release();
    } catch (...) {
        set_error(error, error_capacity, "mnn_opencl_initialization_exception");
        return nullptr;
    }
}

int hexagonflow_mnn_clip_opencl_infer(void* opaque, const float* nchw,
                                      size_t input_count, float* output,
                                      size_t output_capacity, size_t* output_count,
                                      char* error, size_t error_capacity) {
    try {
        auto* state = static_cast<ClipSession*>(opaque);
        if (!state || !nchw || !output || !output_count) {
            set_error(error, error_capacity, "invalid_inference_arguments");
            return -1;
        }
        const auto shape = state->input->shape();
        const auto expected = state->input->elementSize();
        if (input_count != expected || state->input->dimensions() != 4 ||
            state->input->batch() != 1 || state->input->channel() != 3 ||
            input_count != 3u * 224u * 224u) {
            set_error(error, error_capacity, "clip_input_shape_mismatch");
            return -2;
        }
        MNN::Tensor host_input(state->input, state->input->getDimensionType());
        auto* destination = host_input.host<float>();
        if (!destination) {
            set_error(error, error_capacity, "clip_host_input_allocation_failed");
            return -3;
        }
        if (host_input.getDimensionType() == MNN::Tensor::CAFFE) {
            std::copy(nchw, nchw + input_count, destination);
        } else if (host_input.getDimensionType() == MNN::Tensor::TENSORFLOW) {
            // The bridge accepts the product's normalized NCHW tensor and
            // converts only if the exported MNN graph expects NHWC.
            constexpr size_t plane = 224u * 224u;
            for (size_t pixel = 0; pixel < plane; ++pixel) {
                destination[pixel * 3] = nchw[pixel];
                destination[pixel * 3 + 1] = nchw[plane + pixel];
                destination[pixel * 3 + 2] = nchw[2 * plane + pixel];
            }
        } else {
            set_error(error, error_capacity, "clip_input_layout_unsupported");
            return -4;
        }
        if (!state->input->copyFromHostTensor(&host_input)) {
            set_error(error, error_capacity, "opencl_input_transfer_failed");
            return -5;
        }
        if (state->interpreter->runSession(state->session) != MNN::NO_ERROR) {
            set_error(error, error_capacity, "opencl_inference_failed");
            return -6;
        }
        MNN::Tensor host_output(state->output, state->output->getDimensionType());
        if (!state->output->copyToHostTensor(&host_output)) {
            set_error(error, error_capacity, "opencl_output_transfer_failed");
            return -7;
        }
        const auto count = host_output.elementSize();
        if (count == 0 || count > output_capacity) {
            set_error(error, error_capacity, "clip_output_size_mismatch");
            return -8;
        }
        std::copy(host_output.host<float>(), host_output.host<float>() + count, output);
        *output_count = count;
        set_error(error, error_capacity, "");
        return state->backend_type;
    } catch (...) {
        set_error(error, error_capacity, "mnn_opencl_inference_exception");
        return -9;
    }
}

int hexagonflow_mnn_clip_opencl_backend(void* opaque) {
    const auto* state = static_cast<const ClipSession*>(opaque);
    return state ? state->backend_type : -1;
}

int hexagonflow_mnn_clip_opencl_device_name(void* opaque, char* destination, size_t capacity) {
    const auto* state = static_cast<const ClipSession*>(opaque);
    if (!state || !destination || capacity == 0) return -1;
    const auto& name = state->opencl_device_name;
    const auto count = std::min(capacity - 1, name.size());
    std::memcpy(destination, name.data(), count);
    destination[count] = '\0';
    return name.empty() ? 0 : static_cast<int>(count);
}

void hexagonflow_mnn_clip_opencl_destroy(void* opaque) {
    delete static_cast<ClipSession*>(opaque);
}
}  // extern "C"
