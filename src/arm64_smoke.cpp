// Minimal Linux/ARM64 ABI smoke test. It does not load QNN or claim hardware
// acceleration. The binary must be built for the target's observed ABI before
// it is executed on a device.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

int main() {
#if defined(__aarch64__)
    constexpr const char* compile_arch = "aarch64";
#elif defined(__x86_64__)
    constexpr const char* compile_arch = "x86_64";
#else
    constexpr const char* compile_arch = "other";
#endif

    std::printf("hexagonflow_smoke=true\n");
    std::printf("compile_arch=%s\n", compile_arch);
    std::printf("compile_pointer_bits=%zu\n", sizeof(void*) * 8U);

    struct utsname system {};
    if (uname(&system) == 0) {
        std::printf("kernel_sysname=%s\n", system.sysname);
        std::printf("kernel_release=%s\n", system.release);
        std::printf("runtime_machine=%s\n", system.machine);
    } else {
        std::printf("uname_error=%s\n", std::strerror(errno));
        return 2;
    }

    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    std::printf("online_cpus=%ld\n", online_cpus);
    struct statvfs filesystem {};
    if (statvfs("/", &filesystem) == 0) {
        const unsigned long long available =
            static_cast<unsigned long long>(filesystem.f_bavail) *
            static_cast<unsigned long long>(filesystem.f_frsize);
        std::printf("root_available_bytes=%llu\n", available);
    } else {
        std::printf("root_statvfs_error=%s\n", std::strerror(errno));
    }
    std::printf("smoke_scope=abi_and_basic_syscalls_only\n");
    return 0;
}
