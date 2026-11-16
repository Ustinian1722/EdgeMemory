#pragma once

#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hexagonflow_test {

using TestCase = std::pair<std::string, std::function<void()>>;

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

class Registrar {
public:
    Registrar(std::string name, std::function<void()> fn) {
        registry().emplace_back(std::move(name), std::move(fn));
    }
};

inline void require(bool condition,
                    const char* expression,
                    const char* file,
                    int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" +
                                 std::to_string(line) + ": " + expression);
    }
}

}  // namespace hexagonflow_test

#define HF_TEST(name)                                                        \
    static void name();                                                      \
    static const hexagonflow_test::Registrar name##_registrar(#name, name); \
    static void name()

#define HF_REQUIRE(expression)                                                \
    hexagonflow_test::require(static_cast<bool>(expression), #expression,    \
                              __FILE__, __LINE__)
