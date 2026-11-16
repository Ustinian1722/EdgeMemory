#include "test_framework.hpp"

#include <iostream>

int main() {
    int failures = 0;
    for (const auto& [name, test] : hexagonflow_test::registry()) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": unknown exception\n";
        }
    }
    std::cout << "tests=" << hexagonflow_test::registry().size()
              << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
