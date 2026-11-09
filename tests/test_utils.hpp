// test_utils.hpp - minimal dependency-free test harness.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testing_util {

struct TestFailure {
    std::string message;
};

inline int& failure_count() {
    static int n = 0;
    return n;
}

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::ostringstream _os;                                              \
            _os << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed";   \
            throw ::testing_util::TestFailure{_os.str()};                        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        const double _a = (a), _b = (b), _t = (tol);                             \
        if (!(std::fabs(_a - _b) <= _t)) {                                       \
            std::ostringstream _os;                                              \
            _os << __FILE__ << ":" << __LINE__ << ": |" #a " - " #b "| = "       \
                << std::fabs(_a - _b) << " > " << _t << " (" << _a << " vs "     \
                << _b << ")";                                                    \
            throw ::testing_util::TestFailure{_os.str()};                        \
        }                                                                        \
    } while (0)

// Runs one named test, prints PASS/FAIL, records failures. `print` lets MPI
// tests print only on rank 0.
inline void run_test(const std::string& name, const std::function<void()>& fn,
                     bool print = true) {
    try {
        fn();
        if (print) std::cout << "[PASS] " << name << "\n";
    } catch (const TestFailure& f) {
        ++failure_count();
        std::cout << "[FAIL] " << name << ": " << f.message << "\n";
    } catch (const std::exception& e) {
        ++failure_count();
        std::cout << "[FAIL] " << name << ": exception: " << e.what() << "\n";
    }
}

inline int finish(const std::string& suite, bool print = true) {
    if (print) {
        std::cout << suite << ": " << (failure_count() == 0 ? "ALL TESTS PASSED" : "FAILURES")
                  << " (" << failure_count() << " failed)\n";
    }
    return failure_count() == 0 ? 0 : 1;
}

inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace testing_util
