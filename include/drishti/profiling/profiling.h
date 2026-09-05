#ifndef DRISHTI_PROFILING_PROFILING_H
#define DRISHTI_PROFILING_PROFILING_H

#include <chrono>
#include <string>
#include <vector>

namespace drishti::profiling {

struct TimerSample {
    std::string label;
    std::chrono::nanoseconds duration{};
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string label, std::vector<TimerSample>* out = nullptr);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string label_;
    std::vector<TimerSample>* out_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace drishti::profiling

#endif
