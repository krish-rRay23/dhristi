#include "drishti/profiling/profiling.h"

namespace drishti::profiling {

ScopedTimer::ScopedTimer(std::string label, std::vector<TimerSample>* out)
    : label_(std::move(label)), out_(out), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    const auto dur = std::chrono::steady_clock::now() - start_;
    if (out_) {
        out_->push_back(TimerSample{label_, dur});
    }
}

}
