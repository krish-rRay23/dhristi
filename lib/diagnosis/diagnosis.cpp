#include "drishti/diagnosis/diagnosis.h"

#include <algorithm>

namespace drishti::diagnosis {

bool DiagnosticEngine::has_errors() const noexcept {
    return std::any_of(diag_.begin(), diag_.end(),
                       [](const Diagnostic& d) { return d.severity == Severity::Error; });
}

}
