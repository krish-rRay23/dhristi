#ifndef DRISHTI_DIAGNOSIS_DIAGNOSIS_H
#define DRISHTI_DIAGNOSIS_DIAGNOSIS_H

#include <string>
#include <vector>

namespace drishti::diagnosis {

enum class Severity { Note, Warning, Error };

struct Diagnostic {
    Severity severity = Severity::Note;
    std::string message;
    std::string location;
};

class DiagnosticEngine {
public:
    void emit(Diagnostic d) { diag_.push_back(std::move(d)); }
    [[nodiscard]] const std::vector<Diagnostic>& all() const noexcept { return diag_; }
    [[nodiscard]] bool has_errors() const noexcept;
    void clear() noexcept { diag_.clear(); }

private:
    std::vector<Diagnostic> diag_;
};

}  // namespace drishti::diagnosis

#endif
