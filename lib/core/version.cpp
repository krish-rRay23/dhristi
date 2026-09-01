#include "drishti/core/version.h"

#include <sstream>
#include <string>

namespace drishti::core {

std::string banner() {
    std::ostringstream oss;
    oss << "D\u1e5b\u1e63\u1e6di v" << DRISHTI_VERSION_STRING
        << " - ML Compiler and GPU Performance Analysis Framework";
    return oss.str();
}

}  // namespace drishti::core
