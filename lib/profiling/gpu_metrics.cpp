#include "drishti/profiling/gpu_metrics.h"

#include <iomanip>
#include <sstream>

namespace drishti::profiling {
namespace {

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

std::string dbl(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << v;
    return oss.str();
}

}  // namespace

std::string gpu_metrics_to_json(const GpuProfileMetrics& m) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"schema\": \"drishti.gpu_profile/v1\",\n"
        << "  \"ok\": " << (m.ok ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(m.error) << "\",\n"
        << "  \"backend\": \"" << json_escape(m.device.backend) << "\",\n"
        << "  \"device\": {\n"
        << "    \"present\": " << (m.device.present ? "true" : "false") << ",\n"
        << "    \"name\": \"" << json_escape(m.device.name) << "\",\n"
        << "    \"compute_capability\": \"" << m.device.compute_major << "."
        << m.device.compute_minor << "\",\n"
        << "    \"sm_count\": " << m.device.sm_count << ",\n"
        << "    \"max_threads_per_block\": " << m.device.max_threads_per_block << ",\n"
        << "    \"clock_mhz\": " << m.device.clock_mhz << ",\n"
        << "    \"mem_clock_mhz\": " << m.device.mem_clock_mhz << ",\n"
        << "    \"mem_bus_width_bits\": " << m.device.mem_bus_width_bits << ",\n"
        << "    \"mem_theoretical_gbps\": " << dbl(m.device.mem_theoretical_gbps) << ",\n"
        << "    \"total_mem_bytes\": " << m.device.total_mem_bytes << ",\n"
        << "    \"driver_version\": \"" << json_escape(m.device.driver_version) << "\"\n"
        << "  },\n"
        << "  \"kernel\": \"" << json_escape(m.kernel) << "\",\n"
        << "  \"num_elements\": " << m.num_elements << ",\n"
        << "  \"block_size\": " << m.block_size << ",\n"
        << "  \"grid_size\": " << m.grid_size << ",\n"
        << "  \"repeats\": " << m.repeats << ",\n"
        << "  \"kernel_ms_avg\": " << dbl(m.kernel_ms_avg) << ",\n"
        << "  \"kernel_ms_min\": " << dbl(m.kernel_ms_min) << ",\n"
        << "  \"h2d_ms\": " << dbl(m.h2d_ms) << ",\n"
        << "  \"d2h_ms\": " << dbl(m.d2h_ms) << ",\n"
        << "  \"wall_ms\": " << dbl(m.wall_ms) << ",\n"
        << "  \"gbps_effective\": " << dbl(m.gbps_effective) << ",\n"
        << "  \"bytes_moved\": " << m.bytes_moved << ",\n"
        << "  \"correct\": " << (m.correct ? "true" : "false") << ",\n"
        << "  \"timing_source\": \"" << json_escape(m.timing_source) << "\",\n"
        << "  \"cupti_available\": " << (m.cupti_available ? "true" : "false") << ",\n"
        << "  \"cupti_version\": \"" << json_escape(m.cupti_version) << "\",\n"
        << "  \"notes\": \"" << json_escape(m.notes) << "\"\n"
        << "}";
    return oss.str();
}

std::string format_gpu_report(const GpuProfileMetrics& m) {
    std::ostringstream oss;
    oss << "Drishti GPU Profile Report\n"
        << "  Backend : " << m.device.backend << "\n"
        << "  Device  : " << (m.device.present ? m.device.name : "(none)") << "\n";
    if (m.device.present) {
        oss << "  CC/SMs  : sm_" << m.device.compute_major << m.device.compute_minor
            << ", " << m.device.sm_count << " SMs, " << m.device.clock_mhz << " MHz\n"
            << "  Memory  : " << (m.device.total_mem_bytes / (1024u * 1024u))
            << " MiB total";
        if (m.device.mem_theoretical_gbps > 0.0) {
            oss << ", ~" << dbl(m.device.mem_theoretical_gbps) << " GB/s theoretical";
        } else {
            oss << ", theoretical BW unknown";
        }
        oss << "\n  Driver  : " << m.device.driver_version << "\n";
    }
    if (!m.ok) {
        oss << "  Status  : FAILED: " << m.error << "\n";
        return oss.str();
    }
    oss << "  Kernel  : " << m.kernel << " (N=" << m.num_elements
        << ", block=" << m.block_size << ", grid=" << m.grid_size
        << ", repeats=" << m.repeats << ")\n"
        << "  Kernel  : avg " << dbl(m.kernel_ms_avg) << " ms, min "
        << dbl(m.kernel_ms_min) << " ms (" << m.timing_source << ")\n"
        << "  Copies  : HtoD " << dbl(m.h2d_ms) << " ms, DtoH " << dbl(m.d2h_ms)
        << " ms, wall " << dbl(m.wall_ms) << " ms (incl. setup)\n"
        << "  Tput    : " << dbl(m.gbps_effective) << " GB/s effective"
        << " (" << m.bytes_moved << " bytes moved)\n"
        << "  Verify  : " << (m.correct ? "PASS" : "FAIL") << "\n"
        << "  CUPTI   : "
        << (m.cupti_available ? m.cupti_version : "unavailable (" + m.cupti_version + ")")
        << "\n";
    if (!m.notes.empty()) oss << "  Notes   : " << m.notes << "\n";
    return oss.str();
}

drishti::provenance::PassInfo gpu_metrics_to_pass_info(const GpuProfileMetrics& m) {
    drishti::provenance::PassInfo p;
    p.name = "cuda-profile:" + (m.kernel.empty() ? "unknown" : m.kernel);
    std::ostringstream oss;
    oss << (m.ok ? "ok" : "failed") << " N=" << m.num_elements
        << " avg_ms=" << dbl(m.kernel_ms_avg) << " gbps=" << dbl(m.gbps_effective)
        << " correct=" << (m.correct ? "yes" : "no") << " on "
        << (m.device.name.empty() ? m.device.backend : m.device.name);
    p.description = oss.str();
    p.enabled_by_default = true;
    return p;
}

}  // namespace drishti::profiling
