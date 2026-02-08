// Stub for MaybeCreateSignpostProfiler - the real implementation uses Apple's
// os_signpost API but isn't included in the LiteRT static archive.
#include <memory>

namespace tflite {
class Profiler;
namespace profiling {
std::unique_ptr<tflite::Profiler> MaybeCreateSignpostProfiler() {
  return nullptr;
}
}  // namespace profiling
}  // namespace tflite