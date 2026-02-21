// Stub for missing TFLite profiling symbol in prebuilt libLiteRt.a.
// MaybeCreateSignpostProfiler is referenced but undefined — returning
// nullptr disables signpost profiling (no functional impact).
namespace tflite {
namespace profiling {
class Profiler;
Profiler* MaybeCreateSignpostProfiler() { return nullptr; }
}
}
