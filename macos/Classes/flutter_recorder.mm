// Relative import to be able to reuse the C sources.
// See the comment in ../flutter_recorder.podspec for more information.
#include "../../src/flutter_recorder.cpp"

#include "../../src/analyzer.cpp"
#include "../../src/capture.cpp"
#include "../../src/fft/soloud_fft.cpp"
#include "../../src/filters/aec/adaptive_echo_cancellation.cpp"
#include "../../src/filters/aec/aec_test.cpp"
#include "../../src/filters/aec/calibration.cpp"
#include "../../src/filters/aec/delay_estimator.cpp"
#include "../../src/filters/aec/neural_post_filter.cpp"
#include "../../src/filters/aec/reference_buffer.cpp"
#include "../../src/filters/aec/vss_nlms_filter.cpp"
#include "../../src/filters/autogain.cpp"
#include "../../src/filters/echo_cancellation.cpp"
#include "../../src/filters/filters.cpp"
