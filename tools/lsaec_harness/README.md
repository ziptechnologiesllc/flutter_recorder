# LSAEC offline harness

Drives the real `SynchronousEchoTemplate` on the host under UBSan/
float-divide-by-zero with a chaos scenario (silent-ref stretches, seed IR,
notify storms, period changes, ref discontinuities, learn-freeze spans).

Built in one line, no build system:

    clang++ -std=c++17 -g -O1 \
      -fsanitize=undefined,float-divide-by-zero -fno-sanitize-recover=all \
      -I ../../src -I ../../src/filters/aec -I ../../src/fft \
      harness.cpp \
      ../../src/filters/aec/synchronous_echo_template.cpp \
      ../../src/filters/aec/spectral_governor.cpp \
      ../../src/fft/soloud_fft.cpp -o hunt && ./hunt

History: proved the template IEEE-clean through 400 chaos passes during the
2026-08-22 Android "brick wall" hunt — which redirected the search to the
wrapper's inputs and cracked the real bug (the capture.cpp format-tag
mismatch feeding the filter chain f32 data labeled s16).
