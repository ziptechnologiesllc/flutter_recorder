// Offline LSAEC NaN hunt: drive the REAL SynchronousEchoTemplate with a
// realistic loop (music-ish + silent stretches + performer double-talk +
// period changes) under UBSan/float-divide-by-zero. First bad float op traps
// with file:line.
#include "synchronous_echo_template.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

void aecLog(const char *fmt, ...) { (void)fmt; } // stub

static bool scanFinite(const float *p, size_t n, const char *what, int pass) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(p[i])) {
      printf("!! NON-FINITE in %s at idx %zu, pass %d\n", what, i, pass);
      return false;
    }
  return true;
}

int main() {
  const unsigned sr = 48000, ch = 2, block = 480;
  const int64_t P = 139200; // 2.9s like the TB330FU loops
  SynchronousEchoTemplate tpl(sr, ch);

  // Loop content: chords + SILENT gaps (silent-ref phases suspected trigger)
  std::vector<float> loop(P * ch);
  for (int64_t i = 0; i < P; ++i) {
    double t = (double)i / sr;
    float v = 0.0f;
    bool silent = fmod(t, 1.0) > 0.72; // ~28% of each second silent
    if (!silent)
      v = 0.28f * (sinf(2 * M_PI * 220 * t) + 0.5f * sinf(2 * M_PI * 330 * t) +
                   0.3f * sinf(2 * M_PI * 440.5f * t));
    loop[i * ch] = v;
    loop[i * ch + 1] = v;
  }

  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0f, 0.004f);

  std::vector<float> mic(block * ch), ref(block * ch);
  const int64_t echoDelay = 3600; // 75ms echo path
  int64_t frame = 0;
  const int passes = 400;
  const int64_t totalFrames = (int64_t)passes * P;

  tpl.setLearnBoost(1.0f);

  // Calibrated IR like the device applies post-calibration (decaying, with
  // the direct path near tap 32 and chassis ringing later — TB330FU anatomy)
  {
    std::vector<float> ir(8192, 0.0f);
    for (int i = 0; i < 8192; ++i) {
      float env = expf(-i / 900.0f);
      ir[i] = env * (i == 32 ? 0.06f
                             : 0.02f * sinf(2 * M_PI * 1100.0f * i / sr));
    }
    tpl.setSeedImpulseResponse(ir.data(), ir.size());
  }

  int pass = 0;
  for (; frame < totalFrames; frame += block) {
    pass = (int)(frame / P);
    // On-device chaos, ingredient by ingredient:
    // (1) notify STORM — the duplex-ring xrun watch fired ~28 seed arms
    if (frame % (P / 3) < block) tpl.notifyReferenceChanged();

    for (unsigned i = 0; i < block; ++i) {
      int64_t gf = frame + i;
      int64_t phi = gf % P;
      float r0 = loop[phi * ch], r1 = loop[phi * ch + 1];
      ref[i * ch] = r0;
      ref[i * ch + 1] = r1;
      int64_t ephi = (gf - echoDelay) % P;
      if (ephi < 0) ephi += P;
      float echo = 0.35f * loop[ephi * ch];
      // performer bursts (double-talk) 10% of the time
      double t = (double)gf / sr;
      float performer =
          (fmod(t, 7.0) < 0.7) ? 0.4f * sinf(2 * M_PI * 660 * t) : 0.0f;
      float m = echo + performer + noise(rng);
      mic[i * ch] = m;
      mic[i * ch + 1] = m;
    }
    // (2) period changes: overdub doubles the composite period at pass 25,
    //     a fresh shorter base replaces it at pass 60 (non-multiple change)
    int64_t Pnow = P;
    if (pass >= 25 && pass < 60) Pnow = 2 * P;
    else if (pass >= 60) Pnow = (P * 2) / 3;
    // (3) reference discontinuity ~ ring shed: jump the ref phase briefly
    bool shed = (frame % (2 * P)) < block && pass > 10;
    // (4) learn-freeze spans like DTD holds
    bool learnNow = (pass % 9) != 5;
    tpl.process(mic.data(), ref.data(), block, ch,
                frame + (shed ? 720 : 0), Pnow, 0, learnNow);
    if (!scanFinite(mic.data(), block * ch, "OUTPUT", pass)) return 1;
    if ((frame / block) % 1000 == 0)
      printf("pass %d ok (frame %lld)\n", pass, (long long)frame);
  }
  printf("COMPLETED %d passes, no non-finite output. Confidence=%.2f\n",
         passes, tpl.meanConfidence());
  return 0;
}
