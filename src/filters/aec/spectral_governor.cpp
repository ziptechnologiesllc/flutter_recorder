#include "spectral_governor.h"

#include "../../fft/soloud_fft.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
constexpr int kFftSize = 1024;      // FFT::fft1024 — 512 packed complex bins
constexpr int kHop = 1024;          // non-overlapping windows (Welch handles it)
constexpr float kWelchAlpha = 1.0f / 16.0f; // ~0.35 s memory @ 21 ms windows

// Log-spaced band edges over bins [2, 320] (~94 Hz .. 15 kHz @ 48 kHz).
constexpr int kBandLoBin = 2;
constexpr int kBandHiBin = 320;

// Far-end per-band activity floor: bands where the loop isn't playing carry
// no echo information and must not vote.
constexpr double kRefBandFloor = 1e-9;

// Controller: bounded authority, slow relative to template convergence.
constexpr float kBoostMax = 5.0f;
constexpr float kLeakEngage = 0.30f;  // above: echo leaking, raise gain
constexpr float kLeakRelease = 0.12f; // below: converged, anneal back down
constexpr float kUpPerSec = 0.9f;     // slew limits
constexpr float kDownPerSec = 0.45f;
constexpr float kControllerDt = 0.21f; // update cadence (10 windows)

// Minimum windows before the controller trusts the spectra.
constexpr int kWarmupWindows = 16;
} // namespace

SpectralGovernor &SpectralGovernor::instance() {
  static SpectralGovernor g;
  return g;
}

SpectralGovernor::SpectralGovernor() {
  mRing = new float[kRingFrames * 2];
  memset(mRing, 0, sizeof(float) * kRingFrames * 2);
}

SpectralGovernor::~SpectralGovernor() {
  stop();
  delete[] mRing;
}

void SpectralGovernor::start() {
  bool expected = false;
  if (!mRunning.compare_exchange_strong(expected, true))
    return; // already running
  mWorker = std::thread([this] { workerLoop(); });
}

void SpectralGovernor::stop() {
  if (!mRunning.exchange(false))
    return;
  if (mWorker.joinable())
    mWorker.join();
  mBoost.store(1.0f, std::memory_order_relaxed);
}

void SpectralGovernor::push(const float *ref, const float *res,
                            unsigned int frameCount, unsigned int channels) {
  if (!mRunning.load(std::memory_order_relaxed) || !ref || !res ||
      channels == 0)
    return;
  const uint64_t w = mWriteIdx.load(std::memory_order_relaxed);
  const uint64_t r = mReadIdx.load(std::memory_order_acquire);
  if (w - r + frameCount > kRingFrames)
    return; // ring full: drop (worker is behind; never block the RT thread)
  const float inv = 1.0f / static_cast<float>(channels);
  for (unsigned int f = 0; f < frameCount; ++f) {
    float rm = 0.0f, sm = 0.0f;
    for (unsigned int ch = 0; ch < channels; ++ch) {
      rm += ref[f * channels + ch];
      sm += res[f * channels + ch];
    }
    const size_t slot = static_cast<size_t>((w + f) % kRingFrames) * 2;
    mRing[slot] = rm * inv;
    mRing[slot + 1] = sm * inv;
  }
  mWriteIdx.store(w + frameCount, std::memory_order_release);
}

void SpectralGovernor::workerLoop() {
  float refW[kFftSize];
  float resW[kFftSize];
  float hann[kFftSize];
  for (int i = 0; i < kFftSize; ++i)
    hann[i] = 0.5f * (1.0f - cosf(2.0f * static_cast<float>(M_PI) * i /
                                  (kFftSize - 1)));
  int windowsSinceControl = 0;
  int windowsSinceLog = 0;

  while (mRunning.load(std::memory_order_relaxed)) {
    const uint64_t w = mWriteIdx.load(std::memory_order_acquire);
    const uint64_t r = mReadIdx.load(std::memory_order_relaxed);
    if (w - r < static_cast<uint64_t>(kHop)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
      continue;
    }
    for (int i = 0; i < kFftSize; ++i) {
      const size_t slot = static_cast<size_t>((r + i) % kRingFrames) * 2;
      refW[i] = mRing[slot] * hann[i];
      resW[i] = mRing[slot + 1] * hann[i];
    }
    mReadIdx.store(r + kHop, std::memory_order_release);

    processWindow(refW, resW);

    if (++windowsSinceControl >= 10) { // ~0.21 s
      windowsSinceControl = 0;
      controllerUpdate();
    }
    if (++windowsSinceLog >= 96) { // ~2 s — worker thread, NOT the RT thread
      windowsSinceLog = 0;
      fprintf(stderr, "[GOV] leak=%.2f boost=%.2f\n",
              mLeak.load(std::memory_order_relaxed),
              mBoost.load(std::memory_order_relaxed));
    }
  }
}

void SpectralGovernor::processWindow(const float *ref, const float *res) {
  float fr[kFftSize];
  float fs[kFftSize];
  memcpy(fr, ref, sizeof(fr));
  memcpy(fs, res, sizeof(fs));
  FFT::fft1024(fr); // packed complex: bin i -> (fr[2i], fr[2i+1]), i < 512
  FFT::fft1024(fs);

  // Accumulate per log-band Welch EMAs.
  for (int b = 0; b < kBands; ++b) {
    const double t0 = static_cast<double>(b) / kBands;
    const double t1 = static_cast<double>(b + 1) / kBands;
    const int lo = static_cast<int>(kBandLoBin *
                                    pow(double(kBandHiBin) / kBandLoBin, t0));
    const int hi = static_cast<int>(kBandLoBin *
                                    pow(double(kBandHiBin) / kBandLoBin, t1));
    double sxx = 0, syy = 0, sre = 0, sim = 0;
    for (int i = lo; i < hi && i < kFftSize / 2; ++i) {
      const double xr = fr[2 * i], xi = fr[2 * i + 1];
      const double yr = fs[2 * i], yi = fs[2 * i + 1];
      sxx += xr * xr + xi * xi;
      syy += yr * yr + yi * yi;
      // Sxy = Y * conj(X)
      sre += yr * xr + yi * xi;
      sim += yi * xr - yr * xi;
    }
    mSxx[b] += kWelchAlpha * (sxx - mSxx[b]);
    mSyy[b] += kWelchAlpha * (syy - mSyy[b]);
    mSxyRe[b] += kWelchAlpha * (sre - mSxyRe[b]);
    mSxyIm[b] += kWelchAlpha * (sim - mSxyIm[b]);
  }
  ++mWindowsSeen;
}

void SpectralGovernor::controllerUpdate() {
  if (mWindowsSeen < kWarmupWindows)
    return;

  // Leakage = residual-energy-weighted coherence over far-end-active bands.
  // Coherence gamma^2 = |Sxy|^2 / (Sxx*Syy): the fraction of residual energy
  // in a band that is linearly explained by the reference — i.e. echo the
  // template failed to remove. Performer/room noise is incoherent and scores
  // ~0, so the controller cannot be tricked into chasing the musician.
  double wsum = 0.0, leak = 0.0;
  for (int b = 0; b < kBands; ++b) {
    if (mSxx[b] <= kRefBandFloor)
      continue; // loop not playing in this band: no vote
    const double denom = mSxx[b] * mSyy[b] + 1e-20;
    const double coh =
        (mSxyRe[b] * mSxyRe[b] + mSxyIm[b] * mSxyIm[b]) / denom;
    const double wgt = mSyy[b];
    leak += wgt * (coh > 1.0 ? 1.0 : coh);
    wsum += wgt;
  }
  const float leakF = wsum > 0 ? static_cast<float>(leak / wsum) : 0.0f;
  mLeak.store(leakF, std::memory_order_relaxed);

  float boost = mBoost.load(std::memory_order_relaxed);
  if (leakF > kLeakEngage) {
    boost += kUpPerSec * kControllerDt;
  } else if (leakF < kLeakRelease) {
    boost -= kDownPerSec * kControllerDt;
  }
  if (boost < 1.0f)
    boost = 1.0f;
  if (boost > kBoostMax)
    boost = kBoostMax;
  mBoost.store(boost, std::memory_order_relaxed);
}
