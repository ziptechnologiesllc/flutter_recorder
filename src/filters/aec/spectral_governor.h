#ifndef AEC_SPECTRAL_GOVERNOR_H
#define AEC_SPECTRAL_GOVERNOR_H

#include <atomic>
#include <cstddef>
#include <thread>

/**
 * SpectralGovernor — closed-loop, frequency-domain tuning of LSAEC.
 *
 * The control idea (a self-tuning regulator):
 *   SENSOR    On a worker thread (never the render thread), compute windowed
 *             FFTs of the echo-aligned REFERENCE and the post-template
 *             RESIDUAL, Welch-averaged over ~0.7 s, and from them the per-band
 *             COHERENCE — "how correlated is what's left in the mic with what
 *             the speaker played?" High coherence in a band = echo leaking
 *             there = the template is under-converged at those frequencies.
 *             Low coherence = the residual is the performer or room noise.
 *   CONTROL   A slew-limited, clamped gain: measured leakage raises the
 *             template's learning rate (fast re-convergence after layer adds,
 *             path changes — no event plumbing needed); low leakage anneals it
 *             back to the deep, quiet floor.
 *   SAFETY    Coherence-gating is what the failed phase-tracker never had:
 *             the loop physically cannot chase the performer, because the
 *             performer is incoherent with the reference. Double-talk mixes
 *             the residual, coherence DROPS, and the boost decays — the safe
 *             direction. Authority is bounded (boost <= kBoostMax) and slow
 *             (integrator time constants >> template convergence time).
 *
 * RT contract: push() is wait-free (mono downmix + ring write, drops when
 * full); learningBoost() is one relaxed atomic load. Everything else runs on
 * the worker thread.
 */
class SpectralGovernor {
public:
  static SpectralGovernor &instance();

  /** Start the worker thread (idempotent; call from a non-RT thread). */
  void start();
  /** Stop and join the worker (idempotent). */
  void stop();

  /**
   * RT thread: feed one block of echo-aligned reference and post-template
   * residual (both interleaved, `channels` wide). Wait-free; drops the block
   * if the ring is full or the governor is not running.
   */
  void push(const float *refInterleaved, const float *residInterleaved,
            unsigned int frameCount, unsigned int channels);

  /** RT thread: current learning-rate multiplier, in [1, kBoostMax]. */
  float learningBoost() const {
    return mBoost.load(std::memory_order_relaxed);
  }

  /** Telemetry: last leakage estimate (residual-energy-weighted coherence). */
  float leakage() const { return mLeak.load(std::memory_order_relaxed); }

  /** Correlation-based double-talk detector: TRUE while the residual carries
   * substantial energy the reference CANNOT explain (low-coherence residual
   * well above its quiet floor) — i.e. the performer is playing/jamming or
   * the room is noisy. The template must freeze learning while this holds:
   * without it, practice licks and room noise fold into E[phi] at alpha and
   * get replayed inverted on later passes. Complements the leak signal
   * (coherent residual => learn HARDER); this is its mirror (incoherent
   * residual => STOP learning). */
  bool nearEndHold() const { return mNearEndHold.load(std::memory_order_relaxed); }
  /** Telemetry: incoherent-residual-to-quiet-floor ratio driving the hold. */
  float nearEndRatio() const { return mNearEndRatio.load(std::memory_order_relaxed); }

private:
  SpectralGovernor();
  ~SpectralGovernor();
  void workerLoop();
  void processWindow(const float *ref, const float *res);
  void controllerUpdate();

  // SPSC ring of (ref, residual) mono sample pairs. Single producer = render
  // thread, single consumer = worker.
  static constexpr size_t kRingFrames = 1 << 16; // ~1.4 s @ 48 kHz
  float *mRing = nullptr;                        // kRingFrames * 2 floats
  std::atomic<uint64_t> mWriteIdx{0};
  std::atomic<uint64_t> mReadIdx{0};

  std::atomic<bool> mRunning{false};
  std::thread mWorker;

  // Controller outputs / telemetry
  std::atomic<float> mBoost{1.0f};
  std::atomic<float> mLeak{0.0f};
  std::atomic<bool> mNearEndHold{false};
  std::atomic<float> mNearEndRatio{0.0f};
  double mIncoherentFloor = -1.0; // worker-thread-only quiet-floor tracker
  double mIncoherentFast = 0.0;   // fast EMA (~85ms) driving hold onset
  // Hold-loop damping state (worker-thread-only; the 3.5 Hz warble fix —
  // see the constants block in the .cpp for the full mechanism):
  int mEngageStreak = 0; // consecutive fast-path windows over the 8x ratio
  int mTicksHeld = 0;    // controller ticks since the hold engaged
  // Reference-onset tracker (transient-bleed fix — see processWindow):
  double mRefPowPrev = 0.0;  // smoothed broadband ref power, prev windows
  int mRefOnsetCooldown = 0; // windows of onset-aware engage skepticism

  // Worker-only spectral state (Welch EMA per band)
  static constexpr int kBands = 16;
  double mSxx[kBands] = {0};   // ref auto-power
  double mSyy[kBands] = {0};   // residual auto-power
  double mSxyRe[kBands] = {0}; // cross-power
  double mSxyIm[kBands] = {0};
  int mWindowsSeen = 0;
};

#endif // AEC_SPECTRAL_GOVERNOR_H
