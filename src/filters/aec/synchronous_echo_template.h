#ifndef AEC_SYNCHRONOUS_ECHO_TEMPLATE_H
#define AEC_SYNCHRONOUS_ECHO_TEMPLATE_H

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <vector>

/**
 * SynchronousEchoTemplate — the core of LSAEC (Loop-Synchronous AEC).
 *
 * WHY this exists (and why it replaces the NLMS/shadow/promotion stack):
 * CloudLoop plays back a loop of KNOWN period P (frames) and controls its
 * timing, so the speaker output — and therefore the acoustic echo of it — is
 * exactly PERIODIC at P. The performer's live overdub is NOT periodic at P.
 * That single fact is a separation axis a performer physically cannot defeat.
 *
 * So instead of solving a moving, ill-conditioned linear inverse (an adaptive
 * filter, which on a periodic music reference wanders and diverges — the
 * "recording grows over time" failure), we keep one echo estimate PER LOOP
 * PHASE: E[phi], phi in [0, P). Each loop pass, E[phi] is nudged toward the mic
 * by a small leaky average. The period-coherent echo ACCUMULATES; the
 * period-incoherent performance AVERAGES TOWARD ZERO (~sqrt(#periods)). The
 * cancelled output is simply mic - E[phi].
 *
 * There are NO filter weights, so unbounded weight inflation — the entire
 * divergence failure mode — is structurally impossible. The acoustic delay D
 * never enters the learning: the echo is periodic at P whatever the delay, so
 * E indexed by the mic's loop phase captures it directly.
 *
 * Single-threaded: process() runs on the capture/audio thread. No allocation
 * after construction (capacity is sized once for the maximum supported period).
 */
class SynchronousEchoTemplate {
public:
  SynchronousEchoTemplate(unsigned int sampleRate, unsigned int channels);

  /**
   * Learning-rate multiplier from the spectral governor (>=1). Applied on top
   * of the annealed alpha, capped at the hot ceiling — measured echo leakage
   * re-heats learning; convergence lets it anneal back down. RT-safe setter
   * (relaxed atomic store from the render thread itself).
   */
  void setLearnBoost(float b) {
    mLearnBoost.store(b < 1.0f ? 1.0f : b, std::memory_order_relaxed);
  }

  /** Clear all learned echo (on enable/disable/reset). */
  void reset();

  /**
   * Cancel one interleaved block IN PLACE.
   *
   * @param micInOut       Interleaved mic in; cleaned (mic - echo) out.
   * @param alignedRef     Interleaved echo-aligned reference for far-end
   *                       gating (only LEARN where the speaker is playing).
   *                       May be null -> always learn.
   * @param frameCount     Frames in this block.
   * @param channels       Interleave width (must be <= construction channels).
   * @param blockStartFrame Engine frame counter of the first sample (same
   *                       coordinate as loopStartFrame).
   * @param loopFrames     Loop period P in frames (0 or >capacity -> passthrough).
   * @param loopStartFrame Engine frame where loop phase 0 falls.
   * @param learn          When false, cancel but do NOT update the template
   *                       (double-talk / transient freeze; see LSAEC E3).
   */
  void process(float *micInOut, const float *alignedRef,
               unsigned int frameCount, unsigned int channels,
               int64_t blockStartFrame, int64_t loopFrames,
               int64_t loopStartFrame, bool learn);

  /** Active loop period actually in use (0 if passthrough). */
  int64_t activeLoopFrames() const { return mActiveLoopFrames; }

  /** Mean per-phase confidence over the active period (0..1), for telemetry. */
  float meanConfidence() const;

  /** E3 diagnostics (monotonic counters since reset). */
  uint32_t freezeCount() const { return mFreezeCount; }
  uint32_t reopenCount() const { return mReopenCount; }

private:
  unsigned int mSampleRate;
  unsigned int mChannels;
  size_t mCapacityFrames;          // max P we can hold (per channel)
  std::vector<float> mTemplate;    // E[phi*channels + ch] — echo estimate
  std::vector<float> mConfidence;  // per-phase saturating update count (anneal)
  int64_t mActiveLoopFrames = 0;   // P currently in use (<= capacity)

  // E3 block-level double-talk freeze. A smoothed per-BLOCK residual floor; a
  // block whose residual spikes well above it is near-end -> skip learning that
  // block (per-frame thresholding thrashed — audio energy is too spiky).
  // Smoothed reference power envelope for the far-end learn gate. Gating on
  // INSTANTANEOUS per-sample power leaves unlearned pinholes at every zero
  // crossing of the reference waveform (hundreds/sec) — the mic echo (delayed
  // ~26 ms, nonzero there) pokes through each hole every pass = broadband
  // crackle that WORSENS as the rest of the template converges.
  std::atomic<float> mLearnBoost{1.0f}; // spectral-governor gain

  float mRefEnvelope = 0.0f;

  float mResidBaseline = 0.0f; // EMA of per-block residual energy
  uint32_t mLearnedBlocks = 0; // non-frozen learning blocks (arms the detector)
  uint32_t mFreezeCount = 0;   // blocks frozen (near-end) — telemetry
  uint32_t mReopenCount = 0;   // reserved (block-level recovery) — telemetry
};

#endif // AEC_SYNCHRONOUS_ECHO_TEMPLATE_H
