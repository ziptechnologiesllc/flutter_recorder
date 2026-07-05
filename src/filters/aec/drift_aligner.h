#ifndef AEC_DRIFT_ALIGNER_H
#define AEC_DRIFT_ALIGNER_H

#include <cstddef>
#include <vector>

class AECReferenceBuffer;

/**
 * DriftCompensatedReferenceAligner (DCRA).
 *
 * Root problem: on every platform miniaudio targets, the duplex device is two
 * independently-clocked streams (built-in mic vs built-in speaker on macOS;
 * two RemoteIO units on iOS; two streams on AAudio/WASAPI/ALSA) bridged by a
 * fixed-rate ring buffer with NO drift resampling. So the reference (speaker
 * clock) and the mic echo (mic clock) slowly slide apart — measured ~-346 ppm
 * on macOS, gentler on iOS. A fixed-tap echo canceller cannot identify a target
 * whose alignment is continuously moving, so cancellation either never locks
 * (macOS) or converges then degrades (iOS).
 *
 * DCRA is a delay-locked loop on the REFERENCE read. Instead of reading the
 * reference at a fixed integer offset, it advances a DOUBLE fractional read
 * pointer at (1 + drift) per mic frame and 2-tap interpolates, so the filter
 * always sees a stationary echo path. A slow estimator (cross-correlation of
 * the produced-aligned-ref vs the mic, ~3 Hz) measures the residual
 * misalignment; a conservative PI controller drives it to zero (P term snaps
 * the current offset, I term tracks the ongoing drift rate). On a genuine
 * shared clock the measured drift collapses to ~0 and DCRA is a graceful no-op
 * — same code, no platform branch.
 *
 * Single-threaded: all methods run on the capture/audio thread. No allocation
 * after construction; histories are ring buffers sized once.
 */
class DriftAligner {
public:
  DriftAligner(unsigned int sampleRate, unsigned int channels);

  /**
   * Produce frameCount drift-aligned reference frames (interleaved) into outRef.
   * @param refBuf          The shared reference ring (speaker output).
   * @param outRef          Destination, frameCount*channels interleaved.
   * @param frameCount      Frames to produce.
   * @param seedDelayFrames Initial bulk delay (acoustic+pipeline) used to seed
   *                        the read position before the first estimate locks.
   * @return frames produced (0 if the buffer can't satisfy the read yet).
   */
  size_t produceAligned(const AECReferenceBuffer *refBuf, float *outRef,
                        size_t frameCount, size_t seedDelayFrames);

  /**
   * Feed one mono frame of the JUST-PRODUCED aligned reference (ch0) and the
   * mic (ch0) to the drift estimator. Call once per frame after produceAligned.
   * Runs the periodic estimate internally when enough frames have accumulated.
   */
  void appendHistory(float alignedRefMono, float micMono);

  /** Reset all state (on enable/disable/recalibration/filter-length change). */
  void reset();

  // Telemetry (read on the audio thread for the [AEC] logs).
  double driftPpm() const { return mDriftRatio * 1e6; }
  double bulkDelayFrames() const { return mBulkDelay; }
  double residual() const { return mLastResidual; }
  bool primed() const { return mPrimed; }

private:
  void maybeEstimate();

  unsigned int mSampleRate;
  unsigned int mChannels;

  // --- delay-locked-loop state ---
  double mRefReadPos = 0.0;   // fractional absolute output-frame read position
  double mBulkDelay = 0.0;    // smoothed bulk delay (frames)
  double mDriftRatio = 0.0;   // extra ref frames per mic frame (~ppm/1e6)
  double mLastResidual = 0.0; // last measured residual misalignment (frames)
  bool mPrimed = false;       // read position initialised
  bool mLocked = false;       // first good estimate landed (narrow search after)
  int mGrowCount = 0;         // consecutive estimates with growing |residual|
  int mOutlierCount = 0;      // consecutive physically-impossible residual jumps

  // --- mono histories for the estimator (ring) ---
  std::vector<float> mRefHist;
  std::vector<float> mMicHist;
  size_t mHistSize = 0;
  size_t mHistPos = 0;    // next write index
  size_t mHistFilled = 0; // frames written so far (caps at mHistSize)
  size_t mFramesSinceEstimate = 0;

  // scratch windows for the estimate (preallocated, reused)
  std::vector<float> mRefWin;
  std::vector<float> mMicWin;
};

#endif // AEC_DRIFT_ALIGNER_H
