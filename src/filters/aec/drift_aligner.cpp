#include "drift_aligner.h"

#include "delay_estimator.h"
#include "reference_buffer.h"

#include <algorithm>
#include <cmath>

extern void aecLog(const char *fmt, ...);

// ---- tuning constants (set conservatively; refined on the macOS soak test) ----
namespace {
constexpr size_t kEstWindow = 2048;   // correlation window (frames)
constexpr double kMinCorr = 0.15;     // peak-quality gate (double-talk/silence)
constexpr double kP = 0.25;           // proportional: snap current offset
constexpr double kI = 0.10;           // integral: learn the drift rate
constexpr double kLeak = 2e-4;        // tiny DC anchor on the read pointer
constexpr double kMaxDriftRatio = 3e-4; // ±300 ppm clamp (real device drift <100ppm)
constexpr double kMaxDriftStep = 5e-5;  // ±50 ppm max change per estimate
constexpr double kDriftLeakTau = 6.0;   // drift-rate leak time constant (seconds)
constexpr int kWatchdogGrow = 4;      // consecutive growing estimates -> reset
} // namespace

DriftAligner::DriftAligner(unsigned int sampleRate, unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels) {
  mHistSize = static_cast<size_t>(sampleRate) * 3 / 2; // ~1.5s mono history
  if (mHistSize < kEstWindow * 2)
    mHistSize = kEstWindow * 2;
  mRefHist.assign(mHistSize, 0.0f);
  mMicHist.assign(mHistSize, 0.0f);
  mRefWin.assign(kEstWindow, 0.0f);
  mMicWin.assign(kEstWindow, 0.0f);
  // Operate at a small positive residual so the (causal, tau>=0) estimator can
  // always see the peak and the FIR has pre-margin for the direct path.
  mLastResidual = static_cast<double>(mSampleRate) * 0.002; // ~2 ms
}

void DriftAligner::reset() {
  mRefReadPos = 0.0;
  mBulkDelay = 0.0;
  mDriftRatio = 0.0;
  mLastResidual = static_cast<double>(mSampleRate) * 0.002;
  mPrimed = false;
  mLocked = false;
  mGrowCount = 0;
  std::fill(mRefHist.begin(), mRefHist.end(), 0.0f);
  std::fill(mMicHist.begin(), mMicHist.end(), 0.0f);
  mHistPos = 0;
  mHistFilled = 0;
  mFramesSinceEstimate = 0;
}

size_t DriftAligner::produceAligned(const AECReferenceBuffer *refBuf,
                                    float *outRef, size_t frameCount,
                                    size_t seedDelayFrames) {
  if (!refBuf || !outRef || frameCount == 0)
    return 0;

  size_t totalWritten = refBuf->getFramesWritten();
  size_t bufFrames = refBuf->sizeInFrames();

  if (!mPrimed) {
    mBulkDelay = static_cast<double>(seedDelayFrames);
    if (totalWritten < frameCount + seedDelayFrames + 4) {
      std::fill(outRef, outRef + frameCount * mChannels, 0.0f);
      return 0; // not enough data behind the seed delay yet
    }
    mRefReadPos = static_cast<double>(totalWritten) -
                  static_cast<double>(frameCount) - mBulkDelay;
    mDriftRatio = 0.0;
    mLocked = false;
    mGrowCount = 0;
    mPrimed = true;
  } else {
    // Advance at the drift-corrected mic rate (the core of DCRA).
    mRefReadPos += static_cast<double>(frameCount) * (1.0 + mDriftRatio);
    // Tiny DC anchor so the free-running pointer can't wander out of the buffer
    // if estimates stall; small enough that it does not fight the drift slew.
    double anchor = static_cast<double>(totalWritten) -
                    static_cast<double>(frameCount) - mBulkDelay;
    mRefReadPos += kLeak * (anchor - mRefReadPos);
    // Leaky integrator on the DRIFT RATE itself: bleed it toward zero with a
    // ~6 s time constant. A genuinely-drifting device (e.g. USB mic + built-in
    // speaker on different crystals) has its rate continuously re-fed by the
    // I-term each estimate, so the leak balances at the true drift. A SHARED-
    // clock device (iOS/Android built-in duplex, true drift ~0) self-heals any
    // spurious estimate back to zero instead of slewing forever. This is the
    // portable fix that makes a bad estimate non-fatal.
    double leak = std::exp(-static_cast<double>(frameCount) /
                           (kDriftLeakTau * static_cast<double>(mSampleRate)));
    mDriftRatio *= leak;
  }

  // Clamp into the valid in-buffer span (margin for the +1 interpolation tap).
  double hi = static_cast<double>(totalWritten) -
              static_cast<double>(frameCount) - 2.0;
  double lo = (totalWritten > bufFrames)
                  ? static_cast<double>(totalWritten - bufFrames) + 2.0
                  : 2.0;
  if (hi < lo) {
    std::fill(outRef, outRef + frameCount * mChannels, 0.0f);
    return 0;
  }
  if (mRefReadPos > hi)
    mRefReadPos = hi;
  if (mRefReadPos < lo)
    mRefReadPos = lo;

  return refBuf->readFramesFractional(outRef, frameCount, mRefReadPos);
}

void DriftAligner::appendHistory(float alignedRefMono, float micMono) {
  if (mHistSize == 0)
    return;
  mRefHist[mHistPos] = alignedRefMono;
  mMicHist[mHistPos] = micMono;
  mHistPos = (mHistPos + 1) % mHistSize;
  if (mHistFilled < mHistSize)
    ++mHistFilled;

  if (++mFramesSinceEstimate >= static_cast<size_t>(mSampleRate) / 3) {
    mFramesSinceEstimate = 0;
    maybeEstimate();
  }
}

void DriftAligner::maybeEstimate() {
  if (mHistFilled < kEstWindow)
    return;

  // Copy the most-recent kEstWindow frames in chronological order.
  for (size_t i = 0; i < kEstWindow; ++i) {
    size_t idx = (mHistPos + mHistSize - kEstWindow + i) % mHistSize;
    mRefWin[i] = mRefHist[idx];
    mMicWin[i] = mMicHist[idx];
  }

  // Wide-ish one-time acquisition, then VERY narrow steady-state tracking. The
  // narrow locked search keeps the per-estimate CPU spike tiny (RT-safe on
  // small mobile buffers): drift between estimates is only a few frames, so
  // ±1 ms is ample once locked.
  int center = static_cast<int>(std::lround(mLastResidual));
  int search = mLocked ? static_cast<int>(mSampleRate * 0.004)  // ±4 ms locked
                       : static_cast<int>(mSampleRate * 0.01);  // ±10 ms cold
  double fracLag = 0.0, peak = 0.0;
  DelayEstimator::estimateDelayTargeted(mRefWin, mMicWin, center, search,
                                        &fracLag, &peak);
  if (peak < kMinCorr)
    return; // double-talk / silence -> hold last-good

  double residual = fracLag;
  if (std::fabs(residual) > search * 0.95)
    return; // ran into the search edge -> untrustworthy, hold

  // PHYSICAL RATE LIMIT (the key stability gate). True clock drift is bounded
  // by kMaxDriftRatio, so between estimates (T frames apart) the real alignment
  // can shift only a handful of frames. A residual that jumps far more than
  // that is a SPURIOUS correlation peak: music auto-correlates, so |corr| has
  // many comparable peaks at WRONG lags (we measured confident peaks — 0.5..0.84
  // — landing at 4, 70, 130, 160 frames within seconds). Acting on those jerks
  // the reference every 0.3 s, which detonates the NLMS filter (its weights
  // inflate, the output grows LOUDER than the input). Reject the outlier and
  // hold the existing lock; only a SUSTAINED disagreement means the path truly
  // moved (device/route change) and we should re-acquire from scratch.
  if (mLocked) {
    const double T0 = static_cast<double>(mSampleRate) / 3.0;
    const double maxPhysStep = kMaxDriftRatio * T0 * 8.0; // ~38 frames headroom
    if (std::fabs(residual - mLastResidual) > maxPhysStep) {
      if (++mOutlierCount < 12)
        return; // spurious peak — ignore it, keep the lock stable
      mLocked = false; // sustained shift: drop lock, re-acquire wide next time
      mDriftRatio = 0.0;
      mOutlierCount = 0;
      aecLog("[AEC DCRA] sustained residual shift -> re-acquiring\n");
      return;
    }
    mOutlierCount = 0;
  }

  // Divergence watchdog: if |residual| keeps growing, the loop is fighting
  // itself (sign/seed wrong) -> zero the drift slew and re-acquire.
  if (mLocked && std::fabs(residual) > std::fabs(mLastResidual) + 2.0) {
    if (++mGrowCount >= kWatchdogGrow) {
      mDriftRatio = 0.0;
      mGrowCount = 0;
      aecLog("[AEC DCRA] watchdog: residual diverging, drift reset\n");
    }
  } else {
    mGrowCount = 0;
  }

  // PI control toward a small positive target residual (keeps it findable).
  double target = static_cast<double>(mSampleRate) * 0.002; // ~2 ms
  double err = residual - target;
  // residual>target => aligned-ref too recent => read older => decrease pos.
  mRefReadPos -= kP * err;

  bool wasLocked = mLocked;
  // Only integrate drift on a STEADY-STATE lock. On cold acquisition the offset
  // error is large and is the P-term's job to snap away; feeding it to the I-
  // term is what produced the spurious -1184 ppm latch. The first lock snaps
  // offset only; drift starts learning from the (small) post-snap residual.
  if (wasLocked) {
    double T = static_cast<double>(mSampleRate) / 3.0;
    double step = kI * (err / T);
    // Clamp per-estimate change so a single noisy estimate can't dump a huge
    // rate. Real drift moves slowly; legitimate changes accumulate over many
    // estimates.
    if (step > kMaxDriftStep) step = kMaxDriftStep;
    if (step < -kMaxDriftStep) step = -kMaxDriftStep;
    mDriftRatio -= step;
    if (mDriftRatio > kMaxDriftRatio) mDriftRatio = kMaxDriftRatio;
    if (mDriftRatio < -kMaxDriftRatio) mDriftRatio = -kMaxDriftRatio;
  }

  // Re-center the next search on the POST-snap expected residual, not the raw
  // measured one — otherwise the ±4 ms locked window points at where the echo
  // *was* before the P-snap and we lose lock after one correction.
  mLastResidual = target + (residual - target) * (1.0 - kP);
  mLocked = true;

  aecLog("[AEC DCRA] residual=%.2f peak=%.2f drift=%.0fppm locked=%d bulk=%.0f pos=%.0f\n",
         residual, peak, mDriftRatio * 1e6, wasLocked ? 1 : 0, mBulkDelay,
         mRefReadPos);
}
