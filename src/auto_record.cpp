#include "auto_record.h"

#include "capture.h"
#include "common.h"
#include "native_scheduler.h"

#include <cmath>

#ifdef _IS_ANDROID_
#include <android/log.h>
#define AR_LOG(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "AutoRecorder", fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define AR_LOG(fmt, ...) printf("[AutoRecorder] " fmt "\n", ##__VA_ARGS__)
#endif

// Debug logging — disable for production (no I/O on the audio thread!).
#define DEBUG_AUTO_RECORD 0
#if DEBUG_AUTO_RECORD
#define AR_DEBUG(fmt, ...) AR_LOG(fmt, ##__VA_ARGS__)
#else
#define AR_DEBUG(fmt, ...) ((void)0)
#endif

// One-pole smoothing coefficient for a time constant `tauMs` at `sr` Hz:
//   coeff = 1 - exp(-1 / (tau_seconds * sr))
static inline float onePoleCoeff(float tauMs, float sr) {
  if (tauMs <= 0.0f || sr <= 0.0f) return 1.0f;
  const float tauSamples = (tauMs * 0.001f) * sr;
  if (tauSamples < 1.0f) return 1.0f;
  return 1.0f - std::exp(-1.0f / tauSamples);
}

static inline float linToDb(float lin) {
  if (lin <= 1e-9f) return -180.0f;
  return 20.0f * std::log10(lin);
}

AutoRecorder& AutoRecorder::instance() {
  static AutoRecorder inst;
  return inst;
}

AutoRecorder::AutoRecorder() {
  updateCoeffsForSampleRate(44100);
}

void AutoRecorder::updateCoeffsForSampleRate(unsigned int sampleRate) {
  if (sampleRate == 0) sampleRate = 44100;
  mSampleRate = sampleRate;
  const float sr = (float)sampleRate;
  mAttackCoeff = onePoleCoeff(kAttackTauMs, sr);
  mReleaseCoeff = onePoleCoeff(kReleaseTauMs, sr);
  mFloorUpCoeff = onePoleCoeff(kFloorUpTauMs, sr);
  mFloorDownCoeff = onePoleCoeff(kFloorDownTauMs, sr);
  mWarmupDecayCoeff = onePoleCoeff(kWarmupDecayTauMs, sr);
  mRefractoryFrames = (int64_t)(kRefractoryMs * 0.001f * sr);
  mAttackLookbackFrames = (int64_t)(kAttackLookbackMs * 0.001f * sr);
  mWarmupFrames = (int64_t)(kArmWarmupMs * 0.001f * sr);
  mShortWarmupFrames = (int64_t)(kPostMeasureSettleMs * 0.001f * sr);
  mArmTimeoutFrames = (int64_t)(kArmTimeoutSec * sr);
  mMaxTakeFrames = (int64_t)(kMaxTakeSec * sr);
}

float AutoRecorder::noiseFloorDb() const {
  return linToDb(mNoiseFloor.load(std::memory_order_relaxed));
}

float AutoRecorder::triggerLevelDb() const {
  return noiseFloorDb() + mOnsetThresholdDb.load(std::memory_order_acquire);
}

void AutoRecorder::arm(const char* wavPath, int barCount, int64_t framesPerBar,
                       unsigned int sampleRate, bool measureAmbient) {
  if (sampleRate != 0 && sampleRate != mSampleRate) {
    updateCoeffsForSampleRate(sampleRate);
  }
  if (wavPath != nullptr) {
    std::strncpy(mWavPath, wavPath, sizeof(mWavPath) - 1);
    mWavPath[sizeof(mWavPath) - 1] = '\0';
  } else {
    mWavPath[0] = '\0';
  }
  mBarCount = barCount;
  mFramesPerBar = framesPerBar;

  // Fresh detector state for this arming.
  mFastEnv = 0.0f;
  mNoiseFloor.store(kAbsFloorLinear, std::memory_order_relaxed);
  mAboveThreshold = false;
  mLastOnsetFrame = -1;
  mWarmupFramesRemaining = mWarmupFrames;
  mMeasuringAmbient.store(measureAmbient, std::memory_order_relaxed);
  mWasMeasuringAmbient = measureAmbient;
  mTakeStartFrame = 0;
  mPendingStopEventId.store(0, std::memory_order_release);
  mEstimatedBpm.store(0.0f, std::memory_order_release);
  mArmedAtFrame = NativeScheduler::instance().getGlobalFrame();

  // Publish: config first, then the state transition (release).
  mState.store(State::Armed, std::memory_order_release);
  // Logged unconditionally (main thread, infrequent) — handy for diagnosing
  // whether the *native* lib actually got the latest build.
  AR_LOG("armed: bars=%d framesPerBar=%lld sr=%u measure=%d armedAt=%lld path=%s",
         barCount, (long long)framesPerBar, mSampleRate, (int)measureAmbient,
         (long long)mArmedAtFrame, mWavPath);
}

void AutoRecorder::endAmbientMeasure() {
  if (mState.load(std::memory_order_acquire) != State::Armed) {
    AR_LOG("endAmbientMeasure: not armed (state=%d) — ignored",
           (int)mState.load(std::memory_order_acquire));
    return;
  }
  // The falling edge of mMeasuringAmbient is picked up in process() (audio
  // thread) — it resets the warm-up tail and the arm-timeout anchor there, so
  // we don't write those non-atomic members from this (main) thread.
  mMeasuringAmbient.store(false, std::memory_order_release);
  AR_LOG("ambient measure ended — locking floor, will listen for onsets");
}

void AutoRecorder::disarm() {
  State st = mState.load(std::memory_order_acquire);
  const uint32_t pending = mPendingStopEventId.exchange(0, std::memory_order_acq_rel);
  if (pending != 0) {
    NativeScheduler::instance().cancelEvent(pending);
  }
  mMeasuringAmbient.store(false, std::memory_order_release);
  // Note: an in-progress take is left for the normal stop path to finish.
  mState.store(State::Idle, std::memory_order_release);
  AR_LOG("disarmed (was state=%d)", (int)st);
}

void AutoRecorder::onRecordingStopped() {
  // Whatever ended the take (manual tap, preset auto-stop, safety net), make
  // sure we don't later fire a stale stop.
  mPendingStopEventId.store(0, std::memory_order_release);
  mMeasuringAmbient.store(false, std::memory_order_release);
  if (mState.load(std::memory_order_acquire) != State::Idle) {
    mState.store(State::Idle, std::memory_order_release);
    AR_DEBUG("recording stopped — back to idle");
  }
}

void AutoRecorder::reset() {
  mState.store(State::Idle, std::memory_order_release);
  mFastEnv = 0.0f;
  mNoiseFloor.store(kAbsFloorLinear, std::memory_order_relaxed);
  mAboveThreshold = false;
  mLastOnsetFrame = -1;
  mWarmupFramesRemaining = 0;
  mMeasuringAmbient.store(false, std::memory_order_release);
  mWasMeasuringAmbient = false;
  mTakeStartFrame = 0;
  mPendingStopEventId.store(0, std::memory_order_release);
  mWavPath[0] = '\0';
  mBarCount = 0;
  mFramesPerBar = 0;
  mEstimatedBpm.store(0.0f, std::memory_order_release);
}

void AutoRecorder::beginTake(int64_t onsetGlobalFrame, int64_t preRollFrames,
                             Capture* capture) {
  // Auto-record is first-loop only — if a base loop exists, the quantize path
  // owns recording. Bail out (back to Idle).
  if (NativeScheduler::instance().getBaseLoopFrames() > 0) {
    mState.store(State::Idle, std::memory_order_release);
    return;
  }

  if (preRollFrames < 0) preRollFrames = 0;
  mTakeStartFrame = onsetGlobalFrame;

  NativeScheduler::instance().beginAutoRecording(onsetGlobalFrame, preRollFrames,
                                                 mWavPath, capture);

  // Deterministic preset-length stop, if a tempo was supplied at arm time.
  mPendingStopEventId.store(0, std::memory_order_release);
  if (mBarCount > 0 && mFramesPerBar > 0) {
    const int64_t stopFrame = onsetGlobalFrame + (int64_t)mBarCount * mFramesPerBar;
    mPendingStopEventId.store(
        NativeScheduler::instance().scheduleEvent(SchedulerAction::StopRecording,
                                                  stopFrame, nullptr),
        std::memory_order_release);
    AR_DEBUG("take started at %lld; preset stop at %lld (%d bars x %lld)",
             (long long)onsetGlobalFrame, (long long)stopFrame, mBarCount,
             (long long)mFramesPerBar);
  } else {
    AR_DEBUG("take started at %lld; no preset stop (bars=%d framesPerBar=%lld)",
             (long long)onsetGlobalFrame, mBarCount, (long long)mFramesPerBar);
  }

  mState.store(State::Recording, std::memory_order_release);
}

void AutoRecorder::process(const float* interleaved, uint32_t frameCount,
                           unsigned int channels, int64_t bufferStartFrame,
                           Capture* capture) {
  const State st = mState.load(std::memory_order_acquire);
  if (st == State::Idle) return;
  if (interleaved == nullptr || frameCount == 0 || channels == 0) return;

  const int64_t bufferEndFrame = bufferStartFrame + (int64_t)frameCount;

  // Detect the falling edge of "measuring ambient" (the held button was
  // released): from here the warm-up tail counts down and onset-listening
  // begins. The arm-timeout clock restarts now so a long hold doesn't expire.
  const bool measuring = mMeasuringAmbient.load(std::memory_order_acquire);
  if (mWasMeasuringAmbient && !measuring) {
    mWarmupFramesRemaining = mShortWarmupFrames;
    mArmedAtFrame = bufferStartFrame;
  }
  mWasMeasuringAmbient = measuring;

  // Envelope follower over this buffer (mono-summed), per sample.
  const float thr = std::pow(
      10.0f, mOnsetThresholdDb.load(std::memory_order_acquire) / 20.0f);
  const float invCh = 1.0f / (float)channels;
  const float floorFloor = kAbsFloorLinear * 0.1f;

  float fastEnv = mFastEnv;
  float floor = mNoiseFloor.load(std::memory_order_relaxed);
  bool above = mAboveThreshold;
  int64_t lastOnset = mLastOnsetFrame;

  bool didStart = false;
  int64_t startGlobalFrame = 0;
  int64_t startPreRoll = 0;

  int64_t warmup = mWarmupFramesRemaining;

  for (uint32_t i = 0; i < frameCount; ++i) {
    const float* f = interleaved + (size_t)i * channels;
    float mag = 0.0f;
    for (unsigned int c = 0; c < channels; ++c) mag += std::fabs(f[c]);
    mag *= invCh;

    if (mag > fastEnv) fastEnv += mAttackCoeff * (mag - fastEnv);
    else               fastEnv += mReleaseCoeff * (mag - fastEnv);

    // Warm-up / ambient-measure: settle the floor to the *recent peak of the
    // envelope* before listening for onsets. Snap up to the peak; otherwise
    // decay slowly toward the current envelope (NOT the instantaneous |sample|,
    // which dips toward zero between cycles and would drag the floor — and thus
    // the trigger — far below the real ambient level → false triggers). The
    // slow decay lets a one-off transient (a cough, the button-release thump)
    // settle out within ~0.6 s. While the button is still held (measuring), the
    // warm-up never counts down.
    if (warmup > 0) {
      if (!measuring) --warmup;
      if (fastEnv > floor) floor = fastEnv;
      else                 floor += mWarmupDecayCoeff * (fastEnv - floor);
      if (floor < floorFloor) floor = floorFloor;
      above = false;
      continue;
    }

    const float trigger = floor * thr;
    const bool nowAbove = (fastEnv > trigger) && (fastEnv > kAbsFloorLinear);

    if (nowAbove && !above) {
      const int64_t frame = bufferStartFrame + (int64_t)i;
      if (lastOnset < 0 || (frame - lastOnset) >= mRefractoryFrames) {
        lastOnset = frame;
        if (st == State::Armed && !didStart) {
          // The downbeat. Back off by the attack-lookback so the very start of
          // the attack is kept (the envelope crosses the threshold a few ms
          // after the true onset). Pre-roll is buffer-relative: how far back
          // from the *end* of what's been written to the ring buffer the
          // onset sits, plus the lookback.
          startPreRoll = (int64_t)frameCount - (int64_t)i + mAttackLookbackFrames;
          startGlobalFrame = frame - mAttackLookbackFrames;
          if (startGlobalFrame < 0) startGlobalFrame = 0;
          didStart = true;
          // Keep scanning so the envelope/floor state is consistent for the
          // rest of the buffer, but don't trigger again.
        }
        // st == Recording: onsets here will feed the Phase-2 tempo estimator.
      }
    }
    above = nowAbove;

    // Adaptive noise floor — frozen while we're "in" an onset so a loud attack
    // doesn't drag the floor up and desensitise the next trigger. Track the
    // *envelope* (fastEnv), not |sample|, for the same reason as the warm-up.
    if (!nowAbove) {
      if (fastEnv > floor) floor += mFloorUpCoeff * (fastEnv - floor);
      else                 floor += mFloorDownCoeff * (fastEnv - floor);
      if (floor < floorFloor) floor = floorFloor;
    }
  }

  mFastEnv = fastEnv;
  mNoiseFloor.store(floor, std::memory_order_relaxed);
  mAboveThreshold = above;
  mLastOnsetFrame = lastOnset;
  mWarmupFramesRemaining = warmup;

  if (didStart) {
    beginTake(startGlobalFrame, startPreRoll, capture);
    return;
  }

  // Timeouts.
  if (st == State::Armed) {
    // Don't time out while the button is still held (measuring) or during the
    // post-release settle — only the actual onset-listening window counts.
    if (!measuring && warmup <= 0 &&
        mArmTimeoutFrames > 0 && (bufferEndFrame - mArmedAtFrame) > mArmTimeoutFrames) {
      // No onset for a long time — give up. Nothing was recorded, so no
      // notification is needed; Dart sees Idle on its next state poll.
      mState.store(State::Idle, std::memory_order_release);
      mMeasuringAmbient.store(false, std::memory_order_release);
      AR_DEBUG("arm timed out (no onset)");
    }
  } else if (st == State::Recording) {
    if (mMaxTakeFrames > 0 && (bufferEndFrame - mTakeStartFrame) > mMaxTakeFrames) {
      // Safety net — never let an armed take run away.
      NativeScheduler::instance().scheduleEvent(SchedulerAction::StopRecording,
                                                bufferEndFrame, nullptr);
      mPendingStopEventId.store(0, std::memory_order_release);
      mState.store(State::Idle, std::memory_order_release);
      AR_DEBUG("take hit max length — stopping");
    }
  }
}
