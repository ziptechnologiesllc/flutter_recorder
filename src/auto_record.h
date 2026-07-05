#ifndef AUTO_RECORD_H
#define AUTO_RECORD_H

#include <atomic>
#include <cstdint>
#include <cstring>

// Forward declaration
class Capture;

/**
 * Hands-free first-loop capture ("auto-record").
 *
 * The user *holds* the record button to arm this. While the button is held the
 * detector measures the ambient noise level (so the trigger threshold adapts to
 * the room); on release ([endAmbientMeasure]) the threshold is locked and the
 * detector starts listening. The first audio onset (a strum, a piano chord, a
 * struck note — an energy attack, not pitch) becomes the loop's downbeat.
 * Everything before it was only ever sitting in the native ring buffer, so it
 * never enters the take — the lead-in silence is trimmed "for free" by rewinding
 * the ring buffer to the onset (minus a small attack-lookback so the pick/key
 * attack itself isn't shaved).
 *
 * Optionally, when a preset phrase length (`barCount`) and a tempo
 * (`framesPerBar`, e.g. from Ableton Link or the metronome) are supplied at arm
 * time, the take auto-stops at exactly `start + barCount * framesPerBar` — fully
 * hands-free, deterministic, sample-aligned. When the tempo is unknown
 * (`framesPerBar == 0`) the preset auto-stop is skipped here; Phase 2 adds an
 * onset-driven tempo estimator + a tap-to-stop "snap to nearest bar" path.
 *
 * Lives entirely on the capture audio thread (driven from `data_callback`),
 * lock-free; `arm()/disarm()/endAmbientMeasure()` are called from the Dart main
 * thread. Start/stop are routed through `NativeScheduler` so the recording
 * bookkeeping (active path, WAV-worker handoff, Dart `RecordingStarted/Stopped`
 * notifications) is identical to the quantized path.
 *
 * Scope: first loop only. Once a base loop exists, the existing quantize/queue
 * path owns recording and auto-record is a no-op.
 */
class AutoRecorder {
public:
  static AutoRecorder& instance();

  enum class State : uint8_t {
    Idle = 0,
    Armed = 1,      // waiting for the first onset (or still measuring ambient)
    Recording = 2,  // onset detected; recording (may have a pending preset stop)
  };

  // ===================== CALLED FROM DART (main thread) =====================

  /// Arm auto-record.
  /// @param wavPath        where the take is written on stop.
  /// @param barCount       preset phrase length in bars; <= 0 = no preset
  ///                       length ("∞" mode — records until a manual stop).
  /// @param framesPerBar   tempo, in frames per bar (one bar = 4 beats, 4/4
  ///                       assumed); > 0 enables the preset auto-stop; 0 =
  ///                       tempo unknown (preset auto-stop skipped — Phase 2).
  /// @param sampleRate     capture sample rate (for refractory / lookback /
  ///                       warm-up timing); 0 = keep current.
  /// @param measureAmbient if true, the warm-up is *open-ended* — the detector
  ///                       keeps tracking the ambient level (and doesn't listen
  ///                       for onsets) until [endAmbientMeasure] is called. This
  ///                       is the "hold the button" model: the longer you hold,
  ///                       the better the ambient estimate. If false, a fixed
  ///                       short warm-up settles the floor and then it listens.
  void arm(const char* wavPath, int barCount, int64_t framesPerBar,
           unsigned int sampleRate, bool measureAmbient);

  /// End the ambient-measure window (the user released the held button): lock
  /// the floor to what was measured and start listening for onsets after a
  /// brief final settle. No-op unless [State::Armed] and currently measuring.
  void endAmbientMeasure();

  /// Disarm. If a recording is already underway it is *not* stopped here — the
  /// caller stops it through the normal stop path; this only cancels the
  /// armed-and-waiting state and any pending preset stop.
  void disarm();

  State state() const { return mState.load(std::memory_order_acquire); }

  /// True while the (armed) detector is still measuring ambient — i.e. the
  /// button is being held. Goes false on [endAmbientMeasure].
  bool isMeasuringAmbient() const { return mMeasuringAmbient.load(std::memory_order_acquire); }

  /// Onset-detector sensitivity: how many dB above the (ambient) noise floor an
  /// envelope jump must reach to count as an attack. Lower = more sensitive
  /// (soft-onset instruments — bowed strings, pads); higher = needs a firmer
  /// hit. Default ~12 dB suits clear attacks (strum / pluck / piano / perc).
  void setOnsetThresholdDb(float db) {
    mOnsetThresholdDb.store(db, std::memory_order_release);
  }
  float onsetThresholdDb() const { return mOnsetThresholdDb.load(std::memory_order_acquire); }

  /// Current measured noise floor in dBFS (0 dB = full scale). ~-inf when
  /// silent. For the UI (e.g. drawing the threshold line on the waveform).
  float noiseFloorDb() const;

  /// Current onset trigger level in dBFS = noiseFloorDb + onsetThresholdDb.
  /// The level an envelope must exceed to fire. For the UI threshold line.
  float triggerLevelDb() const;

  /// Best current tempo estimate in BPM (0 until Phase 2's estimator locks).
  float estimatedBpm() const { return mEstimatedBpm.load(std::memory_order_acquire); }

  /// True while armed or recording — other code paths can use this to defer.
  bool isActive() const { return mState.load(std::memory_order_acquire) != State::Idle; }

  /// Reset to Idle (call on session end / engine teardown).
  void reset();

  /// Called whenever a recording ends (manual stop, preset auto-stop, anything)
  /// so a stale armed/recording state can't later schedule a spurious stop.
  /// Cheap & lock-free; safe from the audio thread.
  void onRecordingStopped();

  // ============== CALLED FROM THE AUDIO THREAD (data_callback) ==============
  // Invoke after the ring-buffer write and before NativeScheduler::processEvents
  // so a same-buffer start lines up with this buffer's bookkeeping.
  //
  // @param interleaved       captured float samples for this buffer
  // @param frameCount        frames in this buffer
  // @param channels          channels in `interleaved`
  // @param bufferStartFrame  global frame of the first sample in this buffer
  //                          (== ring buffer totalFramesWritten *before* this
  //                          buffer's write)
  // @param capture           for the no-ring-buffer fallback path
  void process(const float* interleaved, uint32_t frameCount,
               unsigned int channels, int64_t bufferStartFrame, Capture* capture);

private:
  AutoRecorder();
  ~AutoRecorder() = default;
  AutoRecorder(const AutoRecorder&) = delete;
  AutoRecorder& operator=(const AutoRecorder&) = delete;

  // Recompute the per-sample one-pole coefficients for the current sample rate.
  void updateCoeffsForSampleRate(unsigned int sampleRate);

  /// Begin the take at `onsetGlobalFrame` (the detected attack frame, already
  /// backed off by the attack-lookback) with `preRollFrames` of ring-buffer
  /// rewind (so linear[0..] starts exactly there). Hands off to NativeScheduler.
  /// Audio thread.
  void beginTake(int64_t onsetGlobalFrame, int64_t preRollFrames, Capture* capture);

  // ---- detector state (audio thread only, except mNoiseFloor which is also
  // read by the UI getters on the main thread, hence atomic) ----
  float mFastEnv = 0.0f;               // fast-attack / slow-release envelope
  std::atomic<float> mNoiseFloor{1e-4f}; // adaptive noise floor (linear)
  bool mAboveThreshold = false;        // rising-edge tracking
  int64_t mLastOnsetFrame = -1;        // global frame of the last accepted onset
  // Frames left in the post-arm "warm-up": we let the noise floor settle to the
  // ambient/playing level before listening for onsets, so room tone (or the
  // tail of whatever was playing when you armed) doesn't false-trigger. While
  // measuring ambient (button held) this is held at the full value and only
  // counts down after endAmbientMeasure().
  int64_t mWarmupFramesRemaining = 0;
  std::atomic<bool> mMeasuringAmbient{false};
  bool mWasMeasuringAmbient = false;   // audio-thread copy, for falling-edge detect

  // per-sample one-pole coefficients (recomputed on sample-rate change)
  float mAttackCoeff = 0.5f;
  float mReleaseCoeff = 0.02f;
  float mFloorUpCoeff = 0.0005f;       // noise floor rises slowly
  float mFloorDownCoeff = 0.02f;       // ...drops a touch faster when it's quiet
  float mWarmupDecayCoeff = 0.002f;    // recent-ambient-peak decay during warm-up
  int64_t mRefractoryFrames = 3000;    // ~70 ms @ 44.1k (recomputed)
  int64_t mAttackLookbackFrames = 440; // ~10 ms @ 44.1k (recomputed)
  int64_t mWarmupFrames = 0;           // ~150 ms — initial settle (recomputed)
  int64_t mShortWarmupFrames = 0;      // ~100 ms — final settle after a held measure
  int64_t mArmTimeoutFrames = 0;       // ~20 s (recomputed)
  int64_t mMaxTakeFrames = 0;          // ~40 s (recomputed)

  std::atomic<State> mState{State::Idle};
  std::atomic<float> mOnsetThresholdDb{12.0f};
  std::atomic<float> mEstimatedBpm{0.0f};

  // ---- arm config: written by arm() while Idle, read on the audio thread ----
  char mWavPath[512] = {0};
  int mBarCount = 0;
  int64_t mFramesPerBar = 0;
  unsigned int mSampleRate = 44100;

  // ---- take state (audio thread; mPendingStopEventId also touched by disarm()
  // on the main thread, hence atomic) ----
  int64_t mArmedAtFrame = 0;        // global frame at which arming/listening began
  int64_t mTakeStartFrame = 0;      // the downbeat frame
  std::atomic<uint32_t> mPendingStopEventId{0}; // scheduled preset stop (0=none)

  // ---- constants ----
  static constexpr float kRefractoryMs = 70.0f;
  static constexpr float kAttackLookbackMs = 10.0f;
  static constexpr float kArmWarmupMs = 200.0f;       // fixed initial settle
  static constexpr float kPostMeasureSettleMs = 250.0f; // settle after a held measure
                                                        // (also eats the button-release thump)
  static constexpr float kAttackTauMs = 1.0f;       // envelope attack time
  static constexpr float kReleaseTauMs = 60.0f;     // envelope release time
  static constexpr float kFloorUpTauMs = 1500.0f;   // noise-floor rise time
  static constexpr float kFloorDownTauMs = 250.0f;  // noise-floor fall time
  static constexpr float kWarmupDecayTauMs = 600.0f; // "recent peak" memory while warming
  static constexpr float kAbsFloorLinear = 3.0e-4f; // never trigger below this
  static constexpr float kArmTimeoutSec = 20.0f;
  static constexpr float kMaxTakeSec = 40.0f;
};

#endif // AUTO_RECORD_H
