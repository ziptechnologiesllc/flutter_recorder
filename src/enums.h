#ifndef ENUMS_H
#define ENUMS_H

#include <stdint.h>

/// Possible capture errors
typedef enum CaptureErrors {
  /// No error
  captureNoError = 0,
  /// The capture device has failed to initialize.
  captureInitFailed = 1,
  /// The capture device has not yet been initialized.
  captureNotInited = 2,
  /// Failed to start the device.
  failedToStartDevice = 3,
  /// Failed to initialize wav recording.
  failedToInitializeRecording = 4,
  /// Invalid arguments while initializing wav recording.
  invalidArgs = 5,
  /// Failed to write wav file.
  failedToWriteWav = 6,
  /// Filter not found
  filterNotFound = 7,
  /// The filter has already been added.
  filterAlreadyAdded = 8,
  /// Error getting filter parameter.
  filterParameterGetError = 9
} CaptureErrorsInternal_t;

typedef enum PCMFormat {
  pcm_u8,
  pcm_s16,
  pcm_s24,
  pcm_s32,
  pcm_f32,
  pcm_unknown
} PCMFormatInternal_t;

typedef enum RecorderFilterType {
  autogain,
  echoCancellation,
  adaptiveEchoCancellation
} FilterType_t;

typedef enum AecMode {
  aecModeBypass = 0,
  aecModeAlgo = 1,       // Adaptive NLMS (legacy)
  aecModeNeural = 2,     // Neural post-filter only
  aecModeHybrid = 3,     // Adaptive NLMS + Neural
  aecModeFrozen = 4,      // Frozen FIR (pure calibrated IR, no adaptation)
  aecModeFrozenNeural = 5, // Frozen FIR + Neural post-filter
  aecModeLsaec = 6         // Loop-synchronous echo template (slave mode + known
                           // loop period; falls back to NLMS until a loop exists)
} AecMode_t;

typedef struct {
  float maxAttenuationDb;
  float correlation;
  float echoReturnLossDb;
  // Debug display fields
  int filterLength;         // Current filter length in samples
  float muMax;              // Configured max step size
  float muEffective;        // Last effective step size (runtime)
  float lastErrorDb;        // Last error in dB
  float instantCorrelation; // Instantaneous correlation metric
} AecStats;

// LSAEC E5: lock-free gated-ERLE telemetry snapshot. The audio thread publishes
// windowed energy SUMS (not dB) once per ~0.25 s; a Dart poller computes dB
// off-thread. Plain trivially-copyable struct so it can ride a Seqlock<T>.
// "Far" = far-end (speaker) active — ERLE must be gated to those samples only.
struct AecTelemetrySnapshot {
  double micEnergyFar; // Σ mic² over far-end-active samples (gated ERLE num.)
  double outEnergyFar; // Σ out²(final) over far-end-active samples (denom.)
  double micEnergyAll; // Σ mic² over all samples
  double outEnergyAll; // Σ out²(final) over all samples
  double refEnergyAll; // Σ ref² over all samples
  uint64_t farSamples;   // far-end-active samples in the window
  uint64_t totalSamples; // total samples in the window
  uint64_t generation;   // increments each published window

  // LSAEC debug-overlay fields (added alongside the seed/mute-notify work):
  // distinguish "converging normally" from "seeding" from "stuck" without
  // grepping raw logs. Zeroed when the AEC filter isn't in LSAEC mode.
  float templateConfidence; // mean per-phase confidence, 0..1
  uint32_t freezeCount;     // E3 double-talk-freeze counter (monotonic)
  uint32_t isSeeding;       // 1 while a convergence-seed job is in flight
  uint32_t overCapacity;    // 1 = loop period exceeds 16s cap: cancellation OFF
  float govLeak;            // spectral governor's last coherence-leak reading
  float govBoost;           // spectral governor's current learning-rate boost

  // Convergence-seed lifecycle counters (monotonic). aborts >> lands means
  // mix-change notifies keep killing the one-period reference capture (seed
  // livelock): convergence is riding pure per-pass EMA, which on real loop
  // lengths is 30-60 s of wall clock — the "takes forever to converge" report.
  uint32_t seedArms;
  uint32_t seedAborts;
  uint32_t seedLands;
  uint32_t seedPhase; // see SynchronousEchoTemplate::seedPhase()
  uint32_t seedDiscards; // fit-stage rejections (|alpha|<0.05, stale IR)
  float seedLastAlpha;   // last fitted alpha (0 until a fit completes)

  // Subtraction-gate state for the scrolling-monitor overlay (compressor-
  // threshold-style visual editor): the smoothed far-end power envelope the
  // gate tracks, and the resulting gate opening (0..1, block-smoothed).
  float gateEnv;
  float gateOpen;
};

#endif // ENUMS_H