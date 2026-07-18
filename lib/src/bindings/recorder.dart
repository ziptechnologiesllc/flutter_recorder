// ignore_for_file: avoid_positional_boolean_parameters

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_recorder/src/audio_data_container.dart';
import 'package:flutter_recorder/src/enums.dart';
import 'package:flutter_recorder/src/exceptions/exceptions.dart';
import 'package:flutter_recorder/src/filters/filters.dart';
import 'package:flutter_recorder/src/flutter_recorder.dart';
import 'package:meta/meta.dart';

export 'package:flutter_recorder/src/bindings/recorder_io.dart'
    if (dart.library.js_interop) 'package:flutter_recorder/src/bindings/recorder_web.dart';

/// One timed section of the audio callback (see capture.cpp section list).
class CallbackSection {
  const CallbackSection(this.name, this.lastMicros, this.maxMicros);

  final String name;

  /// Duration of this section in the most recent callback.
  final int lastMicros;

  /// Worst duration of this section since the last reset. Maxima are
  /// per-section (not from the same callback), so they name the hog but
  /// don't sum to [CallbackStats.maxMicros].
  final int maxMicros;

  @override
  String toString() => '$name=${(lastMicros / 1000).toStringAsFixed(2)}/'
      '${(maxMicros / 1000).toStringAsFixed(2)}ms';
}

/// Timing snapshot of the native audio callback. All durations in
/// microseconds. A callback whose duration exceeds [budgetMicros] makes the
/// device's next buffer late — that's an underrun, heard as a pop/click.
class CallbackStats {
  const CallbackStats({
    required this.lastMicros,
    required this.maxMicros,
    required this.budgetMicros,
    required this.overrunCount,
    required this.nearMissCount,
    required this.totalCount,
    this.sections = const [],
  });

  /// Duration of the most recent callback.
  final int lastMicros;

  /// Worst callback duration since the last reset.
  final int maxMicros;

  /// Buffer period — the deadline each callback must beat.
  final int budgetMicros;

  /// Callbacks that exceeded [budgetMicros] (→ underrun → pop).
  final int overrunCount;

  /// Callbacks over 80% of budget — not yet a pop, but close.
  final int nearMissCount;

  /// Total callbacks measured since the last reset.
  final int totalCount;

  /// Per-section breakdown (aec, ring, sched, mix, post) — last/max micros.
  final List<CallbackSection> sections;

  static const CallbackStats zero = CallbackStats(
    lastMicros: 0,
    maxMicros: 0,
    budgetMicros: 0,
    overrunCount: 0,
    nearMissCount: 0,
    totalCount: 0,
  );

  /// Fraction of the budget the worst callback used (1.0 == budget).
  double get worstLoad => budgetMicros > 0 ? maxMicros / budgetMicros : 0;

  @override
  String toString() => 'CallbackStats(last=${lastMicros}us '
      'max=${maxMicros}us budget=${budgetMicros}us '
      'overruns=$overrunCount/$totalCount nearMiss=$nearMissCount)';
}

/// LSAEC E5 gated-ERLE telemetry snapshot. Energies are windowed SUMS published
/// by the audio thread once per ~0.25 s; all dB math is done HERE, off the
/// real-time thread. "Far" = far-end (speaker) active — ERLE is gated to those
/// samples so a filter that eats the performer reads as LOW ERLE, not high.
class AecTelemetry {
  const AecTelemetry({
    required this.micEnergyFar,
    required this.outEnergyFar,
    required this.micEnergyAll,
    required this.outEnergyAll,
    required this.refEnergyAll,
    required this.farSamples,
    required this.totalSamples,
    required this.generation,
    required this.templateConfidence,
    required this.freezeCount,
    required this.isSeeding,
    required this.overCapacity,
    required this.govLeak,
    required this.govBoost,
    this.seedArms = 0,
    this.seedAborts = 0,
    this.seedLands = 0,
    this.gateEnv = 0,
    this.gateOpen = 1,
  });

  /// Σ mic² over far-end-active samples (gated ERLE numerator).
  final double micEnergyFar;

  /// Σ output² (final, post-cancellation) over far-end-active samples.
  final double outEnergyFar;

  /// Σ mic² over all samples in the window.
  final double micEnergyAll;

  /// Σ output² over all samples in the window.
  final double outEnergyAll;

  /// Σ reference² over all samples in the window.
  final double refEnergyAll;

  /// Far-end-active samples in the window.
  final int farSamples;

  /// Total samples in the window.
  final int totalSamples;

  /// Increments each published window (detects a stalled audio thread).
  final int generation;

  /// Mean per-phase LSAEC template confidence, 0..1 (0 = not in LSAEC mode
  /// or template never learned; 1 = every phase at max confidence).
  final double templateConfidence;

  /// E3 double-talk-freeze counter (monotonic, blocks skipped as near-end).
  final int freezeCount;

  /// True while a convergence-seed capture/compute/apply is in flight.
  final bool isSeeding;

  /// True when the loop period exceeds the 16s template capacity —
  /// cancellation is OFF (pure passthrough) until the period drops back
  /// under the cap. See SynchronousEchoTemplate::isOverCapacity().
  final bool overCapacity;

  /// Spectral governor's last coherence-leak reading.
  final double govLeak;

  /// Spectral governor's current learning-rate boost (1.0 = no boost).
  final double govBoost;

  /// Convergence-seed lifecycle counters (monotonic). aborts >> lands means
  /// mix-change notifies keep killing the one-period reference capture, so
  /// convergence is riding pure per-pass EMA (the "takes forever" mode).
  final int seedArms;
  final int seedAborts;
  final int seedLands;

  /// Subtraction-gate state for the scrolling-monitor overlay: smoothed
  /// far-end power envelope and the resulting gate opening (0..1).
  final double gateEnv;
  final double gateOpen;

  static const AecTelemetry zero = AecTelemetry(
    micEnergyFar: 0,
    outEnergyFar: 0,
    micEnergyAll: 0,
    outEnergyAll: 0,
    refEnergyAll: 0,
    farSamples: 0,
    totalSamples: 0,
    generation: 0,
    templateConfidence: 0,
    freezeCount: 0,
    isSeeding: false,
    overCapacity: false,
    govLeak: 0,
    govBoost: 1.0,
  );

  /// Fraction of the window where the speaker (far end) was actually playing.
  double get farActiveFraction =>
      totalSamples > 0 ? farSamples / totalSamples : 0;

  /// Gated echo-return-loss in dB: how much quieter the recorded output is than
  /// the mic input, measured ONLY over far-end-active samples. Higher = better
  /// cancellation; negative means the filter is ADDING energy (divergence).
  /// Null when there isn't enough far-end signal to judge.
  double? get gatedErleDb {
    if (farSamples == 0 || micEnergyFar <= 0 || outEnergyFar <= 0) return null;
    return 10.0 * (math.log(micEnergyFar / outEnergyFar) / math.ln10);
  }

  @override
  String toString() {
    final erle = gatedErleDb;
    final erleStr = erle == null ? 'n/a' : '${erle.toStringAsFixed(1)}dB';
    return 'AecTelemetry(gatedERLE=$erleStr '
        'farActive=${(farActiveFraction * 100).toStringAsFixed(0)}% '
        'conf=${(templateConfidence * 100).toStringAsFixed(0)}% '
        'freeze=$freezeCount '
        '${isSeeding ? 'SEEDING ' : ''}'
        '${overCapacity ? 'OVER-CAPACITY ' : ''}'
        'gov(leak=${govLeak.toStringAsFixed(2)},boost=${govBoost.toStringAsFixed(2)}) '
        'gen=$generation)';
  }
}

/// Use this class to _capture_ audio (such as from a microphone).
abstract class RecorderImpl {
  /// The device ID used to initialize the device.
  int? deviceID;

  ///  PCM format used to initialize the device.
  PCMFormat? format;

  /// Sample rate used to initialize the device.
  int? sampleRate;

  /// Channels used to initialize the device.
  RecorderChannels? channels;

  /// Android input preset used to initialize the device.
  AndroidInputPreset? androidInputPreset;

  /// Controller to listen to silence changed event.
  late final StreamController<SilenceState> silenceChangedEventController =
      StreamController.broadcast();

  /// Stream of silence state changes.
  Stream<SilenceState> get silenceChangedEvents =>
      silenceChangedEventController.stream;

  /// Controller for audio data types.
  late final uint8ListController =
      StreamController<AudioDataContainer>.broadcast();

  /// Streams for audio data types.
  Stream<AudioDataContainer> get uint8ListStream => uint8ListController.stream;

  /// Stream of AEC statistics (max attenuation, correlation, ERL).
  Stream<AecStats> get aecStatsStream;

  /// Set the AEC statistics callback.
  Future<void> setAecStatsCallback() async {}

  /// Stream of recording stopped events (fired from native when auto-stop occurs).
  Stream<RecordingStoppedEvent> get recordingStoppedStream;

  /// Set the recording stopped callback.
  Future<void> setRecordingStoppedCallback() async {}

  /// Stream of recording started events (fired from native when recording starts).
  Stream<RecordingStartedEvent> get recordingStartedStream;

  /// Set the recording started callback.
  Future<void> setRecordingStartedCallback() async {}

  /// Stream of looper playback started events (fired from worker thread when loop playback starts).
  Stream<LooperPlaybackStartedEvent> get looperPlaybackStartedStream;

  /// Set the looper playback started callback.
  Future<void> setLooperPlaybackStartedCallback() async {}

  /// Set Dart functions to call when an event occurs.
  @mustBeOverridden
  Future<void> setDartEventCallbacks();

  /// Enable or disable silence detection.
  ///
  /// [enable] wheter to enable or disable silence detection. Default to false.
  /// [onSilenceChanged] callback when silence state is changed.
  @mustBeOverridden
  void setSilenceDetection({
    required bool enable,
    SilenceCallback? onSilenceChanged,
  });

  /// Set silence threshold in dB.
  ///
  /// [silenceThresholdDb] the silence threshold in dB. A volume under this
  /// value is considered to be silence. Default to -40.
  ///
  /// Note on dB value:
  /// - Decibels (dB) are a relative measure. In digital audio, there is
  /// no 'absolute 0 dB level' that corresponds to absolute silence.
  /// - The 0 dB level is usually defined as the maximum possible signal level,
  /// i.e., the maximum amplitude of the signal that the system can handle
  /// without distortion.
  /// - Negative dB values indicate that the signal's energy is lower compared
  /// to this maximum.
  @mustBeOverridden
  void setSilenceThresholdDb(double silenceThresholdDb);

  /// Set silence duration in seconds.
  ///
  /// [silenceDuration] the duration of silence in seconds. If the volume
  /// remains silent for this duration, the callback will be triggered. Default
  /// to 2 seconds.
  @mustBeOverridden
  void setSilenceDuration(double silenceDuration);

  /// Set seconds of audio to write before starting recording again after
  /// silence.
  ///
  /// [secondsOfAudioToWriteBefore] seconds of audio to write occurred before
  /// starting recording againg after silence. Default to 0 seconds.
  /// ```text
  /// |*** silence ***|******** recording *********|
  ///                 ^ start of recording
  ///             ^ secondsOfAudioToWriteBefore (write some before silence ends)
  /// ```
  @mustBeOverridden
  void setSecondsOfAudioToWriteBefore(double secondsOfAudioToWriteBefore);

  /// List available input devices. Useful on desktop to choose
  /// which input device to use.
  @mustBeOverridden
  List<CaptureDevice> listCaptureDevices();

  /// Initialize input device with [deviceID].
  ///
  /// [captureOnly] - If true, use capture-only mode (no playback output).
  /// Use this when SoLoud has its own playback device to avoid two competing
  /// playback streams. If false, use duplex mode for slave mode where the
  /// recorder drives SoLoud's output through its callback.
  ///
  /// Thows [RecorderInitializeFailedException] if something goes wrong, ie. no
  /// device found with [deviceID] id.
  @mustBeOverridden
  @mustCallSuper
  void init({
    required int deviceID,
    required PCMFormat format,
    required int sampleRate,
    required RecorderChannels channels,
    AndroidInputPreset? androidInputPreset,
    bool captureOnly = false,
  }) {
    this.deviceID = deviceID;
    this.format = format;
    this.sampleRate = sampleRate;
    this.channels = channels;
    this.androidInputPreset = androidInputPreset;
  }

  /// Dispose capture device.
  @mustBeOverridden
  @mustCallSuper
  void deinit() {
    deviceID = null;
    format = null;
    sampleRate = null;
    channels = null;
    androidInputPreset = null;
  }

  /// Whether the device is initialized.
  @mustBeOverridden
  bool isDeviceInitialized();

  /// Whether the device is started.
  @mustBeOverridden
  bool isDeviceStarted();

  /// Start the device.
  ///
  /// Throws [RecorderNotInitializedException].
  /// Throws [RecorderFailedToStartDeviceException].
  @mustBeOverridden
  void start();

  /// Stop the device.
  @mustBeOverridden
  void stop();

  /// Start streaming data.
  @mustBeOverridden
  void startStreamingData();

  /// Stop streaming data.
  @mustBeOverridden
  void stopStreamingData();

  /// Start recording.
  ///
  /// Throws [RecorderNotInitializedException].
  /// Throws [RecorderFailedToInitializeRecordingException].
  @mustBeOverridden
  void startRecording(String path);

  /// Pause recording.
  @mustBeOverridden
  void setPauseRecording({required bool pause});

  /// Stop recording.
  @mustBeOverridden
  void stopRecording();

  /// Smooth FFT data.
  ///
  /// When new data is read and the values are decreasing, the new value will be
  /// decreased with an amplitude between the old and the new value.
  /// This will resul on a less shaky visualization.
  /// [smooth] must be in the [0.0 ~ 1.0] range.
  /// 0 = no smooth, values istantly get their new value.
  /// 1 = values don't get down when they reach their max value.
  /// the new value is calculated with:
  /// newFreq = smooth * oldFreq + (1 - smooth) * newFreq
  @mustBeOverridden
  void setFftSmoothing(double smooth);

  /// Enable or disable low-latency audio monitoring (input passthrough to output).
  @mustBeOverridden
  void setMonitoring(bool enabled);

  /// Set monitoring mode: 0=stereo, 1=leftMono, 2=rightMono, 3=mono.
  @mustBeOverridden
  void setMonitoringMode(int mode);

  // ///////////////////////
  // Filter debug stats
  // ///////////////////////

  /// Get the number of times filter processing was skipped due to lock contention.
  @mustBeOverridden
  int getFilterMissCount();

  /// Get the number of times filter processing completed successfully.
  @mustBeOverridden
  int getFilterProcessCount();

  /// Reset filter stats counters (call at session start).
  @mustBeOverridden
  void resetFilterStats();

  /// Conveninet way to get FFT data. Return a 256 float array containing
  /// FFT data in the range [-1.0, 1.0] not clamped.
  ///
  /// If also wave data is needed consider using [getTexture] or [getTexture2D].
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  @mustBeOverridden
  Float32List getFft({bool alwaysReturnData = true});

  /// Return a 256 float array containing wave data in the range [-1.0, 1.0].
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  @mustBeOverridden
  Float32List getWave({bool alwaysReturnData = true});

  /// Get the audio data representing an array of 256 floats FFT data and
  /// 256 float of wave data.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  @mustBeOverridden
  Float32List getTexture({bool alwaysReturnData = true});

  /// Get the audio data representing an array of 256 floats FFT data and
  /// 256 float of wave data.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  @mustBeOverridden
  Float32List getTexture2D({bool alwaysReturnData = true});

  /// Get the current volume in dB. Returns -100 if the capture is not inited.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  @mustBeOverridden
  double getVolumeDb();

  // ///////////////////////
  //   GETTERS
  // ///////////////////////

  /// Get the actual sample rate.
  @mustBeOverridden
  int getSampleRate() {
    throw UnimplementedError();
  }

  /// Get the actual capture channels.
  @mustBeOverridden
  int getCaptureChannels() {
    throw UnimplementedError();
  }

  /// Get the actual playback channels.
  @mustBeOverridden
  int getPlaybackChannels() {
    throw UnimplementedError();
  }

  /// Get the actual capture format.
  @mustBeOverridden
  int getCaptureFormat() {
    throw UnimplementedError();
  }

  /// Get the actual playback format.
  @mustBeOverridden
  int getPlaybackFormat() {
    throw UnimplementedError();
  }

  // ///////////////////////
  //   FILTERS
  // ///////////////////////

  /// Check if a filter is active.
  /// Return -1 if the filter is not active or its index.
  @mustBeOverridden
  int isFilterActive(RecorderFilterType filterType);

  /// Add a filter.
  ///
  /// Throws [RecorderFilterAlreadyAddedException] if the filter has already
  /// been added.
  /// Throws [RecorderFilterNotFoundException] if the filter could not be found.
  @mustBeOverridden
  void addFilter(RecorderFilterType filterType);

  /// Remove a filter.
  ///
  /// Throws [RecorderFilterNotFoundException] if trying to a non active
  /// filter.
  @mustBeOverridden
  CaptureErrors removeFilter(RecorderFilterType filterType);

  /// Get filter param names.
  @mustBeOverridden
  List<String> getFilterParamNames(RecorderFilterType filterType);

  /// Set filter param value.
  @mustBeOverridden
  void setFilterParamValue(
    RecorderFilterType filterType,
    int attributeId,
    double value,
  );

  /// Get filter param value.
  @mustBeOverridden
  double getFilterParamValue(RecorderFilterType filterType, int attributeId);

  // ///////////////////////
  //   SLAVE MODE
  // ///////////////////////

  /// Check if slave audio is ready (first callback has run successfully).
  /// This is used to wait for the audio pipeline to stabilize before calibration.
  @mustBeOverridden
  bool isSlaveAudioReady();

  bool wasDuplexDenied();

  // ///////////////////////
  //   Phase 2e: Ableton Link
  // ///////////////////////
  //
  // Direct path to the native AudioEngine's AbletonLinkClock. `setEnabled`
  // is NOT realtime-safe (Link spins up its network thread on enable);
  // Dart must call from the main isolate, never from an audio callback.

  /// Enable / disable Link-session participation.
  @mustBeOverridden
  void linkSetEnabled(bool enabled);

  /// Whether Link is currently enabled in the native engine.
  @mustBeOverridden
  bool linkIsEnabled();

  /// Number of peers in the Link session. 0 when disabled or solo.
  @mustBeOverridden
  int linkNumPeers();

  // ///////////////////////
  //   Audio-callback profiling (pops/clicks hunt)
  // ///////////////////////

  /// Snapshot of the native audio callback's recent timing. Polled by the
  /// dev overlay to spot underruns (callbacks that blew the buffer-period
  /// budget — the proximate cause of an audible pop).
  @mustBeOverridden
  CallbackStats getCallbackStats();

  /// Zero the max + counters (keeps last/budget). Call before a test run so
  /// the overrun count reflects just that run.
  @mustBeOverridden
  void resetCallbackStats();

  // ///////////////////////
  //   AEC (Adaptive Echo Cancellation)
  // ///////////////////////

  /// Create the AEC reference buffer.
  /// Returns a pointer to the buffer that should be passed to SoLoud.
  @mustBeOverridden
  int aecCreateReferenceBuffer(int sampleRate, int channels);

  /// Destroy the AEC reference buffer.
  @mustBeOverridden
  void aecDestroyReferenceBuffer();

  /// Get the AEC output callback function pointer.
  /// This should be passed to SoLoud to receive playback audio.
  @mustBeOverridden
  int aecGetOutputCallback();

  /// Reset the AEC buffer (e.g., when switching audio configurations).
  @mustBeOverridden
  void aecResetBuffer();

  /// Enable/disable AEC reference buffer writes.
  /// When disabled, saves CPU when AEC is not needed.
  @mustBeOverridden
  void aecSetEnabled(bool enabled);

  /// Check if AEC reference buffer is enabled.
  @mustBeOverridden
  bool aecIsEnabled();

  /// Set AEC Mode (0=Bypass, 1=Algo, 2=Neural, 3=Hybrid)
  @mustBeOverridden
  void aecSetMode(AecMode mode);

  /// Get current AEC Mode
  @mustBeOverridden
  AecMode aecGetMode();

  /// Load neural model by type
  @mustBeOverridden
  bool aecLoadNeuralModel(NeuralModelType type, String assetBasePath);

  /// Get currently loaded neural model type
  @mustBeOverridden
  NeuralModelType aecGetLoadedNeuralModel();

  /// Enable/disable neural post-filter
  @mustBeOverridden
  void aecSetNeuralEnabled(bool enabled);

  /// Check if neural post-filter is enabled
  @mustBeOverridden
  bool aecIsNeuralEnabled();

  // ==================== AEC CALIBRATION ====================

  /// Generate calibration audio signal.
  /// [signalType] determines the signal:
  ///   - chirp: Logarithmic sine sweep (default)
  ///   - click: Impulse train (better for transients)
  /// Returns WAV data as Uint8List that can be loaded into SoLoud.
  @mustBeOverridden
  Uint8List aecGenerateCalibrationSignal(
    int sampleRate,
    int channels, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  });

  /// Start capturing microphone samples for calibration analysis.
  @mustBeOverridden
  void aecStartCalibrationCapture(int maxSamples);

  /// Stop capturing samples for calibration.
  @mustBeOverridden
  void aecStopCalibrationCapture();

  /// Capture signals from both reference and mic buffers for analysis.
  @mustBeOverridden
  void aecCaptureForAnalysis();

  /// Run cross-correlation analysis on captured signals.
  @mustBeOverridden
  AecCalibrationResult aecRunCalibrationAnalysis(int sampleRate);

  /// Reset calibration state.
  @mustBeOverridden
  void aecResetCalibration();

  /// Run calibration analysis with impulse response computation.
  /// Returns result including impulse length (call aecGetImpulseResponse to get data).
  @mustBeOverridden
  AecCalibrationResultWithImpulse aecRunCalibrationWithImpulse(int sampleRate);

  /// Get stored impulse response from last calibration.
  /// Returns Float32List of coefficients.
  @mustBeOverridden
  Float32List aecGetImpulseResponse(int maxLength);

  /// Apply stored impulse response to AEC filter.
  @mustBeOverridden
  void aecApplyImpulseResponse();

  /// Apply externally-provided impulse response coefficients directly — for
  /// restoring a persisted calibration on cold start, where there's no
  /// live-just-computed impulse response to re-apply.
  @mustBeOverridden
  void aecSetImpulseResponse(Float32List coeffs);

  /// Get captured reference signal for visualization.
  @mustBeOverridden
  Float32List aecGetCalibrationRefSignal(int maxLength);

  /// Get captured mic signal for visualization.
  @mustBeOverridden
  Float32List aecGetCalibrationMicSignal(int maxLength);

  /// Force speaker output on iOS (useful for measurement mode).
  @mustBeOverridden
  void iosForceSpeakerOutput(bool enabled);

  // ==================== AEC TESTING ====================

  /// Start capturing test signals (raw mic + cancelled output).
  /// Call this BEFORE playing the test audio.
  @mustBeOverridden
  void aecStartTestCapture(int maxSamples);

  /// Stop capturing test signals.
  /// Call this AFTER test audio has finished playing.
  @mustBeOverridden
  void aecStopTestCapture();

  /// Run analysis on captured test signals.
  /// Returns test results with cancellation metrics.
  @mustBeOverridden
  AecTestResult aecRunTest(int sampleRate);

  /// Get captured raw mic signal (before AEC) for visualization.
  @mustBeOverridden
  Float32List aecGetTestMicSignal(int maxLength);

  /// Get captured cancelled signal (after AEC) for visualization.
  @mustBeOverridden
  Float32List aecGetTestCancelledSignal(int maxLength);

  /// Reset test data.
  @mustBeOverridden
  void aecResetTest();

  // ==================== VSS-NLMS PARAMETER CONTROL ====================

  /// Set VSS-NLMS maximum step size (0.0-1.0). Set to 0 to freeze weights.
  @mustBeOverridden
  void aecSetVssMuMax(double mu);

  /// Set VSS-NLMS leakage factor (0.99-1.0). Set to 1.0 for no decay.
  @mustBeOverridden
  void aecSetVssLeakage(double lambda);

  /// Set VSS-NLMS smoothing factor (0.9-0.999).
  @mustBeOverridden
  void aecSetVssAlpha(double alpha);

  /// Get current VSS-NLMS maximum step size.
  @mustBeOverridden
  double aecGetVssMuMax();

  /// LSAEC E5: read the lock-free gated-ERLE telemetry snapshot. Cheap; safe to
  /// poll at ~10 Hz. Returns [AecTelemetry.zero] when the AEC filter is absent.
  @mustBeOverridden
  AecTelemetry aecGetTelemetry();

  /// LSAEC: tell the template the audible mix changed (e.g. a one-shot
  /// source like the metronome click toggled on/off) even though no loop-
  /// period change occurred. Mute/unmute/pause/stop on a registered SoLoud
  /// voice already trigger this internally via the native PendingAction
  /// path; call this directly for anything that path doesn't cover. Cheap;
  /// safe to call often — a no-op while a seed job is already in flight.
  @mustBeOverridden
  void aecNotifyReferenceChanged();

  /// LSAEC per-track exact subtraction: register a track's own known audio
  /// (mono float samples, already phase-aligned to the loop's phase-0
  /// origin, length equal to the CURRENT composite loop period) so its echo
  /// contribution can be computed once, off-thread — an exact arithmetic
  /// edit on mute/unmute instead of a reactive suppression heuristic or a
  /// reseed race. Native copies the data immediately; safe to discard
  /// [audioMono] right after this call returns. No-op if LSAEC isn't
  /// active. The mute/unmute/pause/stop toggle itself fires natively (same
  /// fire-frame as the SoLoud setter) — nothing further to call from Dart
  /// once a track is registered.
  @mustBeOverridden
  void aecRegisterTrackAudio(int trackIndex, Float32List audioMono);

  /// Toggle a registered track's echo contribution in/out of the live LSAEC
  /// template (exact edit; no-op until the contribution finishes computing).
  @mustBeOverridden
  void aecSetTrackActive(int trackIndex, bool active);

  /// Release a deleted track's per-track AEC slot back to the pool. Call
  /// whenever a loop is removed so a long session doesn't exhaust the
  /// fixed-size slot table with contributions for loops that no longer
  /// exist. No-op if the track was never registered.
  @mustBeOverridden
  void aecReleaseTrackContribution(int trackIndex);

  /// Enable/disable the LSAEC Stage-2 nonlinear HF residual-echo suppressor —
  /// the post-filter that cleans the high-frequency ghost / metronome click
  /// linear cancellation can't reach. For on-device A/B (linear-only vs.
  /// linear+suppressor). Default enabled in the engine.
  @mustBeOverridden
  void aecSetResidualSuppressor(bool enabled);

  /// Whether the Stage-2 HF residual-echo suppressor is currently enabled.
  @mustBeOverridden
  bool aecGetResidualSuppressor();

  /// Live-tune the LSAEC subtraction gate: far-end envelope attack/release
  /// (ms) and soft-knee floor (dB power). The release+knee pair sets how
  /// long cancellation hangs on after content stops (the audible "tail") —
  /// a per-room, by-ear judgment, hand-tuned from the AEC debug panel.
  @mustBeOverridden
  void aecSetSubGateTuning(double attackMs, double releaseMs, double floorDb);

  /// Get current VSS-NLMS leakage factor.
  @mustBeOverridden
  double aecGetVssLeakage();

  /// Get current VSS-NLMS smoothing factor.
  @mustBeOverridden
  double aecGetVssAlpha();

  // ==================== AEC FILTER LENGTH CONTROL ====================

  /// Set AEC filter length (2048, 4096, 8192 recommended).
  @mustBeOverridden
  void aecSetFilterLength(int length);

  /// Get current AEC filter length.
  @mustBeOverridden
  int aecGetFilterLength();

  // ==================== AEC CALIBRATION LOGGING ====================

  /// Get the calibration log buffer containing debug messages.
  @mustBeOverridden
  String aecGetCalibrationLog();

  /// Clear the calibration log buffer.
  @mustBeOverridden
  void aecClearCalibrationLog();

  // ==================== AEC POSITION-BASED SYNC ====================

  /// Get total frames written to reference buffer (output side counter).
  @mustBeOverridden
  int aecGetOutputFrameCount();

  /// Get total frames captured by recorder (input side counter).
  @mustBeOverridden
  int aecGetCaptureFrameCount();

  /// Record frame counters at calibration start.
  /// Call this when calibration signal starts playing.
  @mustBeOverridden
  void aecRecordCalibrationFrameCounters();

  /// Set the calibrated offset for position-based sync.
  /// offset = (captureAtStart - outputAtStart) + acousticDelaySamples
  @mustBeOverridden
  void aecSetCalibratedOffset(int offset);

  /// Get the current calibrated offset.
  @mustBeOverridden
  int aecGetCalibratedOffset();

  // ==================== ALIGNED CALIBRATION CAPTURE ====================

  /// Start capturing aligned ref+mic from AEC processAudio.
  /// This captures frame-aligned signals for accurate delay estimation.
  @mustBeOverridden
  void aecStartAlignedCalibrationCapture(int maxSamples);

  /// Stop aligned calibration capture.
  @mustBeOverridden
  void aecStopAlignedCalibrationCapture();

  /// Run calibration analysis on aligned buffers and apply impulse response.
  /// [signalType] should match what was used for generation.
  /// Returns the calibration result with delay and impulse info.
  @mustBeOverridden
  AecCalibrationResultWithImpulse aecRunAlignedCalibrationWithImpulse(
    int sampleRate, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  });

  // ==================== NATIVE AUDIO SINK ====================
  // Direct native-to-native streaming (bypasses Dart main thread)

  /// Set native audio sink for direct recorder-to-player streaming.
  /// [callbackAddress] and [userDataAddress] come from SoLoud's
  /// configureNativeAudioSinkRaw().
  @mustBeOverridden
  void setNativeAudioSink(int callbackAddress, int userDataAddress);

  /// Check if native audio sink is active.
  @mustBeOverridden
  bool isNativeAudioSinkActive();

  /// Disable native audio sink (data flows through Dart again).
  @mustBeOverridden
  void disableNativeAudioSink();

  /// Inject preroll audio from ring buffer into SoLoud stream via native path.
  /// This reads [frameCount] frames from the ring buffer and sends them
  /// directly to the native audio sink callback.
  @mustBeOverridden
  void injectPreroll(int frameCount);

  /// Set the looper bridge function pointer for direct native-to-SoLoud playback.
  /// [funcAddress] is the address of the looper_loadAndPlayLoop function.
  @mustBeOverridden
  void setLooperBridge(int funcAddress);

  /// Clear the looper bridge.
  @mustBeOverridden
  void clearLooperBridge();

  // ==================== NATIVE SCHEDULER ====================
  // Sample-accurate timing for recording start/stop in audio callback

  /// Reset the native scheduler state.
  @mustBeOverridden
  void schedulerReset();

  /// Set base loop parameters for quantization.
  /// [loopFrames] is the loop length in frames.
  /// [loopStartFrame] is the global frame when the loop started.
  @mustBeOverridden
  void schedulerSetBaseLoop(int loopFrames, int loopStartFrame);

  /// Clear base loop (free recording mode).
  @mustBeOverridden
  void schedulerClearBaseLoop();

  /// Schedule quantized recording start.
  /// Returns event ID (0 if failed to schedule).
  @mustBeOverridden
  int schedulerScheduleStart(String path);

  /// Schedule quantized recording stop.
  /// [startFrame] is when recording started (for multi-loop calculation).
  /// Returns event ID (0 if failed to schedule).
  @mustBeOverridden
  int schedulerScheduleStop(int startFrame);

  /// Cancel a scheduled event by ID.
  /// Returns true if event was cancelled.
  @mustBeOverridden
  bool schedulerCancelEvent(int eventId);

  /// Cancel all pending events.
  @mustBeOverridden
  void schedulerCancelAll();

  /// Poll for fired event notification.
  /// Returns null if no notification available.
  @mustBeOverridden
  SchedulerNotification? schedulerPollNotification();

  /// Check if there are pending notifications.
  @mustBeOverridden
  bool schedulerHasNotifications();

  /// Get current global frame position.
  @mustBeOverridden
  int schedulerGetGlobalFrame();

  /// Get base loop length in frames.
  @mustBeOverridden
  int schedulerGetBaseLoopFrames();

  /// Get next loop boundary frame.
  @mustBeOverridden
  int schedulerGetNextLoopBoundary();

  /// Set latency compensation in frames (applied at recording start).
  @mustBeOverridden
  void schedulerSetLatencyCompensation(int frames);

  /// Get latency compensation in frames.
  @mustBeOverridden
  int schedulerGetLatencyCompensation();

  /// Set auto-stop enabled (when true, STOP is scheduled upfront with START).
  @mustBeOverridden
  void schedulerSetAutoStop(bool enabled);

  /// Get auto-stop enabled state.
  @mustBeOverridden
  bool schedulerIsAutoStopEnabled();

  /// Set how many base-loop cycles an overdub records before the upfront STOP
  /// (1 = one cycle, N = N cycles, 0 = manual stop — still quantized).
  @mustBeOverridden
  void schedulerSetRecordCycles(int cycles);

  /// Get the configured record-cycles multiplier.
  @mustBeOverridden
  int schedulerGetRecordCycles();

  // ==================== AUTO-RECORD ====================
  // Hands-free first-loop capture: long-press to arm, the first detected onset
  // becomes the loop downbeat (lead-in silence trimmed via the ring buffer).

  /// Arm auto-record. The next detected onset becomes the loop downbeat.
  /// With [barCount] > 0 and [framesPerBar] > 0 the take auto-stops at exactly
  /// `start + barCount * framesPerBar`. [framesPerBar] == 0 means tempo
  /// unknown (preset auto-stop skipped). [sampleRate] == 0 keeps the current.
  /// If [measureAmbient] is true, keep measuring the ambient level (don't
  /// listen for onsets) until [endAutoRecordMeasure] — the "hold the button"
  /// model: the longer you hold, the better the ambient/threshold estimate.
  @mustBeOverridden
  void armAutoRecord(String wavPath, int barCount, int framesPerBar,
      int sampleRate, bool measureAmbient);

  /// End the ambient-measure window (held button released): lock the trigger to
  /// the measured ambient level and start listening for onsets.
  @mustBeOverridden
  void endAutoRecordMeasure();

  /// Disarm auto-record (an in-progress take is left for the normal stop path).
  @mustBeOverridden
  void disarmAutoRecord();

  /// Auto-record state: 0 = idle, 1 = armed (waiting for onset / measuring), 2 = recording.
  @mustBeOverridden
  int getAutoRecordState();

  /// True while the armed detector is still measuring ambient (button held).
  @mustBeOverridden
  bool isAutoRecordMeasuringAmbient();

  /// Best current tempo estimate in BPM (0 until the estimator locks).
  @mustBeOverridden
  double getAutoRecordTempoBpm();

  /// Current measured noise floor in dBFS (for the UI threshold line).
  @mustBeOverridden
  double getAutoRecordNoiseFloorDb();

  /// Current onset trigger level in dBFS = noiseFloorDb + onsetThresholdDb.
  @mustBeOverridden
  double getAutoRecordTriggerLevelDb();

  /// Onset-detector sensitivity: dB above the (ambient) noise floor that counts
  /// as an attack. Lower = more sensitive (soft-onset instruments).
  @mustBeOverridden
  void setAutoRecordOnsetThresholdDb(double db);

  // ==================== NATIVE RING BUFFER ====================
  // Latency compensation via continuous capture with pre-roll

  /// Create/configure the native ring buffer for latency compensation.
  /// [capacitySeconds] How many seconds of audio to keep (typically 5).
  /// [sampleRate] Sample rate in Hz.
  /// [channels] Number of channels (1=mono, 2=stereo).
  @mustBeOverridden
  void createRingBuffer(int capacitySeconds, int sampleRate, int channels);

  /// Destroy/reset the native ring buffer.
  @mustBeOverridden
  void destroyRingBuffer();

  /// Read pre-roll samples for latency compensation.
  /// [frameCount] Number of frames to read.
  /// [rewindFrames] How many frames back in time to start reading.
  /// Returns Float32List with interleaved samples.
  @mustBeOverridden
  Float32List readPreRoll(int frameCount, int rewindFrames);

  /// Get current audio level in dB (RMS).
  /// Calculated continuously in the native audio callback.
  @mustBeOverridden
  double getAudioLevelDb();

  /// Get total frames written to the ring buffer.
  @mustBeOverridden
  int getRingBufferFramesWritten();

  /// Get available frames in the ring buffer.
  @mustBeOverridden
  int getRingBufferAvailable();

  /// Reset the ring buffer (clear all data).
  @mustBeOverridden
  void resetRingBuffer();

  /// Get recorded audio as WAV data from native memory.
  /// Returns a VIEW of native memory - no copy! Very fast.
  /// Pointer valid until next recording or freeRecordedAudio.
  @mustBeOverridden
  Uint8List? getRecordedWav();

  /// Get WAV size (for checking if data available).
  @mustBeOverridden
  int getRecordedWavSize();

  /// Free the recorded audio and WAV buffers in native memory.
  /// Call this after you're done with the audio data to release memory.
  @mustBeOverridden
  void freeRecordedAudio();
}
