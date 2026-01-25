// ignore_for_file: omit_local_variable_types
// ignore_for_file: avoid_positional_boolean_parameters

import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter_recorder/src/audio_data_container.dart';
import 'package:flutter_recorder/src/bindings/recorder.dart';
import 'package:flutter_recorder/src/enums.dart';
import 'package:flutter_recorder/src/exceptions/exceptions.dart';
import 'package:flutter_recorder/src/filters/filters.dart';
import 'package:logging/logging.dart';
import 'package:meta/meta.dart';

export 'audio_data_container.dart';
export 'enums.dart';
export 'exceptions/exceptions.dart';
export 'filters/filters.dart';

/// Callback when silence state is changed.
typedef SilenceCallback = void Function(bool isSilent, double decibel);

/// Silence state.
typedef SilenceState = ({bool isSilent, double decibel});

/// Result of AEC calibration analysis.
class AecCalibrationResult {
  /// Optimal delay in milliseconds between speaker output and microphone input.
  /// Float precision preserves sub-millisecond accuracy for phase alignment.
  final double delayMs;

  /// Echo gain factor (ratio of echo to original signal, 0-1).
  final double echoGain;

  /// Peak correlation coefficient (quality metric, higher = better match).
  final double correlation;

  /// Whether the calibration was successful.
  final bool success;

  const AecCalibrationResult({
    required this.delayMs,
    required this.echoGain,
    required this.correlation,
    required this.success,
  });

  @override
  String toString() =>
      'AecCalibrationResult(delayMs: $delayMs, echoGain: ${echoGain.toStringAsFixed(3)}, '
      'correlation: ${correlation.toStringAsFixed(3)}, success: $success)';
}

/// Result of AEC calibration analysis with impulse response info.
class AecCalibrationResultWithImpulse extends AecCalibrationResult {
  /// Length of the computed impulse response.
  final int impulseLength;

  /// Calibrated offset for position-based sync.
  /// Use this with aecSetCalibratedOffset() to enable sample-accurate sync.
  /// Formula: captureFrame - calibratedOffset = corresponding outputFrame
  final int calibratedOffset;

  const AecCalibrationResultWithImpulse({
    required super.delayMs,
    required super.echoGain,
    required super.correlation,
    required super.success,
    required this.impulseLength,
    required this.calibratedOffset,
  });

  @override
  String toString() =>
      'AecCalibrationResultWithImpulse(delayMs: $delayMs, echoGain: ${echoGain.toStringAsFixed(3)}, '
      'correlation: ${correlation.toStringAsFixed(3)}, success: $success, impulseLength: $impulseLength, '
      'calibratedOffset: $calibratedOffset)';
}

/// Scheduler action type (matches native SchedulerAction enum).
enum SchedulerAction {
  none(0),
  startRecording(1),
  stopRecording(2),
  startPlayback(3),
  stopPlayback(4);

  const SchedulerAction(this.value);
  final int value;

  static SchedulerAction fromValue(int value) {
    return SchedulerAction.values.firstWhere(
      (e) => e.value == value,
      orElse: () => SchedulerAction.none,
    );
  }
}

/// Notification from native scheduler when an event fires.
class SchedulerNotification {
  /// Unique event ID (assigned when event was scheduled).
  final int eventId;

  /// Action that was executed.
  final SchedulerAction action;

  /// Global frame when event fired.
  final int firedAtFrame;

  /// How many frames late the event fired (0 = perfect, positive = late).
  final int latencyFrames;

  const SchedulerNotification({
    required this.eventId,
    required this.action,
    required this.firedAtFrame,
    required this.latencyFrames,
  });

  @override
  String toString() =>
      'SchedulerNotification(id: $eventId, action: $action, frame: $firedAtFrame, latency: $latencyFrames)';
}

/// Result of AEC test analysis.
/// Used to validate echo cancellation quality after calibration.
class AecTestResult {
  /// Energy of raw mic signal (before AEC) in dB.
  final double micEnergyDb;

  /// Energy of cancelled signal (after AEC) in dB.
  final double cancelledEnergyDb;

  /// How much was cancelled (micEnergyDb - cancelledEnergyDb).
  /// Positive values = good cancellation.
  final double cancellationDb;

  /// Correlation of reference with raw mic (before AEC).
  /// High value (~0.8+) indicates echo is present.
  final double correlationBefore;

  /// Correlation of reference with cancelled signal (after AEC).
  /// Should be low (~0) if cancellation is working.
  final double correlationAfter;

  /// Whether the test passed (cancellationDb > threshold).
  final bool passed;

  const AecTestResult({
    required this.micEnergyDb,
    required this.cancelledEnergyDb,
    required this.cancellationDb,
    required this.correlationBefore,
    required this.correlationAfter,
    required this.passed,
  });

  @override
  String toString() =>
      'AecTestResult(cancellationDb: ${cancellationDb.toStringAsFixed(1)}, '
      'correlationBefore: ${correlationBefore.toStringAsFixed(3)}, '
      'correlationAfter: ${correlationAfter.toStringAsFixed(3)}, '
      'passed: $passed)';
}

/// Use this class to _capture_ audio (such as from a microphone).
interface class Recorder {
  /// The private constructor of [Recorder]. This prevents developers from
  /// instantiating new instances.
  Recorder._();

  static final Logger _log = Logger('flutter_recorder.Recorder');

  /// The singleton instance of [Recorder]. Only one Recorder instance
  /// can exist in C++ land, so – for consistency and to avoid confusion
  /// – only one instance can exist in Dart land.
  ///
  /// Using this static field, you can get a hold of the single instance
  /// of this class from anywhere. This ability to access global state
  /// from anywhere can lead to hard-to-debug bugs, though, so it is
  /// preferable to encapsulate this and provide it through a facade.
  /// For example:
  ///
  /// ```dart
  /// final recordingController = MyRecordingController(Recorder.instance);
  ///
  /// // Now provide the recording controller to parts of the app that need it.
  /// // No other part of the codebase need import `package:flutter_recorder`.
  /// ```
  ///
  /// Alternatively, at least create a field with the single instance
  /// of [Recorder], and provide that (without the facade, but also without
  /// accessing [Recorder.instance] from different places of the app).
  /// For example:
  ///
  /// ```dart
  /// class _MyWidgetState extends State<MyWidget> {
  ///   Recorder? _recorder;
  ///
  ///   void _initializeRecording() async {
  ///     // The only place in the codebase that accesses Recorder.instance
  ///     // directly.
  ///     final recorder = Recorder.instance;
  ///     await recorder.initialize();
  ///
  ///     setState(() {
  ///       _recorder = recorder;
  ///     });
  ///   }
  ///
  ///   // ...
  /// }
  /// ```
  static final Recorder instance = Recorder._();

  /// This can be used to access all the available filter functionalities.
  ///
  /// ```dart
  /// final recorder = await Recorder.instance.init();
  /// ...
  /// /// activate the filter.
  ///recorder.filters.autoGainFilter.activate();
  ///
  /// /// Later on, deactivate it.
  /// recorder.filters.autoGainFilter.deactivate();
  /// ```
  ///
  /// It's possible to get and set filter parameters:
  /// ```dart
  /// /// Set
  /// recorder.filters.autoGainFilter.targetRms.value = 0.6;
  /// /// Get
  /// final targetRmsValue = recorder.filters.autoGainFilter.targetRms.value;
  /// ```
  ///
  /// It's possible to query filter parameters:
  /// ```dart
  /// final targetRms = recorder.filters.autoGainFilter.queryTargetRms;
  /// ```
  ///
  /// Now with `targetRms` you have access to:
  /// - `toString()` gives the "human readable" parameter name.
  /// - `min` which represent the minimum accepted value.
  /// - `max` which represent the maximum accepted value.
  /// - `def` which represent the default value.
  @experimental
  final filters = const Filters();

  static RecorderImpl? _mockImplementation;

  /// Set a mock implementation for testing.
  @visibleForTesting
  static void setMockImplementation(RecorderImpl mock) {
    _mockImplementation = mock;
  }

  final _recoreder = RecorderController();

  RecorderImpl get _impl => _mockImplementation ?? _recoreder.impl;

  /// Whether the device is initialized.
  bool _isInitialized = false;

  /// Whether the device is started.
  bool _isStarted = false;

  /// Currently used recorder configuration.
  PCMFormat _recorderFormat = PCMFormat.s16le;

  /// Listening to silence state changes.
  Stream<SilenceState> get silenceChangedEvents => _impl.silenceChangedEvents;

  /// Listen to audio data.
  ///
  /// The streaming must be enabled calling [startStreamingData].
  ///
  /// **NOTE**: Audio data must be processed as it is received. To optimize
  /// performance, the same memory is used to store data for all incoming
  /// streams, meaning the data will be overwritten. Therefore, you must copy
  /// the data if you need to populate a buffer. For example, when using
  /// **RxDart.bufferTime**, it will fill a **List** of `AudioDataContainer`
  /// objects, but when you attempt to read them, you will find that all
  /// the items contain the same data.
  Stream<AudioDataContainer> get uint8ListStream => _impl.uint8ListStream;

  /// Stream of AEC statistics (max attenuation, correlation, ERL).
  Stream<AecStats> get aecStatsStream => _impl.aecStatsStream;

  /// Stream of recording stopped events (fired from native when auto-stop occurs).
  Stream<RecordingStoppedEvent> get recordingStoppedStream =>
      _impl.recordingStoppedStream;

  /// Stream of recording started events (fired from native when recording starts).
  Stream<RecordingStartedEvent> get recordingStartedStream =>
      _impl.recordingStartedStream;

  /// Enable or disable silence detection.
  ///
  /// [enable] wheter to enable or disable silence detection. Default to false.
  /// [onSilenceChanged] callback when silence state is changed.
  void setSilenceDetection({
    required bool enable,
    SilenceCallback? onSilenceChanged,
  }) {
    _impl.setSilenceDetection(
      enable: enable,
      onSilenceChanged: onSilenceChanged,
    );
  }

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
  void setSilenceThresholdDb(double silenceThresholdDb) {
    _impl.setSilenceThresholdDb(silenceThresholdDb);
  }

  /// Set the value in seconds of silence after which silence is considered
  /// as such.
  ///
  /// [silenceDuration] the duration of silence in seconds. If the volume
  /// remains silent for this duration, the [SilenceCallback] callback will be
  /// triggered or the Stream [silenceChangedEvents] will emit silence state.
  /// Default to 2 seconds.
  void setSilenceDuration(double silenceDuration) {
    _impl.setSilenceDuration(silenceDuration);
  }

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
  void setSecondsOfAudioToWriteBefore(double secondsOfAudioToWriteBefore) {
    _impl.setSecondsOfAudioToWriteBefore(secondsOfAudioToWriteBefore);
  }

  /// List available input devices. Useful on desktop to choose
  /// which input device to use.
  List<CaptureDevice> listCaptureDevices() {
    final ret = _impl.listCaptureDevices();

    return ret;
  }

  /// Initialize input device with [deviceID].
  ///
  /// [deviceID] the id of the input device. If -1, the default OS input
  /// device is used.
  /// [format] PCM format. Default to [PCMFormat.s16le].
  /// [sampleRate] sample rate in Hz. Default to 22050.
  /// [channels] number of channels. Default to [RecorderChannels.mono].
  /// [captureOnly] If true, use capture-only mode (no playback output).
  /// Use this when SoLoud has its own playback device to avoid two competing
  /// playback streams causing audio quality issues. If false (default), use
  /// duplex mode for slave mode where the recorder drives SoLoud's output.
  ///
  /// Thows [RecorderInitializeFailedException] if something goes wrong, ie. no
  /// device found with [deviceID] id.
  Future<void> init({
    int deviceID = -1,
    PCMFormat format = PCMFormat.s16le,
    int sampleRate = 22050,
    RecorderChannels channels = RecorderChannels.mono,
    bool captureOnly = false,
  }) async {
    await _impl.setDartEventCallbacks();
    await _impl.setAecStatsCallback();
    await _impl.setRecordingStoppedCallback();
    await _impl.setRecordingStartedCallback();

    // Sets the [_isInitialized].
    // Usefult when the consumer use the hot restart and that flag
    // has been reset.
    isDeviceInitialized();

    if (_isInitialized) {
      _log.warning('init() called when the native device is already '
          'initialized. This is expected after a hot restart but not '
          "otherwise. If you see this in production logs, there's probably "
          'a bug in your code. You may have neglected to deinit() Recorder '
          'during the current lifetime of the app.');
      deinit();
    }

    _impl.init(
      deviceID: deviceID,
      format: format,
      sampleRate: sampleRate,
      channels: channels,
      captureOnly: captureOnly,
    );
    _isInitialized = true;

    // Update _recorderFormat to actual format chosen by the system
    // This is important for Android auto mode where format=unknown
    if (format == PCMFormat.unknown) {
      final actualFormat = _impl.getCaptureFormat();
      // Map miniaudio format ID to PCMFormat
      // miniaudio: unknown=0, u8=1, s16=2, s24=3, s32=4, f32=5
      _recorderFormat = switch (actualFormat) {
        1 => PCMFormat.u8,
        2 => PCMFormat.s16le,
        3 => PCMFormat.s24le,
        4 => PCMFormat.s32le,
        5 => PCMFormat.f32le,
        _ => PCMFormat.unknown,
      };
      _log.info(() => 'Auto-detected format: $actualFormat -> $_recorderFormat');
    } else {
      _recorderFormat = format;
    }
  }

  /// Dispose capture device.
  void deinit() {
    _isInitialized = false;
    stop();
    _impl.deinit();
  }

  /// Whether the device is initialized.
  bool isDeviceInitialized() {
    // ignore: join_return_with_assignment
    _isInitialized = _impl.isDeviceInitialized();
    return _isInitialized;
  }

  /// Whether the device is started.
  bool isDeviceStarted() {
    // ignore: join_return_with_assignment
    _isStarted = _impl.isDeviceStarted();
    return _isStarted;
  }

  /// Start the device.
  ///
  /// WEB NOTE: it's preferable to call this method after the user accepted
  /// the recording permission.
  ///
  /// Throws [RecorderNotInitializedException].
  /// Throws [RecorderFailedToStartDeviceException].
  void start() {
    if (!_isInitialized) {
      _log.warning(() => 'start(): recorder is not initialized.');
      throw const RecorderNotInitializedException();
    }
    _impl.start();
    _isStarted = true;
  }

  /// Stop the device.
  void stop() {
    if (!_isInitialized) {
      _log.warning(() => 'stop(): recorder is not initialized.');
      return;
    }
    _isStarted = false;
    _impl.stop();
  }

  /// Start streaming data.
  ///
  /// Throws [RecorderNotInitializedException].
  void startStreamingData() {
    if (!_isInitialized) {
      _log.warning(() => 'startStreamingData(): recorder is not initialized.');
      throw const RecorderNotInitializedException();
    }
    _impl.startStreamingData();
  }

  /// Stop streaming data.
  void stopStreamingData() {
    if (!_isInitialized) {
      _log.warning(() => 'stopStreamingData(): recorder is not initialized.');
      return;
    }
    _impl.stopStreamingData();
  }

  /// Start recording.
  ///
  /// [completeFilePath] complete file path to save the recording.
  /// This is mandatory on all platforms but on the Web.
  /// NOTE: when running on the  Web, [completeFilePath] is ignored:
  /// when stopping the recording the browser will ask to save the file.
  ///
  /// Throws [RecorderNotInitializedException].
  /// Throws [RecorderCaptureNotStartededException].
  /// Throws [RecorderInvalidFileNameException] if the given file name is
  /// invalid.
  void startRecording({String completeFilePath = ''}) {
    assert(
      () {
        if (!kIsWeb && completeFilePath.isEmpty) {
          return false;
        }
        return true;
      }.call(),
      'completeFilePath is required on all platforms but on the Web.',
    );
    if (!_isInitialized) {
      _log.warning(() => 'startRecording(): recorder is not initialized.');
      throw const RecorderNotInitializedException();
    }
    if (!_isStarted) {
      _log.warning(() => 'startRecording(): recorder is not started.');
      throw const RecorderCaptureNotStartededException();
    }
    _impl.startRecording(completeFilePath);
  }

  /// Pause recording.
  void setPauseRecording({required bool pause}) {
    if (!_isStarted) return;
    _impl.setPauseRecording(pause: pause);
  }

  /// Stop recording.
  void stopRecording() {
    if (!_isStarted) return;
    _impl.stopRecording();
  }

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
  void setFftSmoothing(double smooth) {
    _impl.setFftSmoothing(smooth);
  }

  /// Enable or disable low-latency audio monitoring (input passthrough to output).
  /// When enabled, microphone input is directly routed to speakers at the native level.
  /// WARNING: Can cause feedback! Use headphones or ensure speakers are not near microphone.
  void setMonitoring(bool enabled) {
    _impl.setMonitoring(enabled);
  }

  /// Set monitoring mode for stereo inputs.
  /// - 0: Stereo (normal passthrough)
  /// - 1: Left Mono (left channel to both outputs)
  /// - 2: Right Mono (right channel to both outputs)
  /// - 3: Mono (mix both channels to both outputs)
  void setMonitoringMode(int mode) {
    _impl.setMonitoringMode(mode);
  }

  // ==================== FILTER DEBUG STATS ====================

  /// Get the number of times filter processing was skipped due to lock contention.
  /// High values indicate audio thread is being blocked by UI thread.
  int getFilterMissCount() => _impl.getFilterMissCount();

  /// Get the number of times filter processing completed successfully.
  int getFilterProcessCount() => _impl.getFilterProcessCount();

  /// Reset filter stats counters. Call at session start.
  void resetFilterStats() => _impl.resetFilterStats();

  /// Get the filter miss rate as a percentage (0-100).
  /// Returns 0 if no processing has occurred.
  double getFilterMissRate() {
    final miss = getFilterMissCount();
    final process = getFilterProcessCount();
    final total = miss + process;
    if (total == 0) return 0.0;
    return (miss / total) * 100.0;
  }

  /// Conveninet way to get FFT data. Return a 256 float array containing
  /// FFT data in the range [-1.0, 1.0] not clamped.
  ///
  /// If also wave data is needed consider using [getTexture] or [getTexture2D].
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  Float32List getFft({bool alwaysReturnData = true}) {
    if (!_isInitialized) {
      _log.warning(() => 'getFft: recorder is not initialized.');
      return Float32List(256);
    }
    if (!_isStarted) {
      _log.warning(() => 'getFft: recorder is not started.');
      return Float32List(256);
    }
    if (_recorderFormat != PCMFormat.f32le) {
      _log.warning(
        () => 'getFft: FFT data can be get only using f32le format.',
      );
      return Float32List(256);
    }
    return _impl.getFft(alwaysReturnData: alwaysReturnData);
  }

  /// Return a 256 float array containing wave data in the range [-1.0, 1.0]
  /// not clamped.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  Float32List getWave({bool alwaysReturnData = true}) {
    if (!_isInitialized) {
      _log.warning(() => 'getWave: recorder is not initialized.');
      return Float32List(256);
    }
    if (!_isStarted) {
      _log.warning(() => 'getWave: recorder is not started.');
      return Float32List(256);
    }
    if (_recorderFormat != PCMFormat.f32le) {
      _log.warning(
        () => 'getWave: wave data can be get only using f32le format.',
      );
      return Float32List(256);
    }
    return _impl.getWave(alwaysReturnData: alwaysReturnData);
  }

  /// Get the audio data representing an array of 256 floats FFT data and
  /// 256 float of wave data.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  Float32List getTexture({bool alwaysReturnData = true}) {
    if (!_isInitialized) {
      _log.warning(() => 'getTexture: recorder is not initialized.');
      return Float32List(256);
    }
    if (!_isStarted) {
      _log.warning(() => 'getTexture: recorder is not started.');
      return Float32List(256);
    }
    return _impl.getTexture(alwaysReturnData: alwaysReturnData);
  }

  /// Get the audio data representing an array of 256 floats FFT data and
  /// 256 float of wave data.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  Float32List getTexture2D({bool alwaysReturnData = true}) {
    if (!_isInitialized) {
      _log.warning(() => 'getTexture2D: recorder is not initialized.');
      return Float32List(256);
    }
    if (!_isStarted) {
      _log.warning(() => 'getTexture2D: recorder is not started.');
      return Float32List(256);
    }
    if (_recorderFormat != PCMFormat.f32le) {
      _log.warning(
        () => 'getTexture2D: texture can be get only using f32le format.',
      );
      return Float32List(256);
    }
    return _impl.getTexture2D(alwaysReturnData: alwaysReturnData);
  }

  /// Get the current volume in dB. Returns -100 if the capture is not inited.
  /// 0 is the max volume the capture device can handle.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  /// Get the current volume in dB. Returns -100 if the capture is not inited.
  /// 0 is the max volume the capture device can handle.
  ///
  /// **NOTE**: use this only with format [PCMFormat.f32le].
  double getVolumeDb() {
    if (!_isInitialized) {
      _log.warning(() => 'getVolumeDb: recorder is not initialized.');
      return -100;
    }
    if (!_isStarted) {
      _log.warning(() => 'getVolumeDb: recorder is not started.');
      return -100;
    }
    if (_recorderFormat != PCMFormat.f32le) {
      _log.warning(
        () => 'getVolumeDb: volume can be get only using f32le format.',
      );
      return -100;
    }
    return _impl.getVolumeDb();
  }

  /// Get the actual sample rate configured on the device.
  int getSampleRate() {
    if (!_isInitialized) return 0;
    return _impl.getSampleRate();
  }

  /// Get the actual capture channels configured on the device.
  int getCaptureChannels() {
    if (!_isInitialized) return 0;
    return _impl.getCaptureChannels();
  }

  /// Get the actual playback channels configured on the device.
  int getPlaybackChannels() {
    if (!_isInitialized) return 0;
    return _impl.getPlaybackChannels();
  }

  /// Get the actual capture format ID.
  int getCaptureFormat() {
    if (!_isInitialized) return 0;
    return _impl.getCaptureFormat();
  }

  /// Get the actual playback format ID.
  int getPlaybackFormat() {
    if (!_isInitialized) return 0;
    return _impl.getPlaybackFormat();
  }

  // ///////////////////////
  //   FILTERS
  // ///////////////////////

  /// Check if a filter is active.
  /// Return -1 if the filter is not active or its index.
  int isFilterActive(RecorderFilterType filterType) {
    return _impl.isFilterActive(filterType);
  }

  /// Add a filter.
  ///
  /// Throws [RecorderFilterAlreadyAddedException] if the filter has already
  /// been added.
  /// Throws [RecorderFilterNotFoundException] if the filter could not be found.
  void addFilter(RecorderFilterType filterType) {
    _impl.addFilter(filterType);
  }

  /// Remove a filter.
  ///
  /// Throws [RecorderFilterNotFoundException] if trying to a non active
  /// filter.
  CaptureErrors removeFilter(RecorderFilterType filterType) {
    return _impl.removeFilter(filterType);
  }

  /// Get filter param names.
  List<String> getFilterParamNames(RecorderFilterType filterType) {
    return _impl.getFilterParamNames(filterType);
  }

  /// Set filter param value.
  void setFilterParamValue(
    RecorderFilterType filterType,
    int attributeId,
    double value,
  ) {
    _impl.setFilterParamValue(filterType, attributeId, value);
  }

  /// Get filter param value.
  double getFilterParamValue(RecorderFilterType filterType, int attributeId) {
    return _impl.getFilterParamValue(filterType, attributeId);
  }

  // ///////////////////////
  //   SLAVE MODE
  // ///////////////////////

  /// Check if slave audio is ready (first callback has run successfully).
  /// This is used to wait for the audio pipeline to stabilize before calibration.
  bool isSlaveAudioReady() {
    return _impl.isSlaveAudioReady();
  }

  // ///////////////////////
  //   AEC (Adaptive Echo Cancellation)
  // ///////////////////////

  /// Set AEC Mode (Bypass, Algo, Neural, Hybrid).
  /// Allows switching between different AEC implementations for A/B testing.
  void setAecMode(AecMode mode) {
    _impl.aecSetMode(mode);
  }

  /// Get current AEC Mode.
  AecMode getAecMode() {
    return _impl.aecGetMode();
  }

  /// Load neural model by type.
  /// [type] the model type (aecMaskV3).
  /// [assetBasePath] path to the assets directory.
  bool aecLoadNeuralModel(NeuralModelType type, String assetBasePath) {
    return _impl.aecLoadNeuralModel(type, assetBasePath);
  }

  /// Get currently loaded neural model type.
  NeuralModelType aecGetLoadedNeuralModel() {
    return _impl.aecGetLoadedNeuralModel();
  }

  /// Enable or disable neural post-filter.
  void aecSetNeuralEnabled(bool enabled) {
    _impl.aecSetNeuralEnabled(enabled);
  }

  /// Check if neural post-filter is enabled.
  bool aecIsNeuralEnabled() {
    return _impl.aecIsNeuralEnabled();
  }

  /// Create the AEC reference buffer.
  /// Returns a pointer to the buffer that should be passed to SoLoud.
  /// [sampleRate] and [channels] should match the audio device configuration.
  int aecCreateReferenceBuffer(int sampleRate, int channels) {
    return _impl.aecCreateReferenceBuffer(sampleRate, channels);
  }

  /// Destroy the AEC reference buffer.
  void aecDestroyReferenceBuffer() {
    _impl.aecDestroyReferenceBuffer();
  }

  /// Get the AEC output callback function pointer.
  /// This should be passed to SoLoud to receive playback audio.
  int aecGetOutputCallback() {
    return _impl.aecGetOutputCallback();
  }

  /// Reset the AEC buffer (e.g., when switching audio configurations).
  void aecResetBuffer() {
    _impl.aecResetBuffer();
  }

  /// Enable or disable AEC reference buffer writes.
  /// When disabled, saves CPU when AEC is not needed.
  void aecSetEnabled(bool enabled) {
    _impl.aecSetEnabled(enabled);
  }

  /// Check if AEC reference buffer is enabled.
  bool aecIsEnabled() {
    return _impl.aecIsEnabled();
  }

  // ==================== AEC CALIBRATION ====================

  /// Generate calibration audio signal.
  /// [signalType] determines the signal:
  ///   - chirp: Logarithmic sine sweep (default)
  ///   - click: Impulse train (better for transients)
  /// Returns WAV data as Uint8List that can be loaded into SoLoud.
  Uint8List aecGenerateCalibrationSignal(
    int sampleRate,
    int channels, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  }) {
    return _impl.aecGenerateCalibrationSignal(
      sampleRate,
      channels,
      signalType: signalType,
    );
  }

  /// Start capturing microphone samples for calibration analysis.
  /// [maxSamples] is the maximum number of mono samples to capture.
  /// For ~2 seconds at 48kHz, use 96000.
  void aecStartCalibrationCapture(int maxSamples) {
    _impl.aecStartCalibrationCapture(maxSamples);
  }

  /// Stop capturing samples for calibration.
  void aecStopCalibrationCapture() {
    _impl.aecStopCalibrationCapture();
  }

  /// Capture signals from both reference and mic buffers for analysis.
  /// Call this after the calibration audio has finished playing.
  void aecCaptureForAnalysis() {
    _impl.aecCaptureForAnalysis();
  }

  /// Run cross-correlation analysis on captured signals.
  /// Returns a [AecCalibrationResult] with delay, gain, and correlation values.
  AecCalibrationResult aecRunCalibrationAnalysis(int sampleRate) {
    return _impl.aecRunCalibrationAnalysis(sampleRate);
  }

  /// Reset calibration state.
  void aecResetCalibration() {
    _impl.aecResetCalibration();
  }

  /// Run calibration with impulse response computation.
  /// Returns result including impulse length (call aecGetImpulseResponse to get data).
  AecCalibrationResultWithImpulse aecRunCalibrationWithImpulse(int sampleRate) {
    return _impl.aecRunCalibrationWithImpulse(sampleRate);
  }

  /// Get stored impulse response from last calibration.
  /// Returns Float32List of coefficients.
  Float32List aecGetImpulseResponse(int maxLength) {
    return _impl.aecGetImpulseResponse(maxLength);
  }

  /// Apply stored impulse response to AEC filter.
  void aecApplyImpulseResponse() {
    _impl.aecApplyImpulseResponse();
  }

  /// Get captured reference signal for visualization.
  Float32List aecGetCalibrationRefSignal(int maxLength) {
    return _impl.aecGetCalibrationRefSignal(maxLength);
  }

  /// Get captured mic signal for visualization.
  Float32List aecGetCalibrationMicSignal(int maxLength) {
    return _impl.aecGetCalibrationMicSignal(maxLength);
  }

  /// Force speaker output on iOS (useful for measurement mode).
  void iosForceSpeakerOutput(bool enabled) {
    _impl.iosForceSpeakerOutput(enabled);
  }

  // ==================== AEC TESTING ====================

  /// Start capturing test signals (raw mic + cancelled output).
  /// Call this BEFORE playing the test audio.
  /// [maxSamples] is the maximum number of samples to capture per signal.
  void aecStartTestCapture(int maxSamples) {
    _impl.aecStartTestCapture(maxSamples);
  }

  /// Stop capturing test signals.
  /// Call this AFTER test audio has finished playing.
  void aecStopTestCapture() {
    _impl.aecStopTestCapture();
  }

  /// Run analysis on captured test signals.
  /// Computes cancellation metrics and determines pass/fail.
  /// Returns [AecTestResult] with all metrics.
  AecTestResult aecRunTest(int sampleRate) {
    return _impl.aecRunTest(sampleRate);
  }

  /// Get captured raw mic signal (before AEC) for visualization.
  Float32List aecGetTestMicSignal(int maxLength) {
    return _impl.aecGetTestMicSignal(maxLength);
  }

  /// Get captured cancelled signal (after AEC) for visualization.
  Float32List aecGetTestCancelledSignal(int maxLength) {
    return _impl.aecGetTestCancelledSignal(maxLength);
  }

  /// Reset test data.
  void aecResetTest() {
    _impl.aecResetTest();
  }

  // ==================== VSS-NLMS PARAMETER CONTROL ====================

  /// Set VSS-NLMS maximum step size (0.0-1.0). Set to 0 to freeze weights.
  void aecSetVssMuMax(double mu) {
    _impl.aecSetVssMuMax(mu);
  }

  /// Set VSS-NLMS leakage factor (0.99-1.0). Set to 1.0 for no decay.
  void aecSetVssLeakage(double lambda) {
    _impl.aecSetVssLeakage(lambda);
  }

  /// Set VSS-NLMS smoothing factor (0.9-0.999).
  void aecSetVssAlpha(double alpha) {
    _impl.aecSetVssAlpha(alpha);
  }

  /// Get current VSS-NLMS maximum step size.
  double aecGetVssMuMax() {
    return _impl.aecGetVssMuMax();
  }

  /// Get current VSS-NLMS leakage factor.
  double aecGetVssLeakage() {
    return _impl.aecGetVssLeakage();
  }

  /// Get current VSS-NLMS smoothing factor.
  double aecGetVssAlpha() {
    return _impl.aecGetVssAlpha();
  }

  // ==================== AEC FILTER LENGTH CONTROL ====================

  /// Set AEC filter length (2048, 4096, 8192 recommended).
  /// Longer filters can capture longer reverb tails but use more CPU.
  void aecSetFilterLength(int length) {
    _impl.aecSetFilterLength(length);
  }

  /// Get current AEC filter length.
  int aecGetFilterLength() {
    return _impl.aecGetFilterLength();
  }

  // ==================== AEC CALIBRATION LOGGING ====================

  /// Get the calibration log buffer containing debug messages from native code.
  /// Useful for debugging calibration issues.
  String aecGetCalibrationLog() {
    return _impl.aecGetCalibrationLog();
  }

  /// Clear the calibration log buffer.
  void aecClearCalibrationLog() {
    _impl.aecClearCalibrationLog();
  }

  // ==================== AEC POSITION-BASED SYNC ====================

  /// Get total frames written to reference buffer (output side counter).
  /// Used for sample-accurate AEC synchronization.
  int aecGetOutputFrameCount() {
    return _impl.aecGetOutputFrameCount();
  }

  /// Get total frames captured by recorder (input side counter).
  /// Used for sample-accurate AEC synchronization.
  int aecGetCaptureFrameCount() {
    return _impl.aecGetCaptureFrameCount();
  }

  /// Record frame counters at calibration start.
  /// Call this when calibration signal starts playing.
  /// The counters are used to calculate the position-based offset.
  void aecRecordCalibrationFrameCounters() {
    _impl.aecRecordCalibrationFrameCounters();
  }

  /// Set the calibrated offset for position-based sync.
  /// This should be called after calibration completes:
  /// offset = (captureFramesAtStart - outputFramesAtStart) + acousticDelaySamples
  void aecSetCalibratedOffset(int offset) {
    _impl.aecSetCalibratedOffset(offset);
  }

  /// Get the current calibrated offset.
  int aecGetCalibratedOffset() {
    return _impl.aecGetCalibratedOffset();
  }

  // ==================== ALIGNED CALIBRATION CAPTURE ====================

  /// Start capturing aligned ref+mic from AEC processAudio callback.
  /// This captures frame-aligned signals for accurate delay estimation.
  /// Unlike independent capture (aecStartCalibrationCapture), this uses
  /// signals that are already synchronized inside the AEC callback.
  /// [maxSamples] is the maximum number of mono samples to capture.
  void aecStartAlignedCalibrationCapture(int maxSamples) {
    _impl.aecStartAlignedCalibrationCapture(maxSamples);
  }

  /// Stop aligned calibration capture.
  void aecStopAlignedCalibrationCapture() {
    _impl.aecStopAlignedCalibrationCapture();
  }

  /// Run calibration analysis on aligned buffers and apply impulse response.
  /// [signalType] should match what was used for generation.
  /// Returns the calibration result with delay and impulse info.
  /// This is more accurate than aecRunCalibrationWithImpulse because it uses
  /// frame-aligned signals captured from inside the AEC callback.
  AecCalibrationResultWithImpulse aecRunAlignedCalibrationWithImpulse(
    int sampleRate, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  }) {
    return _impl.aecRunAlignedCalibrationWithImpulse(
      sampleRate,
      signalType: signalType,
    );
  }

  // ==================== NATIVE AUDIO SINK ====================

  /// Set native audio sink for direct recorder-to-player streaming.
  /// This bypasses Dart's main thread for audio data.
  /// [callbackAddress] and [userDataAddress] should come from SoLoud's
  /// configureNativeAudioSinkRaw().
  void setNativeAudioSink(int callbackAddress, int userDataAddress) {
    _impl.setNativeAudioSink(callbackAddress, userDataAddress);
  }

  /// Check if native audio sink is currently active.
  bool isNativeAudioSinkActive() {
    return _impl.isNativeAudioSinkActive();
  }

  /// Disable native audio sink. Audio data will flow through Dart again.
  void disableNativeAudioSink() {
    _impl.disableNativeAudioSink();
  }

  /// Inject preroll audio from ring buffer into SoLoud stream via native path.
  /// This reads [frameCount] frames from the ring buffer and sends them
  /// directly to the native audio sink callback, keeping everything native.
  void injectPreroll(int frameCount) {
    _impl.injectPreroll(frameCount);
  }

  // ==================== NATIVE SCHEDULER ====================
  // Sample-accurate timing for recording start/stop in audio callback

  /// Reset the native scheduler state.
  /// Call this when starting a new session or when timing state should be cleared.
  void schedulerReset() {
    _impl.schedulerReset();
  }

  /// Set base loop parameters for quantization.
  /// [loopFrames] is the loop length in frames.
  /// [loopStartFrame] is the global frame when the loop started.
  /// After setting, scheduled events will align to loop boundaries.
  void schedulerSetBaseLoop(int loopFrames, int loopStartFrame) {
    _impl.schedulerSetBaseLoop(loopFrames, loopStartFrame);
  }

  /// Clear base loop (free recording mode).
  /// Events will fire at next buffer boundary instead of loop boundary.
  void schedulerClearBaseLoop() {
    _impl.schedulerClearBaseLoop();
  }

  /// Schedule quantized recording start.
  /// Recording will start at the next loop boundary (or immediately if no base loop).
  /// [path] is the complete file path for the WAV recording.
  /// Returns event ID (0 if failed to schedule).
  int schedulerScheduleStart(String path) {
    return _impl.schedulerScheduleStart(path);
  }

  /// Schedule quantized recording stop.
  /// Recording will stop at the next loop boundary that completes a whole loop multiple.
  /// [startFrame] is when recording started (for multi-loop calculation).
  /// Returns event ID (0 if failed to schedule).
  int schedulerScheduleStop(int startFrame) {
    return _impl.schedulerScheduleStop(startFrame);
  }

  /// Cancel a scheduled event by ID.
  /// Returns true if event was found and cancelled.
  bool schedulerCancelEvent(int eventId) {
    return _impl.schedulerCancelEvent(eventId);
  }

  /// Cancel all pending events.
  void schedulerCancelAll() {
    _impl.schedulerCancelAll();
  }

  /// Poll for fired event notification.
  /// Returns null if no notification available.
  /// Call this periodically (e.g., every 10-100ms) to get notified when events fire.
  SchedulerNotification? schedulerPollNotification() {
    return _impl.schedulerPollNotification();
  }

  /// Check if there are pending notifications.
  bool schedulerHasNotifications() {
    return _impl.schedulerHasNotifications();
  }

  /// Get current global frame position.
  /// This is updated by the audio callback and represents the last processed frame.
  int schedulerGetGlobalFrame() {
    return _impl.schedulerGetGlobalFrame();
  }

  /// Get base loop length in frames.
  /// Returns 0 if no base loop is set.
  int schedulerGetBaseLoopFrames() {
    return _impl.schedulerGetBaseLoopFrames();
  }

  /// Get next loop boundary frame.
  /// Returns the frame number of the next loop boundary from current position.
  int schedulerGetNextLoopBoundary() {
    return _impl.schedulerGetNextLoopBoundary();
  }

  /// Set latency compensation in frames.
  /// This is applied at recording start - the ring buffer will include
  /// audio from [frames] frames before the start event.
  void schedulerSetLatencyCompensation(int frames) {
    _impl.schedulerSetLatencyCompensation(frames);
  }

  /// Get latency compensation in frames.
  int schedulerGetLatencyCompensation() {
    return _impl.schedulerGetLatencyCompensation();
  }

  /// Set auto-stop enabled.
  /// When true (default), both START and STOP are scheduled upfront for
  /// exactly one loop length. When false, only START is scheduled, allowing
  /// manual control over when to stop (useful for pedal users).
  void schedulerSetAutoStop(bool enabled) {
    _impl.schedulerSetAutoStop(enabled);
  }

  /// Get auto-stop enabled state.
  bool schedulerIsAutoStopEnabled() {
    return _impl.schedulerIsAutoStopEnabled();
  }

  // ==================== NATIVE RING BUFFER ====================
  // Latency compensation via continuous capture with pre-roll

  /// Create/configure the native ring buffer for latency compensation.
  /// The ring buffer continuously captures audio in the native layer,
  /// allowing "pre-roll" reads to capture audio from before the record
  /// button was pressed (compensating for input latency).
  ///
  /// [capacitySeconds] How many seconds of audio to keep (typically 5).
  /// [sampleRate] Sample rate in Hz.
  /// [channels] Number of channels (1=mono, 2=stereo).
  void createRingBuffer(int capacitySeconds, int sampleRate, int channels) {
    _impl.createRingBuffer(capacitySeconds, sampleRate, channels);
  }

  /// Destroy/reset the native ring buffer.
  void destroyRingBuffer() {
    _impl.destroyRingBuffer();
  }

  /// Read pre-roll samples for latency compensation.
  /// This reads audio from the past to compensate for control latency
  /// (touchscreen ~50ms, Bluetooth ~35ms, USB MIDI ~15ms).
  ///
  /// [frameCount] Number of frames to read.
  /// [rewindFrames] How many frames back in time to start reading.
  /// Returns Float32List with interleaved samples.
  Float32List readPreRoll(int frameCount, int rewindFrames) {
    return _impl.readPreRoll(frameCount, rewindFrames);
  }

  /// Get current audio level in dB (RMS).
  /// This is calculated continuously in the native audio callback,
  /// enabling efficient level metering without Dart overhead.
  double getAudioLevelDb() {
    return _impl.getAudioLevelDb();
  }

  /// Get total frames written to the ring buffer.
  /// Useful for synchronization with other components.
  int getRingBufferFramesWritten() {
    return _impl.getRingBufferFramesWritten();
  }

  /// Get available frames in the ring buffer.
  /// Returns the number of valid frames (up to capacity after wrap).
  int getRingBufferAvailable() {
    return _impl.getRingBufferAvailable();
  }

  /// Reset the ring buffer (clear all data).
  void resetRingBuffer() {
    _impl.resetRingBuffer();
  }
}
