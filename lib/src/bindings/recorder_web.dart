// ignore_for_file: omit_local_variable_types
// ignore_for_file: avoid_positional_boolean_parameters, public_member_api_docs

import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter_recorder/src/audio_data_container.dart';
import 'package:flutter_recorder/src/bindings/js_extension.dart';
import 'package:flutter_recorder/src/bindings/recorder.dart';
import 'package:flutter_recorder/src/enums.dart';
import 'package:flutter_recorder/src/exceptions/exceptions.dart';
import 'package:flutter_recorder/src/filters/filters.dart';
import 'package:flutter_recorder/src/flutter_recorder.dart';
import 'package:flutter_recorder/src/worker/worker.dart';
import 'package:meta/meta.dart';

@internal
class RecorderController {
  factory RecorderController() => _instance ??= RecorderController._();

  RecorderController._() {
    impl = RecorderWeb();
  }
  static RecorderController? _instance;

  late final RecorderImpl impl;
}

/// Use this class to _capture_ audio (such as from a microphone).
@internal
class RecorderWeb extends RecorderImpl {
  @override
  Stream<AecStats> get aecStatsStream => Stream.empty();

  @override
  Future<void> setAecStatsCallback() async {}

  @override
  Stream<RecordingStoppedEvent> get recordingStoppedStream => Stream.empty();

  @override
  Future<void> setRecordingStoppedCallback() async {}

  @override
  Stream<RecordingStartedEvent> get recordingStartedStream => Stream.empty();

  @override
  Future<void> setRecordingStartedCallback() async {}

  SilenceCallback? _silenceCallback;

  /// Create the worker in the WASM Module and listen for events coming
  /// from `web/worker.dart.js`
  @override
  Future<void> setDartEventCallbacks() async {
    // This calls the native WASM `createWorkerInWasm()` in `bindings.cpp`.
    // The latter creates a web Worker using `EM_ASM` inlining JS code to
    // create the worker in the WASM `Module`.
    wasmCreateWorkerInWasm();

    wasmSetDartEventCallback(0, 0);

    // Here the `RecorderModule.wasmModule` binded to a local [WorkerController]
    // is used in the main isolate to listen for events coming from native.
    // From native the events can be sent from the main thread and even from
    // other threads like the audio thread.
    final workerController = WorkerController()..setWasmWorker(wasmWorker);
    workerController.onReceive().listen(
      (event) {
        /// The [event] coming from `web/worker.dart.js` is of Map type.
        switch (event) {
          case Map():
            if (event['message'] == 'silenceChangedCallback') {
              final silence = (event['isSilent'] as int) == 1;
              final db = event['energyDb'] as double;
              _silenceCallback?.call(silence, db);
              silenceChangedEventController.add(
                (isSilent: silence, decibel: db),
              );
            }

            if (event['message'] == 'streamDataCallback') {
              final audioData = Uint8List.fromList(event['data'] as Uint8List);
              uint8ListController.add(AudioDataContainer(audioData));
            }
        }
      },
    );
  }

  @override
  void setSilenceDetection({
    required bool enable,
    SilenceCallback? onSilenceChanged,
  }) {
    wasmSetSilenceDetection(enable);

    if (onSilenceChanged != null) {
      _silenceCallback = onSilenceChanged;
    }
    if (!enable) {
      _silenceCallback = null;
    }
  }

  @override
  void setSilenceThresholdDb(double silenceThresholdDb) {
    assert(silenceThresholdDb < 0, 'silenceThresholdDb must be < 0');
    wasmSetSilenceThresholdDb(silenceThresholdDb);
  }

  @override
  void setSilenceDuration(double silenceDuration) {
    assert(silenceDuration >= 0, 'silenceDuration must be >= 0');
    wasmSetSilenceDuration(silenceDuration);
  }

  @override
  void setSecondsOfAudioToWriteBefore(double secondsOfAudioToWriteBefore) {
    assert(
      secondsOfAudioToWriteBefore >= 0,
      'secondsOfAudioToWriteBefore must be >= 0',
    );
    wasmSetSecondsOfAudioToWriteBefore(secondsOfAudioToWriteBefore);
  }

  @override
  List<CaptureDevice> listCaptureDevices() {
    /// allocate 50 device strings
    final namesPtr = wasmMalloc(50 * 255);
    final deviceIdPtr = wasmMalloc(50 * 4);
    final isDefaultPtr = wasmMalloc(50 * 4);
    final nDevicesPtr = wasmMalloc(4); // 4 bytes for an int

    wasmListCaptureDevices(
      namesPtr,
      deviceIdPtr,
      isDefaultPtr,
      nDevicesPtr,
    );

    final nDevices = wasmGetI32Value(nDevicesPtr, '*');
    final devices = <CaptureDevice>[];
    for (var i = 0; i < nDevices; i++) {
      final namePtr = wasmGetI32Value(namesPtr + i * 4, '*');
      final name = wasmUtf8ToString(namePtr);
      final deviceId =
          wasmGetI32Value(wasmGetI32Value(deviceIdPtr + i * 4, '*'), '*');
      final isDefault =
          wasmGetI32Value(wasmGetI32Value(isDefaultPtr + i * 4, '*'), '*');

      devices.add(CaptureDevice(name, isDefault == 1, deviceId));
    }

    wasmFreeListCaptureDevices(namesPtr, deviceIdPtr, isDefaultPtr, nDevices);

    wasmFree(nDevicesPtr);
    wasmFree(deviceIdPtr);
    wasmFree(isDefaultPtr);
    wasmFree(namesPtr);

    return devices;
  }

  @override
  void init({
    required int deviceID,
    required PCMFormat format,
    required int sampleRate,
    required RecorderChannels channels,
    bool captureOnly = false,
  }) {
    // Note: captureOnly is ignored on web - no native duplex mode
    final error = wasmInit(deviceID, format.value, sampleRate, channels.count);
    if (CaptureErrors.fromValue(error) != CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(
        CaptureErrors.fromValue(error),
      );
    }
    super.init(
      deviceID: deviceID,
      format: format,
      sampleRate: sampleRate,
      channels: channels,
      captureOnly: captureOnly,
    );
  }

  @override
  void deinit() {
    _silenceCallback = null;
    wasmDeinit();
    super.deinit();
  }

  @override
  bool isDeviceInitialized() {
    return wasmIsDeviceInitialized() == 1;
  }

  @override
  bool isDeviceStarted() {
    return wasmIsDeviceStarted() == 1;
  }

  @override
  void start() {
    final error = wasmStart();
    if (CaptureErrors.fromValue(error) != CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(
        CaptureErrors.fromValue(error),
      );
    }
  }

  @override
  void stop() {
    wasmStop();
  }

  @override
  void startStreamingData() {
    wasmStartStreamingData();
  }

  @override
  void stopStreamingData() {
    wasmStopStreamingData();
  }

  @override
  void startRecording(String path) {
    final pathPtr = wasmMalloc(path.length);
    for (var i = 0; i < path.length; i++) {
      wasmSetValue(pathPtr + i, path.codeUnits[i], 'i8');
    }
    final error = wasmStartRecording(pathPtr);
    if (CaptureErrors.fromValue(error) != CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(
        CaptureErrors.fromValue(error),
      );
    }
  }

  @override
  void setPauseRecording({required bool pause}) {
    wasmSetPauseRecording(pause);
  }

  @override
  void stopRecording() {
    wasmStopRecording();
  }

  @override
  void setFftSmoothing(double smooth) {
    wasmSetFftSmoothing(smooth);
  }

  @override
  void setMonitoring(bool enabled) {
    // Not implemented on Web for now
  }

  @override
  void setMonitoringMode(int mode) {
    // Not implemented on Web for now
  }

  @override
  int getFilterMissCount() => 0;

  @override
  int getFilterProcessCount() => 0;

  @override
  void resetFilterStats() {}

  @override
  Float32List getFft({bool alwaysReturnData = true}) {
    final samplesPtr = wasmMalloc(4);
    final isTheSameAsBeforePtr = wasmMalloc(4);
    wasmGetFft(samplesPtr, isTheSameAsBeforePtr);
    final isSameData = wasmGetI32Value(isTheSameAsBeforePtr, 'i32');
    wasmFree(isTheSameAsBeforePtr);
    if (!alwaysReturnData && isSameData == 1) {
      wasmFree(samplesPtr);
      return Float32List(0);
    }

    final samplesPtr2 = wasmGetI32Value(samplesPtr, '*');
    final samples = Float32List(256);
    for (var i = 0; i < 256; i++) {
      samples[i] = wasmGetF32Value(samplesPtr2 + i * 4, 'float');
    }
    wasmFree(samplesPtr);
    return samples;
  }

  @override
  Float32List getWave({bool alwaysReturnData = true}) {
    final samplesPtr = wasmMalloc(4);
    final isTheSameAsBeforePtr = wasmMalloc(4);
    wasmGetWave(samplesPtr, isTheSameAsBeforePtr);
    final isSameData = wasmGetI32Value(isTheSameAsBeforePtr, 'i32');
    wasmFree(isTheSameAsBeforePtr);
    if (!alwaysReturnData && isSameData == 1) {
      wasmFree(samplesPtr);
      return Float32List(0);
    }

    final samplesPtr2 = wasmGetI32Value(samplesPtr, '*');
    final samples = Float32List(256);
    for (var i = 0; i < 256; i++) {
      samples[i] = wasmGetF32Value(samplesPtr2 + i * 4, 'float');
    }
    wasmFree(samplesPtr);
    return samples;
  }

  @override
  Float32List getTexture({bool alwaysReturnData = true}) {
    final samplesPtr = wasmMalloc(4);
    final isTheSameAsBeforePtr = wasmMalloc(4);
    wasmGetTexture(samplesPtr, isTheSameAsBeforePtr);
    final isSameData = wasmGetI32Value(isTheSameAsBeforePtr, 'i32');
    wasmFree(isTheSameAsBeforePtr);
    if (!alwaysReturnData && isSameData == 1) {
      wasmFree(samplesPtr);
      return Float32List(0);
    }

    final samplesPtr2 = wasmGetI32Value(samplesPtr, '*');
    final samples = Float32List(512);
    for (var i = 0; i < 512; i++) {
      samples[i] = wasmGetF32Value(samplesPtr2 + i * 4, 'float');
    }
    wasmFree(samplesPtr);
    return samples;
  }

  @override
  Float32List getTexture2D({bool alwaysReturnData = true}) {
    final samplesPtr = wasmMalloc(4);
    final isTheSameAsBeforePtr = wasmMalloc(4);
    wasmGetTexture2D(samplesPtr, isTheSameAsBeforePtr);
    final isSameData = wasmGetI32Value(isTheSameAsBeforePtr, 'i32');
    wasmFree(isTheSameAsBeforePtr);
    if (!alwaysReturnData && isSameData == 1) {
      wasmFree(samplesPtr);
      return Float32List(0);
    }

    final samplesPtr2 = wasmGetI32Value(samplesPtr, '*');
    final samples = Float32List(512 * 256);
    for (var i = 0; i < 512 * 256; i++) {
      samples[i] = wasmGetF32Value(samplesPtr2 + i * 4, 'float');
    }
    wasmFree(samplesPtr);
    return samples;
  }

  @override
  double getVolumeDb() {
    final volumeDbPtr = wasmMalloc(4);
    wasmGetVolumeDb(volumeDbPtr);
    final volumeDb = wasmGetF32Value(volumeDbPtr, 'float');
    wasmFree(volumeDbPtr);
    return volumeDb;
  }

  @override
  int isFilterActive(RecorderFilterType filterType) {
    return wasmIsFilterActive(filterType.value);
  }

  @override
  void addFilter(RecorderFilterType filterType) {
    final error = wasmAddFilter(filterType.value);
    if (CaptureErrors.fromValue(error) != CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(
        CaptureErrors.fromValue(error),
      );
    }
  }

  @override
  CaptureErrors removeFilter(RecorderFilterType filterType) {
    final error = wasmRemoveFilter(filterType.value);
    if (CaptureErrors.fromValue(error) != CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(
        CaptureErrors.fromValue(error),
      );
    }
    return CaptureErrors.fromValue(error);
  }

  @override
  List<String> getFilterParamNames(RecorderFilterType filterType) {
    final namesPtr = wasmMalloc(4);
    final paramsCountPtr = wasmMalloc(4);
    wasmGetFilterParamNames(filterType.value, namesPtr, paramsCountPtr);
    final namesPtr2 = wasmGetI32Value(namesPtr, '*');
    final paramsCount = wasmGetI32Value(paramsCountPtr, '*');
    final names = <String>[];
    for (var i = 0; i < paramsCount; i++) {
      final namePtr = wasmGetI32Value(namesPtr2 + i * 4, '*');
      final name = wasmUtf8ToString(namePtr);
      names.add(name);
    }
    wasmFree(namesPtr);
    wasmFree(paramsCountPtr);
    return names;
  }

  @override
  void setFilterParamValue(
    RecorderFilterType filterType,
    int attributeId,
    double value,
  ) {
    wasmSetFilterParamValue(filterType.value, attributeId, value);
  }

  @override
  double getFilterParamValue(RecorderFilterType filterType, int attributeId) {
    final value = wasmGetFilterParamValue(filterType.value, attributeId);
    return value;
  }

  // ///////////////////////
  //   SLAVE MODE
  // ///////////////////////

  @override
  bool isSlaveAudioReady() {
    // On web, we don't use slave mode - always return true
    return true;
  }

  // ///////////////////////
  //   AEC (Adaptive Echo Cancellation)
  // ///////////////////////

  @override
  int aecCreateReferenceBuffer(int sampleRate, int channels) {
    // AEC is not supported on web platform
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecDestroyReferenceBuffer() {
    // AEC is not supported on web platform
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  int aecGetOutputCallback() {
    // AEC is not supported on web platform
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecResetBuffer() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecSetEnabled(bool enabled) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  bool aecIsEnabled() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecSetMode(AecMode mode) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  AecMode aecGetMode() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  bool aecLoadNeuralModel(NeuralModelType type, String assetBasePath) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  NeuralModelType aecGetLoadedNeuralModel() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecSetNeuralEnabled(bool enabled) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  bool aecIsNeuralEnabled() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  // ==================== AEC CALIBRATION ====================

  @override
  Uint8List aecGenerateCalibrationSignal(
    int sampleRate,
    int channels, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  }) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  void aecStartCalibrationCapture(int maxSamples) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  void aecStopCalibrationCapture() {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  void aecCaptureForAnalysis() {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  AecCalibrationResult aecRunCalibrationAnalysis(int sampleRate) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  void aecResetCalibration() {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  AecCalibrationResultWithImpulse aecRunCalibrationWithImpulse(int sampleRate) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  Float32List aecGetImpulseResponse(int maxLength) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  void aecApplyImpulseResponse() {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  Float32List aecGetCalibrationRefSignal(int maxLength) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  Float32List aecGetCalibrationMicSignal(int maxLength) {
    throw UnsupportedError('AEC calibration is not supported on web platform');
  }

  @override
  void iosForceSpeakerOutput(bool enabled) {}

  // ==================== AEC TESTING ====================

  @override
  void aecStartTestCapture(int maxSamples) {
    throw UnsupportedError('AEC test is not supported on web platform');
  }

  @override
  void aecStopTestCapture() {
    throw UnsupportedError('AEC test is not supported on web platform');
  }

  @override
  AecTestResult aecRunTest(int sampleRate) {
    throw UnsupportedError('AEC test is not supported on web platform');
  }

  @override
  Float32List aecGetTestMicSignal(int maxLength) {
    throw UnsupportedError('AEC test is not supported on web platform');
  }

  @override
  Float32List aecGetTestCancelledSignal(int maxLength) {
    throw UnsupportedError('AEC test is not supported on web platform');
  }

  @override
  void aecResetTest() {
    throw UnsupportedError('AEC test is not supported on web platform');
  }

  // ==================== VSS-NLMS CONTROL ====================

  @override
  void aecSetVssMuMax(double mu) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecSetVssLeakage(double lambda) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  void aecSetVssAlpha(double alpha) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  double aecGetVssMuMax() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  double aecGetVssLeakage() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  double aecGetVssAlpha() {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  @override
  String aecGetCalibrationLog() => '';

  @override
  void aecClearCalibrationLog() {}

  @override
  int aecGetOutputFrameCount() => 0;

  @override
  int aecGetCaptureFrameCount() => 0;

  @override
  void aecRecordCalibrationFrameCounters() {}

  @override
  void aecSetCalibratedOffset(int offset) {}

  @override
  int aecGetCalibratedOffset() => 0;

  @override
  void aecSetFilterLength(int length) {}

  @override
  int aecGetFilterLength() => 0;

  @override
  void aecStartAlignedCalibrationCapture(int maxSamples) {}

  @override
  void aecStopAlignedCalibrationCapture() {}

  @override
  AecCalibrationResultWithImpulse aecRunAlignedCalibrationWithImpulse(
    int sampleRate, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  }) {
    throw UnsupportedError('AEC is not supported on web platform');
  }

  // ==================== NATIVE AUDIO SINK ====================
  // Not supported on web - audio goes through Dart

  @override
  void setNativeAudioSink(int callbackAddress, int userDataAddress) {
    // No-op on web - native sink not supported
  }

  @override
  bool isNativeAudioSinkActive() => false;

  @override
  void disableNativeAudioSink() {
    // No-op on web
  }

  @override
  void injectPreroll(int frameCount) {
    // No-op on web - preroll handled differently
  }

  // ==================== NATIVE SCHEDULER ====================
  // Not supported on web - scheduling uses JavaScript timers

  @override
  void schedulerReset() {
    // No-op on web - use Dart-based scheduling
  }

  @override
  void schedulerSetBaseLoop(int loopFrames, int loopStartFrame) {
    // No-op on web
  }

  @override
  void schedulerClearBaseLoop() {
    // No-op on web
  }

  @override
  int schedulerScheduleStart(String path) {
    // Not supported on web - return 0 (failed)
    return 0;
  }

  @override
  int schedulerScheduleStop(int startFrame) {
    // Not supported on web - return 0 (failed)
    return 0;
  }

  @override
  bool schedulerCancelEvent(int eventId) {
    return false;
  }

  @override
  void schedulerCancelAll() {
    // No-op on web
  }

  @override
  SchedulerNotification? schedulerPollNotification() {
    return null;
  }

  @override
  bool schedulerHasNotifications() {
    return false;
  }

  @override
  int schedulerGetGlobalFrame() {
    return 0;
  }

  @override
  int schedulerGetBaseLoopFrames() {
    return 0;
  }

  @override
  int schedulerGetNextLoopBoundary() {
    return 0;
  }

  @override
  void schedulerSetLatencyCompensation(int frames) {
    // No-op on web
  }

  @override
  int schedulerGetLatencyCompensation() {
    return 0;
  }

  // ==================== NATIVE RING BUFFER ====================
  // Stubs for web - native ring buffer not supported

  @override
  void createRingBuffer(int capacitySeconds, int sampleRate, int channels) {
    // No-op on web
  }

  @override
  void destroyRingBuffer() {
    // No-op on web
  }

  @override
  Float32List readPreRoll(int frameCount, int rewindFrames) {
    // Return empty buffer on web
    return Float32List(0);
  }

  @override
  double getAudioLevelDb() {
    return -100.0; // Silence
  }

  @override
  int getRingBufferFramesWritten() {
    return 0;
  }

  @override
  int getRingBufferAvailable() {
    return 0;
  }

  @override
  void resetRingBuffer() {
    // No-op on web
  }
}
