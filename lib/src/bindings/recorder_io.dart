// ignore_for_file: omit_local_variable_types
// ignore_for_file: avoid_positional_boolean_parameters, public_member_api_docs

import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_recorder/src/audio_data_container.dart';
import 'package:flutter_recorder/src/bindings/flutter_recorder_bindings_generated.dart'
    as bindings_gen;
import 'package:flutter_recorder/src/bindings/recorder.dart';
import 'package:flutter_recorder/src/enums.dart';
import 'package:flutter_recorder/src/exceptions/exceptions.dart';
import 'package:flutter_recorder/src/filters/filters.dart';
import 'package:flutter_recorder/src/flutter_recorder.dart';
import 'package:meta/meta.dart';

@internal
class RecorderController {
  factory RecorderController() => _instance ??= RecorderController._();

  RecorderController._() {
    impl = RecorderFfi();
  }
  static RecorderController? _instance;

  late final RecorderImpl impl;
}

// Helper to convert between generated and local CaptureErrors
CaptureErrors _toLocalCaptureError(bindings_gen.CaptureErrors error) {
  return CaptureErrors.values.firstWhere((e) => e.value == error.value);
}

// Helper to convert between local and generated RecorderFilterType
bindings_gen.RecorderFilterType _toGenFilterType(
    RecorderFilterType filterType) {
  return bindings_gen.RecorderFilterType.values
      .firstWhere((e) => e.value == filterType.value);
}

@internal
class RecorderFfi extends RecorderImpl {
  static const String _libName = 'flutter_recorder';

  /// The dynamic library in which the symbols for [FlutterRecorderBindings]
  /// can be found.
  static final ffi.DynamicLibrary _dylib = () {
    if (Platform.isMacOS || Platform.isIOS) {
      return ffi.DynamicLibrary.open('$_libName.framework/$_libName');
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return ffi.DynamicLibrary.open('lib$_libName.so');
    }
    if (Platform.isWindows) {
      return ffi.DynamicLibrary.open('$_libName.dll');
    }
    throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
  }();

  /// The bindings to the native functions in [_dylib].
  final bindings_gen.FlutterRecorderBindings _bindings =
      bindings_gen.FlutterRecorderBindings(_dylib);

  SilenceCallback? _silenceCallback;

  void _silenceChangedCallback(
    ffi.Pointer<ffi.Bool> silence,
    ffi.Pointer<ffi.Float> db,
  ) {
    _silenceCallback?.call(silence.value, db.value);
    silenceChangedEventController.add(
      (isSilent: silence.value, decibel: db.value),
    );
  }

  void _streamDataCallback(
    ffi.Pointer<ffi.UnsignedChar> data,
    int dataLength,
  ) {
    try {
      // Create a copy of the data
      final audioData = data.cast<ffi.Uint8>().asTypedList(dataLength).toList();
      uint8ListController.add(
        AudioDataContainer(Uint8List.fromList(audioData)),
      );
    } finally {
      // Free the memory allocated in C++
      _bindings.flutter_recorder_nativeFree(data.cast<ffi.Void>());
    }
  }

  ffi.NativeCallable<bindings_gen.dartStreamDataCallback_tFunction>?
      nativeStreamDataCallable;
  @override
  Future<void> setDartEventCallbacks() async {
    // Create a NativeCallable for the Dart functions
    final nativeSilenceChangedCallable = ffi.NativeCallable<
        bindings_gen.dartSilenceChangedCallback_tFunction>.listener(
      _silenceChangedCallback,
    );

    final nativeStreamDataCallable = ffi
        .NativeCallable<bindings_gen.dartStreamDataCallback_tFunction>.listener(
      _streamDataCallback,
    );

    _bindings.flutter_recorder_setDartEventCallback(
      nativeSilenceChangedCallable.nativeFunction,
      nativeStreamDataCallable.nativeFunction,
    );
  }

  final _aecStatsController = StreamController<AecStats>.broadcast();

  @override
  Stream<AecStats> get aecStatsStream => _aecStatsController.stream;

  void _aecStatsCallback(bindings_gen.AecStats stats) {
    _handleAecStats(stats);
  }

  void _handleAecStats(bindings_gen.AecStats stats) {
    if (_aecStatsController.isClosed) return;
    _aecStatsController.add(
      AecStats(
        maxAttenuationDb: stats.maxAttenuationDb,
        correlation: stats.correlation,
        echoReturnLossDb: stats.echoReturnLossDb,
        filterLength: stats.filterLength,
        muMax: stats.muMax,
        muEffective: stats.muEffective,
        lastErrorDb: stats.lastErrorDb,
        instantCorrelation: stats.instantCorrelation,
      ),
    );
  }

  ffi.NativeCallable<bindings_gen.AecStatsCallbackFunction>?
      _nativeAecStatsCallable;

  @override
  Future<void> setAecStatsCallback() async {
    _nativeAecStatsCallable =
        ffi.NativeCallable<bindings_gen.AecStatsCallbackFunction>.listener(
      _aecStatsCallback,
    );
    _bindings.flutter_recorder_set_aec_stats_callback(
      _nativeAecStatsCallable!.nativeFunction,
    );
  }

  @override
  void setSilenceDetection({
    required bool enable,
    SilenceCallback? onSilenceChanged,
  }) {
    _bindings.flutter_recorder_setSilenceDetection(enable);

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
    _bindings.flutter_recorder_setSilenceThresholdDb(silenceThresholdDb);
  }

  @override
  void setSilenceDuration(double silenceDuration) {
    assert(silenceDuration >= 0, 'silenceDuration must be >= 0');
    _bindings.flutter_recorder_setSilenceDuration(silenceDuration);
  }

  @override
  void setSecondsOfAudioToWriteBefore(double secondsOfAudioToWriteBefore) {
    assert(
      secondsOfAudioToWriteBefore >= 0,
      'secondsOfAudioToWriteBefore must be >= 0',
    );
    _bindings.flutter_recorder_setSecondsOfAudioToWriteBefore(
      secondsOfAudioToWriteBefore,
    );
  }

  @override
  List<CaptureDevice> listCaptureDevices() {
    final ret = <CaptureDevice>[];
    final ffi.Pointer<ffi.Pointer<ffi.Char>> deviceNames =
        calloc(ffi.sizeOf<ffi.Pointer<ffi.Pointer<ffi.Char>>>() * 255);
    final ffi.Pointer<ffi.Pointer<ffi.Int>> deviceIds =
        calloc(ffi.sizeOf<ffi.Pointer<ffi.Pointer<ffi.Int>>>() * 50);
    final ffi.Pointer<ffi.Pointer<ffi.Int>> deviceIsDefault =
        calloc(ffi.sizeOf<ffi.Pointer<ffi.Pointer<ffi.Int>>>() * 50);
    final ffi.Pointer<ffi.Int> nDevices = calloc();

    _bindings.flutter_recorder_listCaptureDevices(
      deviceNames,
      deviceIds,
      deviceIsDefault,
      nDevices,
    );

    final ndev = nDevices.value;
    for (var i = 0; i < ndev; i++) {
      var s = 'no name';
      final s1 = (deviceNames + i).value;
      if (s1 != ffi.nullptr) {
        s = s1.cast<Utf8>().toDartString();
      }
      final id1 = (deviceIds + i).value;
      final id = id1.value;
      final n1 = (deviceIsDefault + i).value;
      final n = n1.value;
      ret.add(CaptureDevice(s, n == 1, id));
    }

    // Free allocated memory is done in C.
    // This work on all platforms but not on win.
    // for (int i = 0; i < ndev; i++) {
    //   calloc.free(devices.elementAt(i).value.ref.name);
    //   calloc.free(devices.elementAt(i).value);
    // }
    _bindings.flutter_recorder_freeListCaptureDevices(
      deviceNames,
      deviceIds,
      deviceIsDefault,
      ndev,
    );

    calloc
      ..free(deviceNames)
      ..free(deviceIds)
      ..free(nDevices);
    return ret;
  }

  @override
  void init({
    required int deviceID,
    required PCMFormat format,
    required int sampleRate,
    required RecorderChannels channels,
    bool captureOnly = false,
  }) {
    final error = _bindings.flutter_recorder_init(
      deviceID,
      format.value,
      sampleRate,
      channels.count,
      captureOnly ? 1 : 0,
    );
    if (error != bindings_gen.CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(_toLocalCaptureError(error));
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
    _bindings.flutter_recorder_deinit();
    super.deinit();
  }

  @override
  bool isDeviceInitialized() {
    return _bindings.flutter_recorder_isInited() == 1;
  }

  @override
  bool isDeviceStarted() {
    return _bindings.flutter_recorder_isDeviceStarted() == 1;
  }

  @override
  void start() {
    final error = _bindings.flutter_recorder_start();
    if (error != bindings_gen.CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(_toLocalCaptureError(error));
    }
  }

  @override
  void stop() {
    _bindings.flutter_recorder_stop();
  }

  @override
  void startStreamingData() {
    _bindings.flutter_recorder_startStreamingData();
  }

  @override
  void stopStreamingData() {
    _bindings.flutter_recorder_stopStreamingData();
  }

  @override
  void startRecording(String path) {
    var errorDescription = '';
    // Check the file name is valid for the different platforms.
    bool isValidPathName() {
      // Reserved Windows filenames - these apply to any part of the path
      const reservedNames = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        // ignore: require_trailing_commas
      };

      switch (defaultTargetPlatform) {
        case TargetPlatform.windows:
          bool isValidDriveChar(int value) {
            return ((value | 0x20) - 'a'.codeUnitAt(0)) <=
                ('z'.codeUnitAt(0) - 'a'.codeUnitAt(0));
          }

          bool isDriveCharWithVolumeSeparatorChar(String path) {
            return path.length >= 2 &&
                isValidDriveChar(path.codeUnitAt(0)) &&
                path[1] == ':';
          }

          // Split path into components
          final pathParts = path.split(RegExp(r'[/\\]'));

          var needSkipCheckFirst = isDriveCharWithVolumeSeparatorChar(path);

          // Check each component
          for (final part in pathParts) {
            // Skip empty parts
            if (part.isEmpty) continue;

            if (needSkipCheckFirst) {
              needSkipCheckFirst = false;
              continue;
            }

            if (part.length == 1 && part == '.') continue;

            // Check for invalid characters in each part
            if (part.contains(RegExp('[:*?"<>|]')) ||
                reservedNames.contains(part.toUpperCase().split('.').first) ||
                part.endsWith(' ') ||
                part.endsWith('.')) {
              errorDescription = 'Invalid path component "$part". Path '
                  'components must not '
                  'contain any of these characters: :*?"<>| '
                  'or be a reserved name, or end with space/period.';
              return false;
            }
          }

          // Check total path length (Windows MAX_PATH is 260)
          if (path.length > 259) {
            errorDescription = 'Path is too long. Windows paths must be '
                'less than 260 characters.';
            return false;
          }

        case TargetPlatform.linux:
        case TargetPlatform.android:
          // Check for null bytes and control characters
          if (path.contains(RegExp(r'[\x00-\x1F]'))) {
            errorDescription = 'Path contains invalid control characters.';
            return false;
          }

        case TargetPlatform.macOS:
        case TargetPlatform.iOS:
          // Check for invalid characters on macOS/iOS
          if (path.contains(RegExp('[:<>]'))) {
            errorDescription = 'Path contains invalid characters. '
                'The following characters are not allowed: :<>';
            return false;
          }
          // Check for ._ at start (reserved for resource forks)
          if (path.split('/').any((part) => part.startsWith('._'))) {
            errorDescription =
                'File names cannot start with "._" on macOS/iOS.';
            return false;
          }

        case TargetPlatform.fuchsia:
          throw UnimplementedError();
      }

      return true;
    }

    if (!isValidPathName()) {
      throw RecorderInvalidFileNameException(errorDescription);
    }

    final error =
        _bindings.flutter_recorder_startRecording(path.toNativeUtf8().cast());
    if (error != bindings_gen.CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(_toLocalCaptureError(error));
    }
  }

  @override
  void setPauseRecording({required bool pause}) {
    _bindings.flutter_recorder_setPauseRecording(pause);
  }

  @override
  void stopRecording() {
    _bindings.flutter_recorder_stopRecording();
  }

  @override
  void setFftSmoothing(double smooth) {
    _bindings.flutter_recorder_setFftSmoothing(smooth);
  }

  @override
  void setMonitoring(bool enabled) {
    _bindings.flutter_recorder_setMonitoring(enabled);
  }

  @override
  void setMonitoringMode(int mode) {
    _bindings.flutter_recorder_setMonitoringMode(mode);
  }

  @override
  Float32List getFft({bool alwaysReturnData = true}) {
    final ffi.Pointer<ffi.Pointer<ffi.Float>> fft = calloc();
    final isTheSameAsBefore = calloc<ffi.Bool>();
    _bindings.flutter_recorder_getFft(fft, isTheSameAsBefore);
    if (!alwaysReturnData && isTheSameAsBefore.value) {
      calloc
        ..free(isTheSameAsBefore)
        ..free(fft);
      return Float32List(0);
    }

    final val = ffi.Pointer<ffi.Float>.fromAddress(fft.value.address);
    if (val == ffi.nullptr) {
      calloc.free(fft);
      return Float32List(0);
    }

    final fftList = val.cast<ffi.Float>().asTypedList(256);
    calloc.free(fft);
    return fftList;
  }

  @override
  Float32List getWave({bool alwaysReturnData = true}) {
    final ffi.Pointer<ffi.Pointer<ffi.Float>> wave = calloc();
    final isTheSameAsBefore = calloc<ffi.Bool>();
    _bindings.flutter_recorder_getWave(wave, isTheSameAsBefore);
    if (!alwaysReturnData && isTheSameAsBefore.value) {
      calloc
        ..free(isTheSameAsBefore)
        ..free(wave);
      return Float32List(0);
    }

    final val = ffi.Pointer<ffi.Float>.fromAddress(wave.value.address);
    if (val == ffi.nullptr) {
      calloc.free(wave);
      return Float32List(0);
    }

    final waveList = val.cast<ffi.Float>().asTypedList(256);
    calloc.free(wave);
    return waveList;
  }

  @override
  Float32List getTexture({bool alwaysReturnData = true}) {
    final ffi.Pointer<ffi.Pointer<ffi.Float>> data = calloc();
    final isTheSameAsBefore = calloc<ffi.Bool>();
    _bindings.flutter_recorder_getTexture(data, isTheSameAsBefore);
    if (!alwaysReturnData && isTheSameAsBefore.value) {
      calloc
        ..free(isTheSameAsBefore)
        ..free(data);
      return Float32List(0);
    }

    final val = data.value;
    if (val == ffi.nullptr) return Float32List(512);

    final textureList = val.cast<ffi.Float>().asTypedList(512);
    calloc
      ..free(isTheSameAsBefore)
      ..free(data);

    return textureList;
  }

  @override
  Float32List getTexture2D({bool alwaysReturnData = true}) {
    final ffi.Pointer<ffi.Pointer<ffi.Float>> data = calloc();
    final isTheSameAsBefore = calloc<ffi.Bool>();
    _bindings.flutter_recorder_getTexture2D(data, isTheSameAsBefore);
    if (!alwaysReturnData && isTheSameAsBefore.value) {
      calloc
        ..free(isTheSameAsBefore)
        ..free(data);
      return Float32List(0);
    }

    final val = ffi.Pointer<ffi.Float>.fromAddress(data.value.address);
    if (val == ffi.nullptr) return Float32List(512 * 256);

    calloc.free(data);
    final textureList = val.cast<ffi.Float>().asTypedList(512 * 256);

    return textureList;
  }

  @override
  double getVolumeDb() {
    final ffi.Pointer<ffi.Float> volume = calloc(4);
    _bindings.flutter_recorder_getVolumeDb(volume);
    final v = volume.value;
    calloc.free(volume);
    return v;
  }

  @override
  int getSampleRate() {
    return _bindings.flutter_recorder_getSampleRate();
  }

  @override
  int getCaptureChannels() {
    return _bindings.flutter_recorder_getCaptureChannels();
  }

  @override
  int getPlaybackChannels() {
    return _bindings.flutter_recorder_getPlaybackChannels();
  }

  @override
  int getCaptureFormat() {
    return _bindings.flutter_recorder_getCaptureFormat();
  }

  @override
  int getPlaybackFormat() {
    return _bindings.flutter_recorder_getPlaybackFormat();
  }

  @override
  int isFilterActive(RecorderFilterType filterType) {
    return _bindings
        .flutter_recorder_isFilterActive(_toGenFilterType(filterType));
  }

  @override
  void addFilter(RecorderFilterType filterType) {
    final error =
        _bindings.flutter_recorder_addFilter(_toGenFilterType(filterType));
    if (error != bindings_gen.CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(_toLocalCaptureError(error));
    }
  }

  @override
  CaptureErrors removeFilter(RecorderFilterType filterType) {
    final error =
        _bindings.flutter_recorder_removeFilter(_toGenFilterType(filterType));
    if (error != bindings_gen.CaptureErrors.captureNoError) {
      throw RecorderCppException.fromRecorderError(_toLocalCaptureError(error));
    }
    return _toLocalCaptureError(error);
  }

  @override
  List<String> getFilterParamNames(RecorderFilterType filterType) {
    final ffi.Pointer<ffi.Pointer<ffi.Char>> names =
        calloc(ffi.sizeOf<ffi.Pointer<ffi.Pointer<ffi.Char>>>() * 30);
    final ffi.Pointer<ffi.Int> paramsCount = calloc(ffi.sizeOf<ffi.Int>());
    _bindings.flutter_recorder_getFilterParamNames(
      _toGenFilterType(filterType),
      names,
      paramsCount,
    );
    final List<String> ret = [];
    for (var i = 0; i < paramsCount.value; i++) {
      final s1 = (names + i).value;
      final s = s1.cast<Utf8>().toDartString();
      ret.add(s);
      _bindings.flutter_recorder_nativeFree(s1.cast<ffi.Void>());
    }
    calloc
      ..free(names)
      ..free(paramsCount);
    return ret;
  }

  @override
  void setFilterParamValue(
    RecorderFilterType filterType,
    int attributeId,
    double value,
  ) {
    _bindings.flutter_recorder_setFilterParams(
        _toGenFilterType(filterType), attributeId, value);
  }

  @override
  double getFilterParamValue(RecorderFilterType filterType, int attributeId) {
    return _bindings.flutter_recorder_getFilterParams(
        _toGenFilterType(filterType), attributeId);
  }

  // ///////////////////////
  //   SLAVE MODE
  // ///////////////////////

  @override
  bool isSlaveAudioReady() {
    return _bindings.flutter_recorder_isSlaveAudioReady() == 1;
  }

  // ///////////////////////
  //   AEC (Adaptive Echo Cancellation)
  // ///////////////////////

  @override
  int aecCreateReferenceBuffer(int sampleRate, int channels) {
    final ptr = _bindings.flutter_recorder_aec_createReferenceBuffer(
      sampleRate,
      channels,
    );
    return ptr.address;
  }

  @override
  void aecDestroyReferenceBuffer() {
    _bindings.flutter_recorder_aec_destroyReferenceBuffer();
  }

  @override
  int aecGetOutputCallback() {
    final ptr = _bindings.flutter_recorder_aec_getOutputCallback();
    return ptr.address;
  }

  @override
  void aecResetBuffer() {
    _bindings.flutter_recorder_aec_resetBuffer();
  }

  @override
  void aecSetMode(AecMode mode) {
    _bindings.flutter_recorder_aec_setMode(mode.value);
  }

  @override
  AecMode aecGetMode() {
    final modeValue = _bindings.flutter_recorder_aec_getMode();
    try {
      return AecMode.fromValue(modeValue);
    } catch (_) {
      return AecMode.hybrid;
    }
  }

  @override
  bool aecLoadNeuralModel(NeuralModelType type, String assetBasePath) {
    final assetPathPtr = assetBasePath.toNativeUtf8();
    try {
      final res = _bindings.flutter_recorder_neural_loadModel(
        type.value,
        assetPathPtr.cast<ffi.Char>(),
      );
      return res == 1;
    } finally {
      malloc.free(assetPathPtr);
    }
  }

  @override
  NeuralModelType aecGetLoadedNeuralModel() {
    final res = _bindings.flutter_recorder_neural_getLoadedModel();
    return NeuralModelType.fromValue(res);
  }

  @override
  void aecSetNeuralEnabled(bool enabled) {
    _bindings.flutter_recorder_neural_setEnabled(enabled ? 1 : 0);
  }

  @override
  bool aecIsNeuralEnabled() {
    return _bindings.flutter_recorder_neural_isEnabled() == 1;
  }

  // ==================== AEC CALIBRATION ====================

  @override
  Uint8List aecGenerateCalibrationSignal(
    int sampleRate,
    int channels, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  }) {
    final outSize = calloc<ffi.Size>();
    try {
      final ptr = _bindings.flutter_recorder_aec_generateCalibrationSignal(
        sampleRate,
        channels,
        outSize,
        signalType.value,
      );
      final size = outSize.value;
      if (ptr.address == 0 || size == 0) {
        return Uint8List(0);
      }
      // Copy data to Dart memory
      final data = Uint8List.fromList(ptr.asTypedList(size));
      // Free native memory
      _bindings.flutter_recorder_nativeFree(ptr.cast());
      return data;
    } finally {
      calloc.free(outSize);
    }
  }

  @override
  void aecStartCalibrationCapture(int maxSamples) {
    _bindings.flutter_recorder_aec_startCalibrationCapture(maxSamples);
  }

  @override
  void aecStopCalibrationCapture() {
    _bindings.flutter_recorder_aec_stopCalibrationCapture();
  }

  @override
  void aecCaptureForAnalysis() {
    _bindings.flutter_recorder_aec_captureForAnalysis();
  }

  @override
  AecCalibrationResult aecRunCalibrationAnalysis(int sampleRate) {
    final outDelayMs = calloc<ffi.Float>();
    final outEchoGain = calloc<ffi.Float>();
    final outCorrelation = calloc<ffi.Float>();
    try {
      final success = _bindings.flutter_recorder_aec_runCalibrationAnalysis(
        sampleRate,
        outDelayMs,
        outEchoGain,
        outCorrelation,
      );
      return AecCalibrationResult(
        delayMs: outDelayMs.value,
        echoGain: outEchoGain.value,
        correlation: outCorrelation.value,
        success: success == 1,
      );
    } finally {
      calloc.free(outDelayMs);
      calloc.free(outEchoGain);
      calloc.free(outCorrelation);
    }
  }

  @override
  void aecResetCalibration() {
    _bindings.flutter_recorder_aec_resetCalibration();
  }

  @override
  AecCalibrationResultWithImpulse aecRunCalibrationWithImpulse(int sampleRate) {
    final outDelayMs = calloc<ffi.Float>();
    final outEchoGain = calloc<ffi.Float>();
    final outCorrelation = calloc<ffi.Float>();
    final outImpulseLength = calloc<ffi.Int>();
    final outCalibratedOffset = calloc<ffi.Int64>();
    try {
      final success = _bindings.flutter_recorder_aec_runCalibrationWithImpulse(
        sampleRate,
        outDelayMs,
        outEchoGain,
        outCorrelation,
        outImpulseLength,
        outCalibratedOffset,
      );
      return AecCalibrationResultWithImpulse(
        delayMs: outDelayMs.value,
        echoGain: outEchoGain.value,
        correlation: outCorrelation.value,
        success: success == 1,
        impulseLength: outImpulseLength.value,
        calibratedOffset: outCalibratedOffset.value,
      );
    } finally {
      calloc.free(outDelayMs);
      calloc.free(outEchoGain);
      calloc.free(outCorrelation);
      calloc.free(outImpulseLength);
      calloc.free(outCalibratedOffset);
    }
  }

  @override
  Float32List aecGetImpulseResponse(int maxLength) {
    final dest = calloc<ffi.Float>(maxLength);
    try {
      final actualLength = _bindings.flutter_recorder_aec_getImpulseResponse(
        dest,
        maxLength,
      );
      if (actualLength == 0) {
        return Float32List(0);
      }
      return Float32List.fromList(dest.asTypedList(actualLength));
    } finally {
      calloc.free(dest);
    }
  }

  @override
  void aecApplyImpulseResponse() {
    _bindings.flutter_recorder_aec_applyImpulseResponse();
  }

  @override
  Float32List aecGetCalibrationRefSignal(int maxLength) {
    final dest = calloc<ffi.Float>(maxLength);
    try {
      final actualLength =
          _bindings.flutter_recorder_aec_getCalibrationRefSignal(
        dest,
        maxLength,
      );
      if (actualLength == 0) {
        return Float32List(0);
      }
      return Float32List.fromList(dest.asTypedList(actualLength));
    } finally {
      calloc.free(dest);
    }
  }

  @override
  Float32List aecGetCalibrationMicSignal(int maxLength) {
    final dest = calloc<ffi.Float>(maxLength);
    try {
      final actualLength =
          _bindings.flutter_recorder_aec_getCalibrationMicSignal(
        dest,
        maxLength,
      );
      if (actualLength == 0) {
        return Float32List(0);
      }
      return Float32List.fromList(dest.asTypedList(actualLength));
    } finally {
      calloc.free(dest);
    }
  }

  @override
  void iosForceSpeakerOutput(bool enabled) {
    if (Platform.isIOS) {
      _bindings.flutter_recorder_ios_force_speaker_output(enabled);
    }
  }

  // ==================== AEC TESTING ====================

  @override
  void aecStartTestCapture(int maxSamples) {
    _bindings.flutter_recorder_aec_startTestCapture(maxSamples);
  }

  @override
  void aecStopTestCapture() {
    _bindings.flutter_recorder_aec_stopTestCapture();
  }

  @override
  AecTestResult aecRunTest(int sampleRate) {
    final outCancellationDb = calloc<ffi.Float>();
    final outCorrelationBefore = calloc<ffi.Float>();
    final outCorrelationAfter = calloc<ffi.Float>();
    final outPassed = calloc<ffi.Int>();
    final outMicEnergyDb = calloc<ffi.Float>();
    final outCancelledEnergyDb = calloc<ffi.Float>();
    try {
      final success = _bindings.flutter_recorder_aec_runTest(
        sampleRate,
        outCancellationDb,
        outCorrelationBefore,
        outCorrelationAfter,
        outPassed,
        outMicEnergyDb,
        outCancelledEnergyDb,
      );
      // success returns 1 if test ran successfully (regardless of pass/fail)
      // outPassed indicates if cancellation met threshold
      return AecTestResult(
        micEnergyDb: outMicEnergyDb.value,
        cancelledEnergyDb: outCancelledEnergyDb.value,
        cancellationDb: outCancellationDb.value,
        correlationBefore: outCorrelationBefore.value,
        correlationAfter: outCorrelationAfter.value,
        passed: outPassed.value == 1,
      );
    } finally {
      calloc.free(outCancellationDb);
      calloc.free(outCorrelationBefore);
      calloc.free(outCorrelationAfter);
      calloc.free(outPassed);
      calloc.free(outMicEnergyDb);
      calloc.free(outCancelledEnergyDb);
    }
  }

  @override
  Float32List aecGetTestMicSignal(int maxLength) {
    final dest = calloc<ffi.Float>(maxLength);
    try {
      final actualLength = _bindings.flutter_recorder_aec_getTestMicSignal(
        dest,
        maxLength,
      );
      if (actualLength == 0) {
        return Float32List(0);
      }
      return Float32List.fromList(dest.asTypedList(actualLength));
    } finally {
      calloc.free(dest);
    }
  }

  @override
  Float32List aecGetTestCancelledSignal(int maxLength) {
    final dest = calloc<ffi.Float>(maxLength);
    try {
      final actualLength =
          _bindings.flutter_recorder_aec_getTestCancelledSignal(
        dest,
        maxLength,
      );
      if (actualLength == 0) {
        return Float32List(0);
      }
      return Float32List.fromList(dest.asTypedList(actualLength));
    } finally {
      calloc.free(dest);
    }
  }

  @override
  void aecResetTest() {
    _bindings.flutter_recorder_aec_resetTest();
  }

  // ==================== VSS-NLMS PARAMETER CONTROL ====================

  @override
  void aecSetVssMuMax(double mu) {
    _bindings.flutter_recorder_aec_setVssMuMax(mu);
  }

  @override
  void aecSetVssLeakage(double lambda) {
    _bindings.flutter_recorder_aec_setVssLeakage(lambda);
  }

  @override
  void aecSetVssAlpha(double alpha) {
    _bindings.flutter_recorder_aec_setVssAlpha(alpha);
  }

  @override
  double aecGetVssMuMax() {
    return _bindings.flutter_recorder_aec_getVssMuMax();
  }

  @override
  double aecGetVssLeakage() {
    return _bindings.flutter_recorder_aec_getVssLeakage();
  }

  @override
  double aecGetVssAlpha() {
    return _bindings.flutter_recorder_aec_getVssAlpha();
  }

  // ==================== AEC FILTER LENGTH CONTROL ====================

  @override
  void aecSetFilterLength(int length) {
    _bindings.flutter_recorder_aec_setFilterLength(length);
  }

  @override
  int aecGetFilterLength() {
    return _bindings.flutter_recorder_aec_getFilterLength();
  }

  // ==================== AEC CALIBRATION LOGGING ====================

  @override
  String aecGetCalibrationLog() {
    final ptr = _bindings.flutter_recorder_aec_getCalibrationLog();
    if (ptr == ffi.nullptr) {
      return '';
    }
    return ptr.cast<Utf8>().toDartString();
  }

  @override
  void aecClearCalibrationLog() {
    _bindings.flutter_recorder_aec_clearCalibrationLog();
  }

  // ==================== AEC POSITION-BASED SYNC ====================

  @override
  int aecGetOutputFrameCount() {
    return _bindings.flutter_recorder_aec_getOutputFrameCount();
  }

  @override
  int aecGetCaptureFrameCount() {
    return _bindings.flutter_recorder_aec_getCaptureFrameCount();
  }

  @override
  void aecRecordCalibrationFrameCounters() {
    _bindings.flutter_recorder_aec_recordCalibrationFrameCounters();
  }

  @override
  void aecSetCalibratedOffset(int offset) {
    _bindings.flutter_recorder_aec_setCalibratedOffset(offset);
  }

  @override
  int aecGetCalibratedOffset() {
    return _bindings.flutter_recorder_aec_getCalibratedOffset();
  }

  // ==================== ALIGNED CALIBRATION CAPTURE ====================

  @override
  void aecStartAlignedCalibrationCapture(int maxSamples) {
    _bindings.flutter_recorder_aec_startAlignedCalibrationCapture(maxSamples);
  }

  @override
  void aecStopAlignedCalibrationCapture() {
    _bindings.flutter_recorder_aec_stopAlignedCalibrationCapture();
  }

  @override
  AecCalibrationResultWithImpulse aecRunAlignedCalibrationWithImpulse(
    int sampleRate, {
    CalibrationSignalType signalType = CalibrationSignalType.chirp,
  }) {
    return using((arena) {
      final delaySamplesPtr = arena<ffi.Int>();
      final delayMsPtr = arena<ffi.Float>();
      final gainPtr = arena<ffi.Float>();
      final correlationPtr = arena<ffi.Float>();
      final impulseLengthPtr = arena<ffi.Int>();
      final calibratedOffsetPtr = arena<ffi.Int64>();

      final success =
          _bindings.flutter_recorder_aec_runAlignedCalibrationWithImpulse(
        sampleRate,
        delaySamplesPtr,
        delayMsPtr,
        gainPtr,
        correlationPtr,
        impulseLengthPtr,
        calibratedOffsetPtr,
        signalType.value,
      );

      return AecCalibrationResultWithImpulse(
        delayMs: delayMsPtr.value,
        echoGain: gainPtr.value,
        correlation: correlationPtr.value,
        success: success != 0,
        impulseLength: impulseLengthPtr.value,
        calibratedOffset: calibratedOffsetPtr.value,
      );
    });
  }
}
