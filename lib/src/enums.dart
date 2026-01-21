// ignore_for_file: sort_constructors_first, public_member_api_docs

/// CaptureDevice exposed to Dart
final class CaptureDevice {
  /// Constructs a new [CaptureDevice].
  // ignore: avoid_positional_boolean_parameters
  const CaptureDevice(this.name, this.isDefault, this.id);

  /// The name of the device.
  final String name;

  /// Whether this is the default capture device.
  final bool isDefault;

  /// The ID of the device.
  final int id;
}

/// Possible capture errors
enum CaptureErrors {
  /// No error
  captureNoError(0),

  /// The capture device has failed to initialize.
  captureInitFailed(1),

  /// The capture device has not yet been initialized.
  captureNotInited(2),

  /// Failed to start the device.
  failedToStartDevice(3),

  /// Failed to initialize wav recording.
  failedToInitializeRecording(4),

  /// Invalid arguments while initializing wav recording.
  invalidArgs(5),

  /// Failed to write wav file.
  failedToWriteWav(6),

  /// Filter not found
  filterNotFound(7),

  /// The filter has already been added.
  filterAlreadyAdded(8),

  /// Error getting filter parameter.
  filterParameterGetError(9);

  /// Internal value
  final int value;

  /// Create a [CaptureErrors] from an internal value
  const CaptureErrors(this.value);

  static CaptureErrors fromValue(int value) => switch (value) {
        0 => captureNoError,
        1 => captureInitFailed,
        2 => captureNotInited,
        3 => failedToStartDevice,
        4 => failedToInitializeRecording,
        5 => invalidArgs,
        6 => failedToWriteWav,
        7 => filterNotFound,
        8 => filterAlreadyAdded,
        9 => filterParameterGetError,
        _ => throw ArgumentError('Unknown value for CaptureErrors: $value'),
      };
}

/// The channels to be used while initializing the player.
enum RecorderChannels {
  /// One channel.
  mono(1),

  /// Two channels.
  stereo(2);

  const RecorderChannels(this.count);

  /// The channels count.
  final int count;
}

/// The PCM format
enum PCMFormat {
  /// 8-bit unsigned.
  u8(0),

  /// 16-bit signed, little-endian.
  s16le(1),

  /// 24-bit signed, little-endian.
  s24le(2),

  /// 32-bit signed, little-endian.
  s32le(3),

  /// 32-bit float, little-endian.
  f32le(4),

  /// Unknown format - let the system choose optimal (AAudio best practice).
  unknown(5);

  final int value;

  const PCMFormat(this.value);

  static PCMFormat fromValue(int value) => switch (value) {
        0 => u8,
        1 => s16le,
        2 => s24le,
        3 => s32le,
        4 => f32le,
        5 => unknown,
        _ => throw ArgumentError('Unknown value for PCMFormat: $value'),
      };
}

/// Statistics from the Acoustic Echo Cancellation filter.
class AecStats {
  /// Constructs a new [AecStats].
  const AecStats({
    required this.maxAttenuationDb,
    required this.correlation,
    required this.echoReturnLossDb,
    this.filterLength = 8192,
    this.muMax = 0.5,
    this.muEffective = 0.0,
    this.lastErrorDb = -100.0,
    this.instantCorrelation = 0.0,
  });

  /// Maximum attenuation achieved in dB.
  final double maxAttenuationDb;

  /// Correlation between reference and mic signal.
  final double correlation;

  /// Echo Return Loss in dB.
  final double echoReturnLossDb;

  /// Current filter length in samples.
  final int filterLength;

  /// Configured maximum step size.
  final double muMax;

  /// Last effective step size (runtime).
  final double muEffective;

  /// Last error in dB.
  final double lastErrorDb;

  /// Instantaneous correlation metric.
  final double instantCorrelation;

  @override
  String toString() {
    return 'AecStats(ERL: ${echoReturnLossDb.toStringAsFixed(2)} dB, '
        'corr: ${correlation.toStringAsFixed(3)}, '
        'filterLen: $filterLength, '
        'muMax: ${muMax.toStringAsFixed(2)}, '
        'muEff: ${muEffective.toStringAsFixed(3)})';
  }
}

/// AEC Mode for A/B testing
enum AecMode {
  /// Raw microphone input (no AEC)
  bypass(0),

  /// Adaptive NLMS (may cause artifacts on transients)
  algo(1),

  /// Neural (DTLN-AEC) only
  neural(2),

  /// Hybrid: Adaptive NLMS + Neural Post-Filter
  hybrid(3),

  /// Frozen FIR: Pure calibrated IR, no adaptation (stable for transients)
  frozen(4),

  /// Frozen FIR + Neural Post-Filter (best for transients with neural cleanup)
  frozenNeural(5);

  final int value;
  const AecMode(this.value);

  static AecMode fromValue(int value) => switch (value) {
        0 => bypass,
        1 => algo,
        2 => neural,
        3 => hybrid,
        4 => frozen,
        5 => frozenNeural,
        _ => throw ArgumentError('Unknown value for AecMode: $value'),
      };
}

/// Neural model types for AEC
enum NeuralModelType {
  /// No neural model loaded
  none(0),

  /// DTLN-AEC 48kHz model
  dtlnAec48k(1),

  /// LSTM-based AEC model
  lstmV1(2);

  final int value;
  const NeuralModelType(this.value);

  static NeuralModelType fromValue(int value) => switch (value) {
        0 => none,
        1 => dtlnAec48k,
        2 => lstmV1,
        _ => throw ArgumentError('Unknown value for NeuralModelType: $value'),
      };
}

/// Calibration signal type for AEC
enum CalibrationSignalType {
  /// Logarithmic sine sweep (chirp) - good for frequency response
  chirp(0),

  /// Click train (impulse) - good for transient response, direct IR measurement
  click(1);

  final int value;
  const CalibrationSignalType(this.value);
}
