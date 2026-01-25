#ifndef ENUMS_H
#define ENUMS_H

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
  pcm_f32
} PCMFormatInternal_t;

typedef enum RecorderFilterType {
  autogain,
  echoCancellation,
  adaptiveEchoCancellation
} FilterType_t;

typedef enum AecMode {
  aecModeBypass = 0,
  aecModeAlgo = 1,
  aecModeNeural = 2,
  aecModeHybrid = 3
} AecMode_t;

typedef struct {
  float maxAttenuationDb;
  float avgAttenuationDb;
  float minAttenuationDb;
  float correlation;
  float echoReturnLossDb;
  float refEnergy;
  float micEnergy;
  // Debug display fields
  int filterLength;         // Current filter length in samples
  float muMax;              // Configured max step size
  float muEffective;        // Last effective step size (runtime)
  float lastErrorDb;        // Last error in dB
  float instantCorrelation; // Instantaneous correlation metric
} AecStats;

#endif // ENUMS_H