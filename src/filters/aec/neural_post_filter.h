#ifndef NEURAL_POST_FILTER_H
#define NEURAL_POST_FILTER_H

#include <string>
#include <vector>

// USE_TFLITE is defined by CMake/build system:
// - Android: via CMakeLists.txt (Google Play Services TFLite)
// - iOS/macOS: via CMakeLists.txt (LiteRT library)
// - Windows/Linux: via CMakeLists.txt (if LiteRT source present)

#ifdef USE_TFLITE
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#endif

// Neural model types for runtime selection
enum class NeuralModelType {
  NONE = 0,         // No neural processing (bypass mode)
  AEC_MASK_V3 = 1,  // AEC mask v3 (1M params, 16-frame context)
  ERB_DF_V3 = 4     // ERB deep-filter v3 distill2 (700K params, stateful GRU,
                    // 5-tap complex deep filter) - erb_df_v3.tflite
};

/**
 * Neural Post-Filter for Acoustic Echo Cancellation.
 *
 * This class acts as a wrapper around a LiteRT CompiledModel (e.g., DTLN-aec)
 * to handle residual echo and non-linear distortions after the primary
 * linear AEC stage.
 *
 * Integration:
 * Mic -> NLMS Filter (Linear AEC) -> Neural Post-Filter (Residual AEC) ->
 * Looper
 */
class NeuralPostFilter {
public:
  NeuralPostFilter(unsigned int sampleRate, unsigned int channels);
  ~NeuralPostFilter();

  /**
   * Initializes the LiteRT CompiledModel and loads the model.
   * @param modelPath Path to the .tflite model file.
   * @return true if successful.
   */
  bool loadModel(const std::string &modelPath);

  /**
   * Loads a model by type from the asset bundle.
   * @param modelType The type of model to load.
   * @param assetBasePath Base path to assets directory (platform-specific).
   * @return true if successful.
   */
  bool loadModelByType(NeuralModelType modelType,
                       const std::string &assetBasePath);

  /**
   * Processes a block of audio.
   * @param micSignal The microphone signal (after linear AEC). May be
   *        interleaved; channel 0 is processed (capture channels are
   *        duplicates) and the result is written to every output channel.
   * @param refSignal The reference loopback signal (same layout).
   * @param output The clean output signal (may alias micSignal).
   * @param frameCount Number of FRAMES (not samples) to process.
   * @param channels Interleaved channel count (1 = mono).
   */
  void process(const float *micSignal, const float *refSignal, float *output,
               unsigned int frameCount, unsigned int channels = 1);

  /**
   * Resets the internal state (LSTM hidden states, etc.).
   */
  void reset();

  void setEnabled(bool enabled) { mEnabled = enabled; }
  bool isEnabled() const { return mEnabled; }

  NeuralModelType getLoadedModelType() const { return mCurrentModelType; }

private:
  static constexpr int N_FFT = 1024;
  static constexpr int HOP_SIZE = 256;
  static constexpr int N_BINS = (N_FFT / 2 + 1);      // 513
  static constexpr int CONTEXT_FRAMES = 16;
  static constexpr int DF_ORDER = 5;                  // ERB_DF_V3 taps
  static constexpr int GRU_UNITS = 256;               // ERB_DF_V3 state size

  void processSingleStage(const float *micSignal, const float *refSignal,
                          float *output, unsigned int frameCount);
  void processErbDeepFilter();

  // STFT Processing
  void performSTFT(const float *micBlock, const float *refBlock);
  void performIFFT(float *outputBlock);
  void performIFFTComplex(float *outputBlock);

  unsigned int mSampleRate;
  unsigned int mChannels;
  bool mEnabled;
  bool mIsLoaded;
  NeuralModelType mCurrentModelType;

  // STFT Buffers and State
  std::vector<float> mWindow;
  std::vector<float> mInputBufferMic;
  std::vector<float> mInputBufferLpb;
  std::vector<float> mOutputAccumulator;
  std::vector<float> mFFTWorkBuffer;

  // Model features
  std::vector<float> mMagMic;
  std::vector<float> mPhaseMic;
  std::vector<float> mMagLpb;

  unsigned int mWindowPos = 0;

  // v3 context buffer: ring buffer for 16 frames of [mic_mag, lpb_mag]
  std::vector<float> mContextBuffer;  // [CONTEXT_FRAMES * N_BINS * 2]
  size_t mContextWritePos = 0;
  size_t mContextFrameCount = 0;

  // ERB_DF_V3 state. The model was trained with tf.signal STFT (periodic
  // Hann analysis, OLA-normalized synthesis) so this path uses its own
  // windows rather than the symmetric Hann above.
  std::vector<float> mWindowPeriodic;   // periodic Hann analysis window
  std::vector<float> mSynthWindow;      // inverse_stft_window_fn equivalent
  std::vector<float> mMicRe, mMicIm;    // current frame complex spectra
  std::vector<float> mLpbRe, mLpbIm;
  std::vector<float> mEnhRe, mEnhIm;    // mic - predicted bleed
  std::vector<float> mLpbRingRe;        // [DF_ORDER * N_BINS] past lpb frames
  std::vector<float> mLpbRingIm;
  size_t mLpbRingPos = 0;               // slot holding the CURRENT frame
  std::vector<float> mGru1State;        // [GRU_UNITS], carried across frames
  std::vector<float> mGru2State;

#ifdef USE_TFLITE
  // LiteRT C API members
  LiteRtEnvironment mEnv = nullptr;
  LiteRtModel mModel = nullptr;
  LiteRtCompiledModel mCompiledModel = nullptr;

  // Pre-allocated working buffers and tensor buffers
  std::vector<float> mWorkingBuffer;
  std::vector<LiteRtTensorBuffer> mInputBuffers;
  std::vector<LiteRtTensorBuffer> mOutputBuffers;
  std::vector<void*> mHostMemory;  // Zero-copy host memory allocations

  // Tensor order in the flatbuffer is arbitrary, so ERB_DF_V3 packs its I/O
  // into shape-unique tensors: in spec(1,4,513)+states(1,512), out
  // taps(1,1,513,10)+states(1,512). Identify by element count.
  std::vector<size_t> mInputElems;    // parallel to mInputBuffers
  std::vector<size_t> mOutputElems;   // parallel to mOutputBuffers
  std::vector<void*> mInputHostMem;   // parallel to mInputBuffers
  std::vector<void*> mOutputHostMem;  // parallel to mOutputBuffers
  float *inputByElems(size_t elems) const;
  float *outputByElems(size_t elems) const;
#endif // USE_TFLITE
};

#endif // NEURAL_POST_FILTER_H
