#include "neural_post_filter.h"
extern void aecLog(const char *fmt, ...);
#include <cstring>
#include <cstdlib>

// Android API < 28 doesn't have aligned_alloc, use posix_memalign instead
#if defined(__ANDROID__) && __ANDROID_API__ < 28
static inline void* portable_aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}
#define aligned_alloc portable_aligned_alloc
#endif

#ifdef USE_TFLITE
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#endif

#include "../../fft/soloud_fft.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

NeuralPostFilter::NeuralPostFilter(unsigned int sampleRate,
                                   unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels), mEnabled(true),
      mIsLoaded(false), mCurrentModelType(NeuralModelType::NONE) {

  // Initialize STFT buffers
  mWindow.resize(N_FFT);
  for (int i = 0; i < N_FFT; ++i) {
    // Hann window
    mWindow[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N_FFT - 1)));
  }

  mInputBufferMic.assign(N_FFT, 0.0f);
  mInputBufferLpb.assign(N_FFT, 0.0f);
  mOutputAccumulator.assign(N_FFT, 0.0f);
  mFFTWorkBuffer.assign(N_FFT * 2, 0.0f); // Complex interleaved

  mMagMic.resize(N_BINS);
  mPhaseMic.resize(N_BINS);
  mMagLpb.resize(N_BINS);

  // Initialize v3 context buffer: 16 frames × (513 mic + 513 lpb)
  mContextBuffer.assign(CONTEXT_FRAMES * N_BINS * 2, 0.0f);
  mContextWritePos = 0;
  mContextFrameCount = 0;

  // ERB_DF_V3: periodic Hann analysis (tf.signal.stft convention -- divisor
  // N, not N-1) and OLA-normalized synthesis (inverse_stft_window_fn):
  // synth[i] = w[i] / sum_m w^2[i mod HOP + m*HOP].
  mWindowPeriodic.resize(N_FFT);
  for (int i = 0; i < N_FFT; ++i) {
    mWindowPeriodic[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / N_FFT));
  }
  mSynthWindow.resize(N_FFT);
  for (int i = 0; i < N_FFT; ++i) {
    float denom = 0.0f;
    for (int j = i % HOP_SIZE; j < N_FFT; j += HOP_SIZE) {
      denom += mWindowPeriodic[j] * mWindowPeriodic[j];
    }
    mSynthWindow[i] = denom > 1e-12f ? mWindowPeriodic[i] / denom : 0.0f;
  }
  mMicRe.assign(N_BINS, 0.0f);
  mMicIm.assign(N_BINS, 0.0f);
  mLpbRe.assign(N_BINS, 0.0f);
  mLpbIm.assign(N_BINS, 0.0f);
  mEnhRe.assign(N_BINS, 0.0f);
  mEnhIm.assign(N_BINS, 0.0f);
  mLpbRingRe.assign(DF_ORDER * N_BINS, 0.0f);
  mLpbRingIm.assign(DF_ORDER * N_BINS, 0.0f);
  mGru1State.assign(GRU_UNITS, 0.0f);
  mGru2State.assign(GRU_UNITS, 0.0f);

#ifdef USE_TFLITE
  // Initialize LiteRT Environment
  if (LiteRtCreateEnvironment(0, nullptr, &mEnv) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to create LiteRT environment\n");
  }
#endif
}

NeuralPostFilter::~NeuralPostFilter() {
#ifdef USE_TFLITE
  reset();
  if (mEnv) {
    LiteRtDestroyEnvironment(mEnv);
    mEnv = nullptr;
  }
#endif
}

void NeuralPostFilter::reset() {
  // Reset context buffer state
  std::fill(mContextBuffer.begin(), mContextBuffer.end(), 0.0f);
  mContextWritePos = 0;
  mContextFrameCount = 0;

  // Reset ERB_DF_V3 streaming state
  std::fill(mLpbRingRe.begin(), mLpbRingRe.end(), 0.0f);
  std::fill(mLpbRingIm.begin(), mLpbRingIm.end(), 0.0f);
  std::fill(mGru1State.begin(), mGru1State.end(), 0.0f);
  std::fill(mGru2State.begin(), mGru2State.end(), 0.0f);
  mLpbRingPos = 0;

#ifdef USE_TFLITE
  // Destroy existing buffers and models
  for (auto &buffer : mInputBuffers) {
    if (buffer)
      LiteRtDestroyTensorBuffer(buffer);
  }
  mInputBuffers.clear();

  for (auto &buffer : mOutputBuffers) {
    if (buffer)
      LiteRtDestroyTensorBuffer(buffer);
  }
  mOutputBuffers.clear();

  // Free host memory (after destroying tensor buffers that reference it)
  for (auto &mem : mHostMemory) {
    if (mem)
      free(mem);
  }
  mHostMemory.clear();

  mInputElems.clear();
  mOutputElems.clear();
  mInputHostMem.clear();
  mOutputHostMem.clear();

  if (mCompiledModel) {
    LiteRtDestroyCompiledModel(mCompiledModel);
    mCompiledModel = nullptr;
  }

  if (mModel) {
    LiteRtDestroyModel(mModel);
    mModel = nullptr;
  }

  mIsLoaded = false;
#endif
}

bool NeuralPostFilter::loadModel(const std::string &modelPath) {
#ifndef USE_TFLITE
  aecLog("[NeuralPostFilter] TFLite not enabled in build\n");
  return false;
#else
  if (!mEnv)
    return false;

  reset();

  aecLog("[NeuralPostFilter] Loading model from: %s\n", modelPath.c_str());

  if (LiteRtCreateModelFromFile(modelPath.c_str(), &mModel) !=
      kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to load model file\n");
    return false;
  }

  // Create compilation options with CPU accelerator
  LiteRtOptions options = nullptr;
  if (LiteRtCreateOptions(&options) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to create compilation options\n");
    return false;
  }
  // CPU only. The GPU accelerator path crashed at model load on macOS
  // (Metal-on-Intel), and these models are far below CPU real-time budget
  // anyway (~0.24ms/frame vs 5.33ms for ERB_DF_V3).
  LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu);

  // Compile the model
  LiteRtStatus compileStatus = LiteRtCreateCompiledModel(mEnv, mModel, options, &mCompiledModel);
  LiteRtDestroyOptions(options);

  if (compileStatus != kLiteRtStatusOk) {
    const char* errorName = "unknown";
    switch (compileStatus) {
      case kLiteRtStatusErrorInvalidArgument: errorName = "InvalidArgument"; break;
      case kLiteRtStatusErrorMemoryAllocationFailure: errorName = "MemoryAllocationFailure"; break;
      case kLiteRtStatusErrorRuntimeFailure: errorName = "RuntimeFailure"; break;
      case kLiteRtStatusErrorUnsupported: errorName = "Unsupported"; break;
      case kLiteRtStatusErrorCompilation: errorName = "Compilation"; break;
      case kLiteRtStatusErrorDynamicLoading: errorName = "DynamicLoading"; break;
      default: break;
    }
    aecLog("[NeuralPostFilter] Failed to compile model: %s (status=%d)\n", errorName, compileStatus);
    return false;
  }

  // Set up input and output buffers using zero-copy host memory
  LiteRtSignature signature;
  if (LiteRtGetModelSignature(mModel, 0, &signature) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to get model signature\n");
    return false;
  }

  LiteRtParamIndex numInputs = 0;
  if (LiteRtGetNumSignatureInputs(signature, &numInputs) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to get num signature inputs\n");
    return false;
  }
  aecLog("[NeuralPostFilter] Model has %u inputs\n", numInputs);

  for (uint32_t i = 0; i < numInputs; ++i) {
    LiteRtTensor tensor;
    if (LiteRtGetSignatureInputTensorByIndex(signature, i, &tensor) != kLiteRtStatusOk) {
      aecLog("[NeuralPostFilter] Failed to get input tensor %u\n", i);
      continue;
    }

    LiteRtRankedTensorType type;
    if (LiteRtGetRankedTensorType(tensor, &type) != kLiteRtStatusOk) {
      aecLog("[NeuralPostFilter] Failed to get tensor type for input %u\n", i);
      continue;
    }

    // Calculate buffer size from tensor dimensions
    // Handle dynamic dimensions (-1) by treating them as 1 (batch size)
    size_t numElements = 1;
    for (int32_t d = 0; d < type.layout.rank; ++d) {
      int32_t dim = type.layout.dimensions[d];
      if (dim <= 0) {
        // Dynamic dimension - use 1 (we process single batches)
        dim = 1;
      }
      numElements *= (size_t)dim;
    }
    size_t bufferSize = numElements * sizeof(float); // Assuming float32

    aecLog("[NeuralPostFilter] Input %u: rank=%d, elements=%zu, size=%zu\n",
           i, type.layout.rank, numElements, bufferSize);

    // Allocate aligned host memory for zero-copy
    // aligned_alloc requires size to be a multiple of alignment (enforced on
    // macOS -- returns NULL otherwise). Pad up; the tensor type still defines
    // the logical size.
    size_t alignedSize =
        (bufferSize + LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1) &
        ~((size_t)LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1);
    void* hostMem = aligned_alloc(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, alignedSize);
    if (!hostMem) {
      aecLog("[NeuralPostFilter] Failed to allocate host memory for input %u\n", i);
      continue;
    }
    memset(hostMem, 0, alignedSize);
    mHostMemory.push_back(hostMem);

    LiteRtTensorBuffer buffer;
    if (LiteRtCreateTensorBufferFromHostMemory(&type, hostMem, alignedSize,
                                                nullptr, &buffer) == kLiteRtStatusOk) {
      mInputBuffers.push_back(buffer);
      mInputElems.push_back(numElements);
      mInputHostMem.push_back(hostMem);
    } else {
      aecLog("[NeuralPostFilter] Failed to create input buffer %u\n", i);
    }
  }

  LiteRtParamIndex numOutputs = 0;
  if (LiteRtGetNumSignatureOutputs(signature, &numOutputs) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to get num signature outputs\n");
    return false;
  }
  aecLog("[NeuralPostFilter] Model has %u outputs\n", numOutputs);

  for (uint32_t i = 0; i < numOutputs; ++i) {
    LiteRtTensor tensor;
    if (LiteRtGetSignatureOutputTensorByIndex(signature, i, &tensor) != kLiteRtStatusOk) {
      aecLog("[NeuralPostFilter] Failed to get output tensor %u\n", i);
      continue;
    }

    LiteRtRankedTensorType type;
    if (LiteRtGetRankedTensorType(tensor, &type) != kLiteRtStatusOk) {
      aecLog("[NeuralPostFilter] Failed to get tensor type for output %u\n", i);
      continue;
    }

    // Handle dynamic dimensions (-1) by treating them as 1 (batch size)
    size_t numElements = 1;
    for (int32_t d = 0; d < type.layout.rank; ++d) {
      int32_t dim = type.layout.dimensions[d];
      if (dim <= 0) {
        // Dynamic dimension - use 1 (we process single batches)
        dim = 1;
      }
      numElements *= (size_t)dim;
    }
    size_t bufferSize = numElements * sizeof(float);

    aecLog("[NeuralPostFilter] Output %u: rank=%d, elements=%zu, size=%zu\n",
           i, type.layout.rank, numElements, bufferSize);

    // Allocate aligned host memory for zero-copy
    // aligned_alloc requires size to be a multiple of alignment (enforced on
    // macOS -- returns NULL otherwise). Pad up; the tensor type still defines
    // the logical size.
    size_t alignedSize =
        (bufferSize + LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1) &
        ~((size_t)LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1);
    void* hostMem = aligned_alloc(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, alignedSize);
    if (!hostMem) {
      aecLog("[NeuralPostFilter] Failed to allocate host memory for output %u\n", i);
      continue;
    }
    memset(hostMem, 0, alignedSize);
    mHostMemory.push_back(hostMem);

    LiteRtTensorBuffer buffer;
    if (LiteRtCreateTensorBufferFromHostMemory(&type, hostMem, alignedSize,
                                                nullptr, &buffer) == kLiteRtStatusOk) {
      mOutputBuffers.push_back(buffer);
      mOutputElems.push_back(numElements);
      mOutputHostMem.push_back(hostMem);
    } else {
      aecLog("[NeuralPostFilter] Failed to create output buffer %u\n", i);
    }
  }

  if (mInputBuffers.empty() || mOutputBuffers.empty()) {
    aecLog("[NeuralPostFilter] Failed to create I/O buffers (in=%zu, out=%zu)\n",
           mInputBuffers.size(), mOutputBuffers.size());
    reset();
    return false;
  }

  mIsLoaded = true;
  aecLog("[NeuralPostFilter] Model loaded and compiled successfully\n");
  return true;
#endif
}

bool NeuralPostFilter::loadModelByType(NeuralModelType modelType,
                                       const std::string &assetBasePath) {
  mCurrentModelType = modelType;
  if (modelType == NeuralModelType::NONE) {
    reset();
    return true;
  }

  // If assetBasePath already points to a .tflite file, use it directly
  if (assetBasePath.size() > 7 &&
      assetBasePath.substr(assetBasePath.size() - 7) == ".tflite") {
    return loadModel(assetBasePath);
  }

  std::string fileName = (modelType == NeuralModelType::ERB_DF_V3)
                             ? "erb_df_v3.tflite"
                             : "aec_mask_v3.tflite";

  std::string fullPath = assetBasePath;
  if (!fullPath.empty() && fullPath.back() != '/')
    fullPath += "/";
  fullPath += fileName;

  return loadModel(fullPath);
}

#ifdef USE_TFLITE
float *NeuralPostFilter::inputByElems(size_t elems) const {
  for (size_t i = 0; i < mInputElems.size(); ++i) {
    if (mInputElems[i] == elems)
      return (float *)mInputHostMem[i];
  }
  return nullptr;
}

float *NeuralPostFilter::outputByElems(size_t elems) const {
  for (size_t i = 0; i < mOutputElems.size(); ++i) {
    if (mOutputElems[i] == elems)
      return (float *)mOutputHostMem[i];
  }
  return nullptr;
}
#endif // USE_TFLITE

// STFT Processing Implementation
void NeuralPostFilter::performSTFT(const float *micBlock,
                                   const float *refBlock) {
  if (mCurrentModelType == NeuralModelType::ERB_DF_V3) {
    // Full-length complex STFT with periodic Hann, matching the
    // tf.signal.stft the model was trained against. NOTE: FFT::fft's length
    // argument is the FLOAT count (2 floats per complex point).
    for (int i = 0; i < N_FFT; ++i) {
      mFFTWorkBuffer[2 * i] = micBlock[i] * mWindowPeriodic[i];
      mFFTWorkBuffer[2 * i + 1] = 0.0f;
    }
    FFT::fft(mFFTWorkBuffer.data(), 2 * N_FFT);
    for (int i = 0; i < N_BINS; ++i) {
      mMicRe[i] = mFFTWorkBuffer[2 * i];
      mMicIm[i] = mFFTWorkBuffer[2 * i + 1];
    }
    for (int i = 0; i < N_FFT; ++i) {
      mFFTWorkBuffer[2 * i] = refBlock[i] * mWindowPeriodic[i];
      mFFTWorkBuffer[2 * i + 1] = 0.0f;
    }
    FFT::fft(mFFTWorkBuffer.data(), 2 * N_FFT);
    for (int i = 0; i < N_BINS; ++i) {
      mLpbRe[i] = mFFTWorkBuffer[2 * i];
      mLpbIm[i] = mFFTWorkBuffer[2 * i + 1];
    }
    return;
  }

  // 1. Mic Stage
  for (int i = 0; i < N_FFT; ++i) {
    mFFTWorkBuffer[2 * i] = micBlock[i] * mWindow[i];
    mFFTWorkBuffer[2 * i + 1] = 0.0f;
  }
  FFT::fft(mFFTWorkBuffer.data(), N_FFT);
  for (int i = 0; i < N_BINS; ++i) {
    float re = mFFTWorkBuffer[2 * i];
    float im = mFFTWorkBuffer[2 * i + 1];
    mMagMic[i] = std::sqrt(re * re + im * im);
    mPhaseMic[i] = std::atan2(im, re);
  }

  // 2. Ref Stage
  for (int i = 0; i < N_FFT; ++i) {
    mFFTWorkBuffer[2 * i] = refBlock[i] * mWindow[i];
    mFFTWorkBuffer[2 * i + 1] = 0.0f;
  }
  FFT::fft(mFFTWorkBuffer.data(), N_FFT);
  for (int i = 0; i < N_BINS; ++i) {
    float re = mFFTWorkBuffer[2 * i];
    float im = mFFTWorkBuffer[2 * i + 1];
    mMagLpb[i] = std::sqrt(re * re + im * im);
  }
}

void NeuralPostFilter::performIFFTComplex(float *outputBlock) {
  // Enhanced complex spectrum -> time block, OLA-normalized synthesis window.
  for (int i = 0; i < N_BINS; ++i) {
    mFFTWorkBuffer[2 * i] = mEnhRe[i];
    mFFTWorkBuffer[2 * i + 1] = mEnhIm[i];
  }
  for (int i = N_BINS; i < N_FFT; ++i) {
    int mirrorIdx = N_FFT - i;
    mFFTWorkBuffer[2 * i] = mFFTWorkBuffer[2 * mirrorIdx];
    mFFTWorkBuffer[2 * i + 1] = -mFFTWorkBuffer[2 * mirrorIdx + 1];
  }
  FFT::ifft(mFFTWorkBuffer.data(), 2 * N_FFT);
  for (int i = 0; i < N_FFT; ++i) {
    outputBlock[i] = mFFTWorkBuffer[2 * i] * mSynthWindow[i];
  }
}

void NeuralPostFilter::performIFFT(float *outputBlock) {
  for (int i = 0; i < N_BINS; ++i) {
    mFFTWorkBuffer[2 * i] = mMagMic[i] * std::cos(mPhaseMic[i]);
    mFFTWorkBuffer[2 * i + 1] = mMagMic[i] * std::sin(mPhaseMic[i]);
  }
  for (int i = N_BINS; i < N_FFT; ++i) {
    int mirrorIdx = N_FFT - i;
    mFFTWorkBuffer[2 * i] = mFFTWorkBuffer[2 * mirrorIdx];
    mFFTWorkBuffer[2 * i + 1] = -mFFTWorkBuffer[2 * mirrorIdx + 1];
  }
  FFT::ifft(mFFTWorkBuffer.data(), N_FFT);
  for (int i = 0; i < N_FFT; ++i) {
    outputBlock[i] = mFFTWorkBuffer[2 * i] * mWindow[i];
  }
}

void NeuralPostFilter::process(const float *micSignal, const float *refSignal,
                               float *output, unsigned int frameCount,
                               unsigned int channels) {
  if (channels == 0)
    channels = 1;
  if (!mEnabled || !mIsLoaded) {
    // Only copy if buffers are different; memcpy with overlapping buffers is UB
    if (output != micSignal) {
      std::memcpy(output, micSignal, frameCount * channels * sizeof(float));
    }
    return;
  }

  for (unsigned int f = 0; f < frameCount; ++f) {
    std::memmove(mInputBufferMic.data(), mInputBufferMic.data() + 1,
                 (N_FFT - 1) * sizeof(float));
    mInputBufferMic[N_FFT - 1] = micSignal[f * channels];
    std::memmove(mInputBufferLpb.data(), mInputBufferLpb.data() + 1,
                 (N_FFT - 1) * sizeof(float));
    mInputBufferLpb[N_FFT - 1] = refSignal[f * channels];

    mWindowPos++;
    if (mWindowPos >= HOP_SIZE) {
      mWindowPos = 0;
      performSTFT(mInputBufferMic.data(), mInputBufferLpb.data());
      std::vector<float> synth(N_FFT);
      if (mCurrentModelType == NeuralModelType::ERB_DF_V3) {
        processErbDeepFilter();
        performIFFTComplex(synth.data());
      } else {
        processSingleStage(nullptr, nullptr, nullptr, 0);
        performIFFT(synth.data());
      }
      for (int i = 0; i < N_FFT; ++i)
        mOutputAccumulator[i] += synth[i];
    }
    for (unsigned int c = 0; c < channels; ++c)
      output[f * channels + c] = mOutputAccumulator[0];
    std::memmove(mOutputAccumulator.data(), mOutputAccumulator.data() + 1,
                 (N_FFT - 1) * sizeof(float));
    mOutputAccumulator[N_FFT - 1] = 0.0f;
  }
}

void NeuralPostFilter::processErbDeepFilter() {
  // Default to passthrough; overwritten on successful inference.
  std::memcpy(mEnhRe.data(), mMicRe.data(), N_BINS * sizeof(float));
  std::memcpy(mEnhIm.data(), mMicIm.data(), N_BINS * sizeof(float));

#ifdef USE_TFLITE
  if (!mCompiledModel)
    return;

  // Push current lpb frame into the DF_ORDER-deep ring (slot = current).
  mLpbRingPos = (mLpbRingPos + 1) % DF_ORDER;
  std::memcpy(mLpbRingRe.data() + mLpbRingPos * N_BINS, mLpbRe.data(),
              N_BINS * sizeof(float));
  std::memcpy(mLpbRingIm.data() + mLpbRingPos * N_BINS, mLpbIm.data(),
              N_BINS * sizeof(float));

  // Inputs are shape-unique: spec (1,4,513) and states (1,512).
  float *spec = inputByElems(4 * N_BINS);
  float *statesIn = inputByElems(2 * GRU_UNITS);
  if (!spec || !statesIn)
    return;
  std::memcpy(spec + 0 * N_BINS, mMicRe.data(), N_BINS * sizeof(float));
  std::memcpy(spec + 1 * N_BINS, mMicIm.data(), N_BINS * sizeof(float));
  std::memcpy(spec + 2 * N_BINS, mLpbRe.data(), N_BINS * sizeof(float));
  std::memcpy(spec + 3 * N_BINS, mLpbIm.data(), N_BINS * sizeof(float));
  std::memcpy(statesIn, mGru1State.data(), GRU_UNITS * sizeof(float));
  std::memcpy(statesIn + GRU_UNITS, mGru2State.data(),
              GRU_UNITS * sizeof(float));

  if (LiteRtRunCompiledModel(mCompiledModel, 0, (uint32_t)mInputBuffers.size(),
                             mInputBuffers.data(),
                             (uint32_t)mOutputBuffers.size(),
                             mOutputBuffers.data()) != kLiteRtStatusOk) {
    return;  // passthrough; state untouched so the stream can recover
  }

  // Outputs: taps (1,1,513,10) = [taps_r(5) || taps_i(5)] per bin,
  // states (1,512) = [gru1 || gru2].
  const float *taps = outputByElems((size_t)N_BINS * 2 * DF_ORDER);
  const float *statesOut = outputByElems(2 * GRU_UNITS);
  if (!taps || !statesOut)
    return;
  std::memcpy(mGru1State.data(), statesOut, GRU_UNITS * sizeof(float));
  std::memcpy(mGru2State.data(), statesOut + GRU_UNITS,
              GRU_UNITS * sizeof(float));

  // bleed[bin] = sum_k taps[bin][k] * lpb[frame-k][bin]  (complex MAC);
  // enhanced = mic - bleed.
  for (int i = 0; i < N_BINS; ++i) {
    const float *t = taps + (size_t)i * 2 * DF_ORDER;
    float br = 0.0f, bi = 0.0f;
    for (int k = 0; k < DF_ORDER; ++k) {
      size_t slot = (mLpbRingPos + DF_ORDER - k) % DF_ORDER;
      float lr = mLpbRingRe[slot * N_BINS + i];
      float li = mLpbRingIm[slot * N_BINS + i];
      float tr = t[k];
      float ti = t[DF_ORDER + k];
      br += tr * lr - ti * li;
      bi += tr * li + ti * lr;
    }
    mEnhRe[i] = mMicRe[i] - br;
    mEnhIm[i] = mMicIm[i] - bi;
  }
#endif
}

void NeuralPostFilter::processSingleStage(const float *micSignal,
                                          const float *refSignal, float *output,
                                          unsigned int frameCount) {
#ifdef USE_TFLITE
  if (!mCompiledModel || mInputBuffers.empty() || mOutputBuffers.empty())
    return;

  // 1. Store current frame's raw magnitudes in ring buffer
  //    Format: [mic_mag(513), lpb_mag(513)] per frame
  size_t frameOffset = mContextWritePos * N_BINS * 2;
  for (int i = 0; i < N_BINS; ++i) {
    mContextBuffer[frameOffset + i] = mMagMic[i];
    mContextBuffer[frameOffset + N_BINS + i] = mMagLpb[i];
  }

  // Advance ring buffer position
  mContextWritePos = (mContextWritePos + 1) % CONTEXT_FRAMES;
  if (mContextFrameCount < CONTEXT_FRAMES) {
    mContextFrameCount++;
  }

  // 2. Skip inference until we have 16 frames of context
  if (mContextFrameCount < CONTEXT_FRAMES) {
    return;  // Output will be unmodified (passthrough via mMagMic)
  }

  // 3. Prepare input tensor: [1, 16, 1026] in temporal order (oldest→newest)
  //    Model has internal log compression: log1p(x*10)/3.0
  void *inputPtr = nullptr;
  if (LiteRtLockTensorBuffer(mInputBuffers[0], &inputPtr,
                             kLiteRtTensorBufferLockModeWrite) ==
      kLiteRtStatusOk) {
    float *inputData = (float *)inputPtr;

    // Copy frames in temporal order: oldest first, newest (current) last
    // mContextWritePos now points to the oldest frame (just wrapped)
    for (int f = 0; f < CONTEXT_FRAMES; ++f) {
      size_t srcFrame = (mContextWritePos + f) % CONTEXT_FRAMES;
      size_t srcOffset = srcFrame * N_BINS * 2;
      size_t dstOffset = f * N_BINS * 2;

      std::memcpy(inputData + dstOffset, mContextBuffer.data() + srcOffset,
                  N_BINS * 2 * sizeof(float));
    }
    LiteRtUnlockTensorBuffer(mInputBuffers[0]);
  }

  // 4. Run inference
  if (LiteRtRunCompiledModel(mCompiledModel, 0, (uint32_t)mInputBuffers.size(),
                             mInputBuffers.data(),
                             (uint32_t)mOutputBuffers.size(),
                             mOutputBuffers.data()) != kLiteRtStatusOk) {
    return;
  }

  // 5. Apply sigmoid mask to current frame magnitude
  void *outputPtr = nullptr;
  if (LiteRtLockTensorBuffer(mOutputBuffers[0], &outputPtr,
                             kLiteRtTensorBufferLockModeRead) ==
      kLiteRtStatusOk) {
    float *mask = (float *)outputPtr;
    for (int i = 0; i < N_BINS; ++i) {
      mMagMic[i] *= mask[i];
    }
    LiteRtUnlockTensorBuffer(mOutputBuffers[0]);
  }
#endif
}
