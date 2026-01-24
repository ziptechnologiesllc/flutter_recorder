#include "neural_post_filter.h"
extern void aecLog(const char *fmt, ...);
#include <cstring>

#ifdef USE_TFLITE
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
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

  // Compile the model
  if (LiteRtCreateCompiledModel(mEnv, mModel, nullptr, &mCompiledModel) !=
      kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to compile model\n");
    return false;
  }

  // Set up input and output buffers
  // For DTLN-AEC, we expect 2 inputs (mic, ref) and 1 output (clean)
  // We'll create managed tensor buffers from requirements
  LiteRtParamIndex numInputs = 0;
  LiteRtSignature signature;
  if (LiteRtGetModelSignature(mModel, 0, &signature) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to get model signature\n");
    return false;
  }
  if (LiteRtGetNumSignatureInputs(signature, &numInputs) != kLiteRtStatusOk) {
    aecLog("[NeuralPostFilter] Failed to get num signature inputs\n");
    return false;
  }

  for (uint32_t i = 0; i < numInputs; ++i) {
    LiteRtTensorBufferRequirements reqs;
    if (LiteRtGetCompiledModelInputBufferRequirements(
            mCompiledModel, 0, i, &reqs) == kLiteRtStatusOk) {
      LiteRtRankedTensorType type;
      LiteRtTensor tensor;
      LiteRtGetSignatureInputTensorByIndex(signature, i, &tensor);
      LiteRtGetRankedTensorType(tensor, &type);

      LiteRtTensorBuffer buffer;
      if (LiteRtCreateManagedTensorBufferFromRequirements(
              mEnv, &type, reqs, &buffer) == kLiteRtStatusOk) {
        mInputBuffers.push_back(buffer);
      }
    }
  }

  LiteRtTensorBufferRequirements outReqs;
  if (LiteRtGetCompiledModelOutputBufferRequirements(
          mCompiledModel, 0, 0, &outReqs) == kLiteRtStatusOk) {
    LiteRtRankedTensorType type;
    LiteRtTensor tensor;
    LiteRtSignature signature;
    LiteRtGetModelSignature(mModel, 0, &signature);
    LiteRtGetSignatureOutputTensorByIndex(signature, 0, &tensor);
    LiteRtGetRankedTensorType(tensor, &type);

    LiteRtTensorBuffer buffer;
    if (LiteRtCreateManagedTensorBufferFromRequirements(
            mEnv, &type, outReqs, &buffer) == kLiteRtStatusOk) {
      mOutputBuffers.push_back(buffer);
    }
  }

  if (mInputBuffers.empty() || mOutputBuffers.empty()) {
    aecLog("[NeuralPostFilter] Model lacks required I/O tensors\n");
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

  // v3 is the only supported model
  std::string fileName = "aec_mask_v3.tflite";

  std::string fullPath = assetBasePath;
  if (!fullPath.empty() && fullPath.back() != '/')
    fullPath += "/";
  fullPath += fileName;

  return loadModel(fullPath);
}

// STFT Processing Implementation
void NeuralPostFilter::performSTFT(const float *micBlock,
                                   const float *refBlock) {
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
                               float *output, unsigned int frameCount) {
  if (!mEnabled || !mIsLoaded) {
    std::memcpy(output, micSignal, frameCount * sizeof(float));
    return;
  }

  for (unsigned int f = 0; f < frameCount; ++f) {
    std::memmove(mInputBufferMic.data(), mInputBufferMic.data() + 1,
                 (N_FFT - 1) * sizeof(float));
    mInputBufferMic[N_FFT - 1] = micSignal[f];
    std::memmove(mInputBufferLpb.data(), mInputBufferLpb.data() + 1,
                 (N_FFT - 1) * sizeof(float));
    mInputBufferLpb[N_FFT - 1] = refSignal[f];

    mWindowPos++;
    if (mWindowPos >= HOP_SIZE) {
      mWindowPos = 0;
      performSTFT(mInputBufferMic.data(), mInputBufferLpb.data());
      processSingleStage(nullptr, nullptr, nullptr, 0);
      std::vector<float> synth(N_FFT);
      performIFFT(synth.data());
      for (int i = 0; i < N_FFT; ++i)
        mOutputAccumulator[i] += synth[i];
    }
    output[f] = mOutputAccumulator[0];
    std::memmove(mOutputAccumulator.data(), mOutputAccumulator.data() + 1,
                 (N_FFT - 1) * sizeof(float));
    mOutputAccumulator[N_FFT - 1] = 0.0f;
  }
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
