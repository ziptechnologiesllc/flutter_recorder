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

  std::string fileName;
  if (modelType == NeuralModelType::DTLN_AEC_48K) {
    fileName = "dtln_aec_48k_final.tflite";
  } else if (modelType == NeuralModelType::LSTM_V1) {
    fileName = "aec_lstm_v1.tflite";
  }

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
    // 1. Write new samples to circular input buffers
    mInputBufferMic[mInputIndex] = micSignal[f];
    mInputBufferLpb[mInputIndex] = refSignal[f];
    mInputIndex = (mInputIndex + 1) % N_FFT;

    mWindowPos++;
    if (mWindowPos >= HOP_SIZE) {
      mWindowPos = 0;

      // 2. Prepare FFT input (contiguous copy from circular buffer)
      float *micBlock = mFFTWorkBuffer.data(); // Use part of FFT work buffer or
                                               // other pre-allocated space
      float *refBlock = mFFTWorkBuffer.data() + N_FFT;

      for (int i = 0; i < N_FFT; ++i) {
        int idx = (mInputIndex - N_FFT + i + N_FFT) % N_FFT;
        micBlock[i] = mInputBufferMic[idx];
        refBlock[i] = mInputBufferLpb[idx];
      }

      performSTFT(micBlock, refBlock);
      processSingleStage(nullptr, nullptr, nullptr, 0);

      float *synth = mFFTWorkBuffer.data(); // Reuse buffer
      performIFFT(synth);

      // 3. Overlap-Add into circular output accumulator
      for (int i = 0; i < N_FFT; ++i) {
        int idx = (mOutputIndex + i) % N_FFT;
        mOutputAccumulator[idx] += synth[i];
      }
    }

    // 4. Read from circular output accumulator
    output[f] = mOutputAccumulator[mOutputIndex];
    mOutputAccumulator[mOutputIndex] = 0.0f; // Clear for next round
    mOutputIndex = (mOutputIndex + 1) % N_FFT;
  }
}

void NeuralPostFilter::processSingleStage(const float *micSignal,
                                          const float *refSignal, float *output,
                                          unsigned int frameCount) {
#ifdef USE_TFLITE
  if (!mCompiledModel || mInputBuffers.empty() || mOutputBuffers.empty())
    return;

  void *inputPtr = nullptr;
  if (LiteRtLockTensorBuffer(mInputBuffers[0], &inputPtr,
                             kLiteRtTensorBufferLockModeWrite) ==
      kLiteRtStatusOk) {
    float eps = 1e-6f;
    for (int i = 0; i < N_BINS; ++i) {
      ((float *)inputPtr)[i] = std::log(mMagMic[i] + eps);
      ((float *)inputPtr)[N_BINS + i] = std::log(mMagLpb[i] + eps);
    }
    LiteRtUnlockTensorBuffer(mInputBuffers[0]);
  }

  if (LiteRtRunCompiledModel(mCompiledModel, 0, (uint32_t)mInputBuffers.size(),
                             mInputBuffers.data(),
                             (uint32_t)mOutputBuffers.size(),
                             mOutputBuffers.data()) != kLiteRtStatusOk) {
    return;
  }

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
