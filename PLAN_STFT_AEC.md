# STFT/ISTFT + Gated DTLN AEC Implementation Plan

## Overview

Add Short-Time Fourier Transform (STFT) and Inverse STFT (ISTFT) capabilities to the flutter_recorder plugin, enabling C++ offloading of signal processing from the DTLN-AEC TensorFlow model. This makes the neural network portion NPU/ANE-friendly by removing custom TF layers.

## Goals

1. Implement STFT returning complex coefficients (magnitude + phase)
2. Implement ISTFT with overlap-add reconstruction
3. Expose both to Dart via FFI
4. Enable simplified LiteRT model for on-device AEC inference

---

## Selected Architecture: Gated DTLN (Option B)

Based on IEEE 9980060 "Nonlinear Residual Echo Suppression Based on Gated DTLN", this hybrid approach provides the best quality while enabling partial NPU acceleration for the GLU components.

**Architecture Details:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gated DTLN Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Frequency Domain (Mask Estimation)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Magnitude → LSTM(512) → LSTM(512) → Dense → Sigmoid → Mask │ │
│  │                ↑            ↑                               │ │
│  │            CPU/GPU      CPU/GPU                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Stage 2: Time Domain (Residual Enhancement with GLU)            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Frames → Conv1D → GLU → LSTM(512) → GLU → Conv1D → Output  │ │
│  │             ↑       ↑       ↑         ↑       ↑            │ │
│  │           NPU     NPU    CPU/GPU    NPU     NPU            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**NPU Acceleration Breakdown:**
| Layer Type | NPU Compatible | Notes |
|------------|----------------|-------|
| Conv1D | Yes | Fully accelerated |
| Dense | Yes | Fully accelerated |
| LayerNorm | Yes | Most NPUs support |
| Sigmoid | Yes | Standard activation |
| LSTM | No | Falls back to CPU |
| Split | Yes | Simple tensor op |
| Multiply | Yes | Element-wise |

**Expected Performance Split:**
- ~40-50% of computation in GLU/Conv layers → NPU
- ~50-60% in LSTM layers → CPU/GPU
- Still significant speedup vs pure CPU

---

## Current Codebase Reference

**Existing FFT primitives** (`src/fft/soloud_fft.h`):
```cpp
void fft(float *aBuffer, unsigned int aBufferLength);   // Forward FFT (power of 2)
void ifft(float *aBuffer, unsigned int aBufferLength);  // Inverse FFT (power of 2)
void fft1024(float *aBuffer);                           // Optimized 1024-point
void ifft256(float *aBuffer);                           // Optimized 256-point
```

**Existing windowing** (`src/analyzer.cpp`):
```cpp
// Blackman window coefficients (lines 95-105)
float alpha = 0.16f;
float a0 = 0.5f * (1.0f - alpha);
float a1 = 0.5f;
float a2 = 0.5f * alpha;
// window[n] = a0 - a1*cos(2πn/N) + a2*cos(4πn/N)
```

**Target parameters** (from DTLN-AEC model):
- Sample rate: 48000 Hz
- FFT size (blockLen): 2048
- Hop size (block_shift): 480 (10ms)
- Overlap: ~77%

---

## Implementation Tasks

### Phase 1: C++ STFT/ISTFT Core

#### Task 1.1: Create STFT header file

**File:** `src/stft/stft.h`

```cpp
#ifndef FLUTTER_RECORDER_STFT_H
#define FLUTTER_RECORDER_STFT_H

#include <cstdint>

namespace STFT {

/**
 * Configuration for STFT/ISTFT operations
 */
struct StftConfig {
    int fftSize;        // FFT window size (e.g., 2048)
    int hopSize;        // Hop between frames (e.g., 480)
    int sampleRate;     // Audio sample rate (e.g., 48000)
};

/**
 * Perform Short-Time Fourier Transform
 *
 * @param input       Input audio samples (float32)
 * @param inputLen    Number of input samples
 * @param outputReal  Output real components [numFrames × (fftSize/2+1)]
 * @param outputImag  Output imaginary components [numFrames × (fftSize/2+1)]
 * @param config      STFT configuration
 * @return            Number of frames produced
 */
int stft(const float *input, int inputLen,
         float *outputReal, float *outputImag,
         const StftConfig &config);

/**
 * Perform Inverse Short-Time Fourier Transform with overlap-add
 *
 * @param inputReal   Input real components [numFrames × (fftSize/2+1)]
 * @param inputImag   Input imaginary components [numFrames × (fftSize/2+1)]
 * @param numFrames   Number of STFT frames
 * @param output      Output audio samples (float32)
 * @param outputLen   Expected output length (for padding/cropping)
 * @param config      STFT configuration
 */
void istft(const float *inputReal, const float *inputImag, int numFrames,
           float *output, int outputLen,
           const StftConfig &config);

/**
 * Calculate number of frames for given input length
 */
int calcNumFrames(int inputLen, const StftConfig &config);

/**
 * Calculate output length for given number of frames
 */
int calcOutputLen(int numFrames, const StftConfig &config);

/**
 * Compute magnitude and phase from complex STFT
 */
void complexToMagPhase(const float *real, const float *imag,
                       float *magnitude, float *phase, int len);

/**
 * Compute complex from magnitude and phase
 */
void magPhaseToComplex(const float *magnitude, const float *phase,
                       float *real, float *imag, int len);

} // namespace STFT

#endif
```

#### Task 1.2: Implement STFT/ISTFT

**File:** `src/stft/stft.cpp`

```cpp
#include "stft.h"
#include "../fft/soloud_fft.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace STFT {

// Pre-computed Blackman window (call once per config)
static std::vector<float> computeBlackmanWindow(int size) {
    std::vector<float> window(size);
    const float alpha = 0.16f;
    const float a0 = 0.5f * (1.0f - alpha);
    const float a1 = 0.5f;
    const float a2 = 0.5f * alpha;
    const float pi = 3.14159265358979323846f;

    for (int i = 0; i < size; i++) {
        float t = static_cast<float>(i) / static_cast<float>(size - 1);
        window[i] = a0 - a1 * cosf(2.0f * pi * t) + a2 * cosf(4.0f * pi * t);
    }
    return window;
}

int calcNumFrames(int inputLen, const StftConfig &config) {
    if (inputLen < config.fftSize) return 0;
    return 1 + (inputLen - config.fftSize) / config.hopSize;
}

int calcOutputLen(int numFrames, const StftConfig &config) {
    return config.fftSize + (numFrames - 1) * config.hopSize;
}

int stft(const float *input, int inputLen,
         float *outputReal, float *outputImag,
         const StftConfig &config) {

    const int fftSize = config.fftSize;
    const int hopSize = config.hopSize;
    const int numBins = fftSize / 2 + 1;
    const int numFrames = calcNumFrames(inputLen, config);

    if (numFrames <= 0) return 0;

    // Pre-compute window
    std::vector<float> window = computeBlackmanWindow(fftSize);

    // FFT buffer (interleaved real/imag for SoLoud FFT)
    std::vector<float> fftBuffer(fftSize);

    for (int frame = 0; frame < numFrames; frame++) {
        int offset = frame * hopSize;

        // Apply window to input frame
        for (int i = 0; i < fftSize; i++) {
            fftBuffer[i] = input[offset + i] * window[i];
        }

        // Perform FFT (in-place, result is interleaved real/imag)
        FFT::fft(fftBuffer.data(), fftSize);

        // Extract real and imaginary components
        // SoLoud FFT output format: [re0, im0, re1, im1, ...]
        // But for real input, we only need bins 0 to fftSize/2
        int outOffset = frame * numBins;
        for (int bin = 0; bin < numBins; bin++) {
            // Note: Verify SoLoud FFT output format and adjust indexing
            outputReal[outOffset + bin] = fftBuffer[bin * 2];
            outputImag[outOffset + bin] = fftBuffer[bin * 2 + 1];
        }
    }

    return numFrames;
}

void istft(const float *inputReal, const float *inputImag, int numFrames,
           float *output, int outputLen,
           const StftConfig &config) {

    const int fftSize = config.fftSize;
    const int hopSize = config.hopSize;
    const int numBins = fftSize / 2 + 1;

    // Pre-compute window for synthesis
    std::vector<float> window = computeBlackmanWindow(fftSize);

    // Overlap-add buffer
    int olaLen = calcOutputLen(numFrames, config);
    std::vector<float> olaBuffer(olaLen, 0.0f);
    std::vector<float> windowSum(olaLen, 0.0f);

    // IFFT buffer
    std::vector<float> fftBuffer(fftSize);

    for (int frame = 0; frame < numFrames; frame++) {
        int inOffset = frame * numBins;

        // Reconstruct full spectrum (conjugate symmetry for real signal)
        for (int bin = 0; bin < numBins; bin++) {
            fftBuffer[bin * 2] = inputReal[inOffset + bin];
            fftBuffer[bin * 2 + 1] = inputImag[inOffset + bin];
        }
        // Mirror for negative frequencies (if needed by SoLoud IFFT)
        // Verify SoLoud IFFT expectations

        // Perform IFFT
        FFT::ifft(fftBuffer.data(), fftSize);

        // Apply synthesis window and overlap-add
        int outOffset = frame * hopSize;
        for (int i = 0; i < fftSize; i++) {
            olaBuffer[outOffset + i] += fftBuffer[i] * window[i];
            windowSum[outOffset + i] += window[i] * window[i];
        }
    }

    // Normalize by window sum (COLA condition)
    for (int i = 0; i < olaLen; i++) {
        if (windowSum[i] > 1e-8f) {
            olaBuffer[i] /= windowSum[i];
        }
    }

    // Copy to output with padding/cropping
    int copyLen = std::min(olaLen, outputLen);
    std::memcpy(output, olaBuffer.data(), copyLen * sizeof(float));

    // Zero-pad if output is longer
    if (outputLen > olaLen) {
        std::memset(output + olaLen, 0, (outputLen - olaLen) * sizeof(float));
    }
}

void complexToMagPhase(const float *real, const float *imag,
                       float *magnitude, float *phase, int len) {
    for (int i = 0; i < len; i++) {
        magnitude[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        phase[i] = atan2f(imag[i], real[i]);
    }
}

void magPhaseToComplex(const float *magnitude, const float *phase,
                       float *real, float *imag, int len) {
    for (int i = 0; i < len; i++) {
        real[i] = magnitude[i] * cosf(phase[i]);
        imag[i] = magnitude[i] * sinf(phase[i]);
    }
}

} // namespace STFT
```

#### Task 1.3: Verify SoLoud FFT format

Before finalizing implementation, verify the exact input/output format of `FFT::fft()` and `FFT::ifft()`:

1. Check `src/fft/soloud_fft.cpp` for buffer layout
2. Test with known signal (e.g., single sine wave)
3. Document whether output is interleaved `[re, im, re, im, ...]` or split

---

### Phase 2: FFI Export Layer

#### Task 2.1: Add C exports to flutter_recorder.h

**File:** `src/flutter_recorder.h` (add to existing)

```cpp
// STFT/ISTFT exports for AEC processing
FFI_PLUGIN_EXPORT int flutter_recorder_stft_calcNumFrames(
    int inputLen, int fftSize, int hopSize);

FFI_PLUGIN_EXPORT int flutter_recorder_stft(
    const float *input, int inputLen,
    float *outputReal, float *outputImag,
    int fftSize, int hopSize);

FFI_PLUGIN_EXPORT void flutter_recorder_istft(
    const float *inputReal, const float *inputImag, int numFrames,
    float *output, int outputLen,
    int fftSize, int hopSize);

FFI_PLUGIN_EXPORT void flutter_recorder_complexToMagPhase(
    const float *real, const float *imag,
    float *magnitude, float *phase, int len);

FFI_PLUGIN_EXPORT void flutter_recorder_magPhaseToComplex(
    const float *magnitude, const float *phase,
    float *real, float *imag, int len);
```

#### Task 2.2: Implement C exports

**File:** `src/flutter_recorder.cpp` (add to existing)

```cpp
#include "stft/stft.h"

FFI_PLUGIN_EXPORT int flutter_recorder_stft_calcNumFrames(
    int inputLen, int fftSize, int hopSize) {
    STFT::StftConfig config = {fftSize, hopSize, 48000};
    return STFT::calcNumFrames(inputLen, config);
}

FFI_PLUGIN_EXPORT int flutter_recorder_stft(
    const float *input, int inputLen,
    float *outputReal, float *outputImag,
    int fftSize, int hopSize) {
    STFT::StftConfig config = {fftSize, hopSize, 48000};
    return STFT::stft(input, inputLen, outputReal, outputImag, config);
}

FFI_PLUGIN_EXPORT void flutter_recorder_istft(
    const float *inputReal, const float *inputImag, int numFrames,
    float *output, int outputLen,
    int fftSize, int hopSize) {
    STFT::StftConfig config = {fftSize, hopSize, 48000};
    STFT::istft(inputReal, inputImag, numFrames, output, outputLen, config);
}

FFI_PLUGIN_EXPORT void flutter_recorder_complexToMagPhase(
    const float *real, const float *imag,
    float *magnitude, float *phase, int len) {
    STFT::complexToMagPhase(real, imag, magnitude, phase, len);
}

FFI_PLUGIN_EXPORT void flutter_recorder_magPhaseToComplex(
    const float *magnitude, const float *phase,
    float *real, float *imag, int len) {
    STFT::magPhaseToComplex(magnitude, phase, real, imag, len);
}
```

#### Task 2.3: Update CMakeLists.txt

Add new source files to the build:

```cmake
add_library(flutter_recorder SHARED
    # ... existing sources ...
    "src/stft/stft.cpp"
)
```

---

### Phase 3: Dart FFI Bindings

#### Task 3.1: Generate FFI bindings

Run `ffigen` or manually add to `lib/src/bindings/flutter_recorder_bindings_generated.dart`:

```dart
int flutter_recorder_stft_calcNumFrames(
    int inputLen, int fftSize, int hopSize);

int flutter_recorder_stft(
    ffi.Pointer<ffi.Float> input, int inputLen,
    ffi.Pointer<ffi.Float> outputReal, ffi.Pointer<ffi.Float> outputImag,
    int fftSize, int hopSize);

void flutter_recorder_istft(
    ffi.Pointer<ffi.Float> inputReal, ffi.Pointer<ffi.Float> inputImag,
    int numFrames,
    ffi.Pointer<ffi.Float> output, int outputLen,
    int fftSize, int hopSize);

void flutter_recorder_complexToMagPhase(
    ffi.Pointer<ffi.Float> real, ffi.Pointer<ffi.Float> imag,
    ffi.Pointer<ffi.Float> magnitude, ffi.Pointer<ffi.Float> phase,
    int len);

void flutter_recorder_magPhaseToComplex(
    ffi.Pointer<ffi.Float> magnitude, ffi.Pointer<ffi.Float> phase,
    ffi.Pointer<ffi.Float> real, ffi.Pointer<ffi.Float> imag,
    int len);
```

#### Task 3.2: Create high-level Dart API

**File:** `lib/src/stft.dart`

```dart
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'bindings/flutter_recorder_bindings_generated.dart';

class StftResult {
  final Float32List real;
  final Float32List imag;
  final int numFrames;
  final int numBins;

  StftResult(this.real, this.imag, this.numFrames, this.numBins);

  /// Convert to magnitude and phase
  (Float32List magnitude, Float32List phase) toMagPhase() {
    // Implementation using flutter_recorder_complexToMagPhase
  }
}

class StftProcessor {
  final FlutterRecorderBindings _bindings;
  final int fftSize;
  final int hopSize;

  StftProcessor(this._bindings, {
    this.fftSize = 2048,
    this.hopSize = 480,
  });

  int get numBins => fftSize ~/ 2 + 1;

  int calcNumFrames(int inputLen) {
    return _bindings.flutter_recorder_stft_calcNumFrames(
        inputLen, fftSize, hopSize);
  }

  /// Perform STFT on input audio
  StftResult stft(Float32List input) {
    final numFrames = calcNumFrames(input.length);
    final outputSize = numFrames * numBins;

    final inputPtr = calloc<ffi.Float>(input.length);
    final realPtr = calloc<ffi.Float>(outputSize);
    final imagPtr = calloc<ffi.Float>(outputSize);

    try {
      // Copy input
      inputPtr.asTypedList(input.length).setAll(0, input);

      // Perform STFT
      _bindings.flutter_recorder_stft(
          inputPtr, input.length, realPtr, imagPtr, fftSize, hopSize);

      // Copy output
      final real = Float32List.fromList(realPtr.asTypedList(outputSize));
      final imag = Float32List.fromList(imagPtr.asTypedList(outputSize));

      return StftResult(real, imag, numFrames, numBins);
    } finally {
      calloc.free(inputPtr);
      calloc.free(realPtr);
      calloc.free(imagPtr);
    }
  }

  /// Perform ISTFT to reconstruct audio
  Float32List istft(Float32List real, Float32List imag,
                    int numFrames, int outputLen) {
    final realPtr = calloc<ffi.Float>(real.length);
    final imagPtr = calloc<ffi.Float>(imag.length);
    final outputPtr = calloc<ffi.Float>(outputLen);

    try {
      realPtr.asTypedList(real.length).setAll(0, real);
      imagPtr.asTypedList(imag.length).setAll(0, imag);

      _bindings.flutter_recorder_istft(
          realPtr, imagPtr, numFrames, outputPtr, outputLen, fftSize, hopSize);

      return Float32List.fromList(outputPtr.asTypedList(outputLen));
    } finally {
      calloc.free(realPtr);
      calloc.free(imagPtr);
      calloc.free(outputPtr);
    }
  }
}
```

---

### Phase 4: Gated DTLN Model + LiteRT Export

#### Task 4.0: Implement Gated DTLN architecture

Modify `train_aec_colab_hq.ipynb` to add GLU layers:

```python
class GatedLinearUnit(tf.keras.layers.Layer):
    """Gated Linear Unit for enhanced temporal modeling."""

    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(
            self.units * 2, self.kernel_size,
            padding='same', use_bias=True
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        conv_out = self.norm(self.conv(x))
        signal, gate = tf.split(conv_out, 2, axis=-1)
        return signal * tf.sigmoid(gate)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'kernel_size': self.kernel_size})
        return config


class GatedDTLN_AEC(DTLN_AEC_LSTM):
    """Gated DTLN with GLU enhancement in Stage 2."""

    def build_model(self):
        # Inputs (same as parent)
        time_mic = tf.keras.Input(batch_shape=[1, None], name='time_mic')
        time_lpb = tf.keras.Input(batch_shape=[1, None], name='time_lpb')

        # === STAGE 1: Frequency Domain (unchanged from DTLN) ===
        # STFT
        mag_mic, angle_mic = STFTLayer(self.blockLen)(time_mic)
        mag_lpb, _ = STFTLayer(self.blockLen)(time_lpb)

        # Concatenate mic + lpb magnitudes
        mag_concat = tf.keras.layers.Concatenate(axis=-1)([mag_mic, mag_lpb])

        # LSTM stack for mask estimation
        lstm_out = mag_concat
        for i in range(self.numLayer // 2):
            lstm_out = tf.keras.layers.LSTM(
                self.numUnits, return_sequences=True,
                stateful=True, name=f'stage1_lstm_{i}'
            )(lstm_out)

        # Frequency mask
        mask_1 = tf.keras.layers.Dense(self.encoder_size, activation='sigmoid')(lstm_out)
        estimated_mag = tf.keras.layers.Multiply()([mag_mic, mask_1])

        # IFFT back to time domain
        estimated_sig_1 = IFFTLayer()([estimated_mag, angle_mic])

        # === STAGE 2: Time Domain with GLU Enhancement ===
        # Frame the signals
        estimated_frames = FrameLayer(self.blockLen, self.block_shift)(estimated_sig_1)
        lpb_frames = FrameLayer(self.blockLen, self.block_shift)(time_lpb)

        # Concatenate framed signals
        concat_frames = tf.keras.layers.Concatenate(axis=-1)([estimated_frames, lpb_frames])

        # Initial Conv1D encoding
        encoded = tf.keras.layers.Conv1D(self.encoder_size, 1, use_bias=False)(concat_frames)

        # PRE-LSTM GLU block (NPU-accelerated)
        glu_pre = GatedLinearUnit(self.encoder_size, kernel_size=3, name='glu_pre')(encoded)

        # LSTM for temporal modeling (CPU/GPU)
        lstm_out_2 = glu_pre
        for i in range(self.numLayer // 2):
            lstm_out_2 = tf.keras.layers.LSTM(
                self.numUnits, return_sequences=True,
                stateful=True, name=f'stage2_lstm_{i}'
            )(lstm_out_2)

        # POST-LSTM GLU block (NPU-accelerated)
        glu_post = GatedLinearUnit(self.encoder_size, kernel_size=3, name='glu_post')(lstm_out_2)

        # Output Conv1D
        decoded = tf.keras.layers.Conv1D(self.blockLen, 1, use_bias=False)(glu_post)

        # Overlap-add reconstruction
        estimated_sig = OverlapAddLayer(self.block_shift)(decoded)
        estimated_sig = MatchLengthLayer(name='match_length')([estimated_sig, time_mic])

        self.model = tf.keras.Model(
            inputs=[time_mic, time_lpb],
            outputs=estimated_sig,
            name='gated_dtln_aec'
        )
```

#### Task 4.1: Export model without signal processing layers

Modify `train_aec_colab_hq.ipynb` to export a stripped model for LiteRT:

```python
def export_inference_model(self, path):
    """Export model without STFT/ISTFT layers for C++ preprocessing."""

    # Input: STFT magnitude [batch, frames, bins]
    mag_input = tf.keras.Input(shape=(None, self.encoder_size), name='magnitude')
    lpb_input = tf.keras.Input(shape=(None, self.blockLen), name='lpb_frames')

    # Stage 1: Frequency domain masking (reuse trained weights)
    # ... extract relevant layers ...

    # Stage 2: Time domain refinement
    # ... extract relevant layers ...

    # Output: Mask [batch, frames, bins]
    lite_model = tf.keras.Model(
        inputs=[mag_input, lpb_input],
        outputs=mask_output,
        name='dtln_aec_lite'
    )

    # Convert to LiteRT format
    converter = tf.lite.TFLiteConverter.from_keras_model(lite_model)

    # Enable NPU/GPU delegate compatibility
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard LiteRT ops
    ]

    # Quantization for NPU acceleration (choose one):
    # Option A: Float16 (GPU delegates, some NPUs)
    converter.target_spec.supported_types = [tf.float16]

    # Option B: Int8 full quantization (most NPU delegates)
    # converter.representative_dataset = representative_dataset_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8

    litert_model = converter.convert()

    # Save as .tflite (LiteRT CompiledModel format)
    with open(path, 'wb') as f:
        f.write(litert_model)

    print(f"Exported LiteRT model to {path}")
    print(f"Model size: {len(litert_model) / 1024 / 1024:.2f} MB")
```

#### Task 4.2: C++ LiteRT CompiledModel wrapper

**File:** `src/aec/litert_aec.h`

```cpp
#ifndef LITERT_AEC_H
#define LITERT_AEC_H

#include <litert/litert.h>
#include <memory>
#include <vector>

class LiteRTAecModel {
public:
    LiteRTAecModel();
    ~LiteRTAecModel();

    // Load model with hardware acceleration
    // accelerator: kLiteRtHwAcceleratorCpu, kLiteRtHwAcceleratorGpu, or kLiteRtHwAcceleratorNpu
    bool load(const char* modelPath, LiteRtHwAccelerator accelerator);

    // Run inference
    // magnitude: [numFrames × numBins] STFT magnitude
    // lpbFrames: [numFrames × frameLen] framed loopback audio
    // outputMask: [numFrames × numBins] output mask (pre-allocated)
    bool runInference(const float* magnitude, const float* lpbFrames,
                      int numFrames, int numBins, int frameLen,
                      float* outputMask);

private:
    std::unique_ptr<litert::Environment> env_;
    std::unique_ptr<litert::Model> model_;
    std::unique_ptr<litert::CompiledModel> compiledModel_;

    std::vector<litert::TensorBuffer> inputBuffers_;
    std::vector<litert::TensorBuffer> outputBuffers_;
};

#endif
```

#### Task 4.3: C++ LiteRT CompiledModel implementation

**File:** `src/aec/litert_aec.cpp`

```cpp
#include "litert_aec.h"
#include <litert/litert_macros.h>

LiteRTAecModel::LiteRTAecModel() = default;
LiteRTAecModel::~LiteRTAecModel() = default;

bool LiteRTAecModel::load(const char* modelPath, LiteRtHwAccelerator accelerator) {
    // Create environment
    auto env_result = litert::Environment::Create({});
    if (!env_result.ok()) return false;
    env_ = std::make_unique<litert::Environment>(std::move(*env_result));

    // Load model from file
    auto model_result = litert::Model::CreateFromFile(modelPath);
    if (!model_result.ok()) return false;
    model_ = std::make_unique<litert::Model>(std::move(*model_result));

    // Create compiled model with hardware acceleration
    auto compiled_result = litert::CompiledModel::Create(*env_, *model_, accelerator);
    if (!compiled_result.ok()) {
        // Fallback to CPU if requested accelerator unavailable
        compiled_result = litert::CompiledModel::Create(*env_, *model_, kLiteRtHwAcceleratorCpu);
        if (!compiled_result.ok()) return false;
    }
    compiledModel_ = std::make_unique<litert::CompiledModel>(std::move(*compiled_result));

    return true;
}

bool LiteRTAecModel::runInference(const float* magnitude, const float* lpbFrames,
                                   int numFrames, int numBins, int frameLen,
                                   float* outputMask) {
    // Get buffer requirements
    auto mag_req = compiledModel_->GetInputBufferRequirements(0, 0);
    auto lpb_req = compiledModel_->GetInputBufferRequirements(0, 1);
    auto out_req = compiledModel_->GetOutputBufferRequirements(0, 0);

    if (!mag_req.ok() || !lpb_req.ok() || !out_req.ok()) return false;

    // Create input buffers (zero-copy from existing memory)
    auto mag_buffer = litert::TensorBuffer::CreateFromHostMemory(
        *mag_req, const_cast<float*>(magnitude), numFrames * numBins * sizeof(float));
    auto lpb_buffer = litert::TensorBuffer::CreateFromHostMemory(
        *lpb_req, const_cast<float*>(lpbFrames), numFrames * frameLen * sizeof(float));

    if (!mag_buffer.ok() || !lpb_buffer.ok()) return false;

    // Create output buffer
    auto out_buffer = litert::TensorBuffer::CreateFromHostMemory(
        *out_req, outputMask, numFrames * numBins * sizeof(float));

    if (!out_buffer.ok()) return false;

    // Run inference
    std::vector<litert::TensorBuffer> inputs = {std::move(*mag_buffer), std::move(*lpb_buffer)};
    std::vector<litert::TensorBuffer> outputs = {std::move(*out_buffer)};

    auto run_result = compiledModel_->Run(inputs, outputs);
    return run_result.ok();
}
```

#### Task 4.4: FFI export for LiteRT

**File:** `src/flutter_recorder.h` (add to existing)

```cpp
// LiteRT AEC model exports
FFI_PLUGIN_EXPORT void* flutter_recorder_aec_create();
FFI_PLUGIN_EXPORT void flutter_recorder_aec_destroy(void* model);
FFI_PLUGIN_EXPORT bool flutter_recorder_aec_load(
    void* model, const char* path, int accelerator);
FFI_PLUGIN_EXPORT bool flutter_recorder_aec_run(
    void* model,
    const float* magnitude, const float* lpbFrames,
    int numFrames, int numBins, int frameLen,
    float* outputMask);
```

**File:** `src/flutter_recorder.cpp` (add to existing)

```cpp
#include "aec/litert_aec.h"

FFI_PLUGIN_EXPORT void* flutter_recorder_aec_create() {
    return new LiteRTAecModel();
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_destroy(void* model) {
    delete static_cast<LiteRTAecModel*>(model);
}

FFI_PLUGIN_EXPORT bool flutter_recorder_aec_load(
    void* model, const char* path, int accelerator) {
    auto* aec = static_cast<LiteRTAecModel*>(model);
    return aec->load(path, static_cast<LiteRtHwAccelerator>(accelerator));
}

FFI_PLUGIN_EXPORT bool flutter_recorder_aec_run(
    void* model,
    const float* magnitude, const float* lpbFrames,
    int numFrames, int numBins, int frameLen,
    float* outputMask) {
    auto* aec = static_cast<LiteRTAecModel*>(model);
    return aec->runInference(magnitude, lpbFrames, numFrames, numBins, frameLen, outputMask);
}
```

#### Task 4.5: Model I/O specification

**Inputs (from C++ STFT):**
- `magnitude`: `[1, num_frames, 1025]` - STFT magnitude (fftSize/2+1 bins)
- `lpb_frames`: `[1, num_frames, 2048]` - Framed loopback audio

**Outputs (to C++ ISTFT):**
- `mask`: `[1, num_frames, 1025]` - Frequency mask to apply

**C++ applies:** `enhanced_mag = mask * magnitude`
**C++ reconstructs:** `enhanced_audio = ISTFT(enhanced_mag, original_phase)`

---

### Phase 5: Integration Pipeline (C++)

#### Task 5.1: Full AEC processor in C++

**File:** `src/aec/aec_processor.h`

```cpp
#ifndef AEC_PROCESSOR_H
#define AEC_PROCESSOR_H

#include "../stft/stft.h"
#include "litert_aec.h"
#include <vector>

class AecProcessor {
public:
    AecProcessor(int fftSize = 2048, int hopSize = 480, int sampleRate = 48000);
    ~AecProcessor();

    // Load the LiteRT model
    bool loadModel(const char* modelPath, LiteRtHwAccelerator accelerator);

    // Process audio: mic input with echo, lpb (loopback/reference), output enhanced
    // All buffers are float32, same length
    bool process(const float* micAudio, const float* lpbAudio,
                 int numSamples, float* outputAudio);

private:
    STFT::StftConfig config_;
    LiteRTAecModel model_;

    // Pre-allocated buffers
    std::vector<float> stftReal_;
    std::vector<float> stftImag_;
    std::vector<float> magnitude_;
    std::vector<float> phase_;
    std::vector<float> lpbFrames_;
    std::vector<float> mask_;
    std::vector<float> enhancedReal_;
    std::vector<float> enhancedImag_;

    int numBins_;
};

#endif
```

#### Task 5.2: Full AEC processor implementation

**File:** `src/aec/aec_processor.cpp`

```cpp
#include "aec_processor.h"
#include <cstring>

AecProcessor::AecProcessor(int fftSize, int hopSize, int sampleRate)
    : config_{fftSize, hopSize, sampleRate}
    , numBins_(fftSize / 2 + 1) {
}

AecProcessor::~AecProcessor() = default;

bool AecProcessor::loadModel(const char* modelPath, LiteRtHwAccelerator accelerator) {
    return model_.load(modelPath, accelerator);
}

bool AecProcessor::process(const float* micAudio, const float* lpbAudio,
                           int numSamples, float* outputAudio) {
    // Calculate frame count
    int numFrames = STFT::calcNumFrames(numSamples, config_);
    if (numFrames <= 0) return false;

    // Resize buffers
    int specSize = numFrames * numBins_;
    int frameSize = numFrames * config_.fftSize;

    stftReal_.resize(specSize);
    stftImag_.resize(specSize);
    magnitude_.resize(specSize);
    phase_.resize(specSize);
    lpbFrames_.resize(frameSize);
    mask_.resize(specSize);
    enhancedReal_.resize(specSize);
    enhancedImag_.resize(specSize);

    // 1. STFT on mic audio
    STFT::stft(micAudio, numSamples, stftReal_.data(), stftImag_.data(), config_);

    // 2. Convert to magnitude and phase
    STFT::complexToMagPhase(stftReal_.data(), stftImag_.data(),
                            magnitude_.data(), phase_.data(), specSize);

    // 3. Frame the LPB audio (simple framing without FFT)
    for (int f = 0; f < numFrames; f++) {
        int inOffset = f * config_.hopSize;
        int outOffset = f * config_.fftSize;
        for (int i = 0; i < config_.fftSize; i++) {
            int idx = inOffset + i;
            lpbFrames_[outOffset + i] = (idx < numSamples) ? lpbAudio[idx] : 0.0f;
        }
    }

    // 4. Run LiteRT model (NPU/GPU accelerated)
    if (!model_.runInference(magnitude_.data(), lpbFrames_.data(),
                             numFrames, numBins_, config_.fftSize,
                             mask_.data())) {
        return false;
    }

    // 5. Apply mask to magnitude
    for (int i = 0; i < specSize; i++) {
        magnitude_[i] *= mask_[i];
    }

    // 6. Convert back to complex
    STFT::magPhaseToComplex(magnitude_.data(), phase_.data(),
                            enhancedReal_.data(), enhancedImag_.data(), specSize);

    // 7. ISTFT to reconstruct audio
    STFT::istft(enhancedReal_.data(), enhancedImag_.data(), numFrames,
                outputAudio, numSamples, config_);

    return true;
}
```

#### Task 5.3: FFI export for full pipeline

**File:** `src/flutter_recorder.h` (add to existing)

```cpp
// Full AEC pipeline exports
FFI_PLUGIN_EXPORT void* flutter_recorder_aec_processor_create(
    int fftSize, int hopSize, int sampleRate);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_processor_destroy(void* processor);
FFI_PLUGIN_EXPORT bool flutter_recorder_aec_processor_load(
    void* processor, const char* modelPath, int accelerator);
FFI_PLUGIN_EXPORT bool flutter_recorder_aec_processor_process(
    void* processor,
    const float* micAudio, const float* lpbAudio,
    int numSamples, float* outputAudio);
```

#### Task 5.4: Dart high-level API

**File:** `lib/src/aec_processor.dart`

```dart
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

enum HwAccelerator { cpu, gpu, npu }

class AecProcessor {
  final ffi.Pointer<ffi.Void> _handle;
  final FlutterRecorderBindings _bindings;
  final int fftSize;
  final int hopSize;

  AecProcessor._(this._handle, this._bindings, this.fftSize, this.hopSize);

  factory AecProcessor.create(FlutterRecorderBindings bindings, {
    int fftSize = 2048,
    int hopSize = 480,
    int sampleRate = 48000,
  }) {
    final handle = bindings.flutter_recorder_aec_processor_create(
        fftSize, hopSize, sampleRate);
    return AecProcessor._(handle, bindings, fftSize, hopSize);
  }

  bool loadModel(String modelPath, {HwAccelerator accelerator = HwAccelerator.gpu}) {
    final pathPtr = modelPath.toNativeUtf8();
    try {
      return _bindings.flutter_recorder_aec_processor_load(
          _handle, pathPtr.cast(), accelerator.index);
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// Process mic audio with echo cancellation
  /// Returns enhanced audio or null on failure
  Float32List? process(Float32List micAudio, Float32List lpbAudio) {
    if (micAudio.length != lpbAudio.length) return null;

    final numSamples = micAudio.length;
    final micPtr = calloc<ffi.Float>(numSamples);
    final lpbPtr = calloc<ffi.Float>(numSamples);
    final outPtr = calloc<ffi.Float>(numSamples);

    try {
      micPtr.asTypedList(numSamples).setAll(0, micAudio);
      lpbPtr.asTypedList(numSamples).setAll(0, lpbAudio);

      final success = _bindings.flutter_recorder_aec_processor_process(
          _handle, micPtr, lpbPtr, numSamples, outPtr);

      if (!success) return null;

      return Float32List.fromList(outPtr.asTypedList(numSamples));
    } finally {
      calloc.free(micPtr);
      calloc.free(lpbPtr);
      calloc.free(outPtr);
    }
  }

  void dispose() {
    _bindings.flutter_recorder_aec_processor_destroy(_handle);
  }
}
```

---

## Testing Plan

### Unit Tests

1. **STFT round-trip test**: `ISTFT(STFT(x)) ≈ x` (within tolerance)
2. **Known signal test**: STFT of sine wave should show peak at expected bin
3. **Frame count test**: Verify `calcNumFrames` matches Python `librosa.stft`
4. **Magnitude/phase test**: `magPhaseToComplex(complexToMagPhase(z)) = z`

### Integration Tests

1. Compare C++ STFT output with Python `tf.signal.stft` on same input
2. Verify TFLite model produces same output as full Keras model
3. End-to-end AEC test with known echo signal

---

## File Summary

| File | Action |
|------|--------|
| `src/stft/stft.h` | Create |
| `src/stft/stft.cpp` | Create |
| `src/aec/litert_aec.h` | Create |
| `src/aec/litert_aec.cpp` | Create |
| `src/aec/aec_processor.h` | Create |
| `src/aec/aec_processor.cpp` | Create |
| `src/flutter_recorder.h` | Modify (add STFT + AEC exports) |
| `src/flutter_recorder.cpp` | Modify (add STFT + AEC implementations) |
| `CMakeLists.txt` | Modify (add sources, link LiteRT) |
| `lib/src/bindings/flutter_recorder_bindings_generated.dart` | Regenerate |
| `lib/src/stft.dart` | Create |
| `lib/src/aec_processor.dart` | Create |
| `train_aec_colab_hq.ipynb` | Modify (add Gated DTLN + lite model export) |

---

## Dependencies

**Existing (in flutter_recorder):**
- SoLoud FFT (`src/fft/soloud_fft.h`)
- Analyzer windowing reference (`src/analyzer.cpp`)

**New C++ dependencies:**
- LiteRT SDK headers and libraries
  - `litert/litert.h`
  - `litert/litert_macros.h`
  - `libLiteRtRuntime.so` (or platform equivalent)
  - GPU accelerator: `libLiteRtGpuAccelerator.so`
  - NPU accelerator: vendor-specific (QNN, Neuron, etc.)

**New Dart dependencies:**
- `ffi: ^2.0.0` (for STFT/AEC FFI calls)

## LiteRT Hardware Accelerator Support

| Accelerator Constant | Platform | Hardware |
|---------------------|----------|----------|
| `kLiteRtHwAcceleratorCpu` | All | CPU (fallback) |
| `kLiteRtHwAcceleratorGpu` | Android/Linux | OpenCL/OpenGL GPU |
| `kLiteRtHwAcceleratorNpu` | Android | Qualcomm Hexagon, MediaTek APU, Samsung NPU |

**Build requirements per platform:**

| Platform | Libraries Required |
|----------|-------------------|
| Android (Qualcomm) | `libLiteRtRuntime.so`, `libQnnHtp.so` |
| Android (MediaTek) | `libLiteRtRuntime.so`, `libneuron_adapter.so` |
| Android (generic) | `libLiteRtRuntime.so`, `libLiteRtGpuAccelerator.so` |
| iOS | LiteRT.framework (GPU only, ANE limited) |
| Linux | `libLiteRtRuntime.so`, `libLiteRtGpuAccelerator.so` |

**Note:** LSTM layers will fall back to CPU on NPU delegates. The GLU layers (Conv1D + sigmoid) will run on NPU, providing ~40-50% acceleration.

## Performance Considerations

- Pre-compute window once, reuse across frames
- Consider SIMD optimization for mag/phase conversion
- Memory pool for FFT buffers to avoid allocation per frame
- For real-time: process in fixed-size chunks matching hopSize
