#ifndef NEURAL_POST_FILTER_H
#define NEURAL_POST_FILTER_H

#include <string>
#include <vector>

/**
 * Neural Post-Filter for Acoustic Echo Cancellation.
 *
 * This class acts as a wrapper around a TFLite model (e.g., DTLN-aec)
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
   * Initializes the TFLite interpreter and loads the model.
   * @param modelPath Path to the .tflite model file.
   * @return true if successful.
   */
  bool loadModel(const std::string &modelPath);

  /**
   * Processes a block of audio.
   * @param micSignal The microphone signal (after linear AEC).
   * @param refSignal The reference loopback signal.
   * @param output The clean output signal.
   * @param frameCount Number of frames to process.
   */
  void process(const float *micSignal, const float *refSignal, float *output,
               unsigned int frameCount);

  /**
   * Resets the internal state (LSTM hidden states, etc.).
   */
  void reset();

  void setEnabled(bool enabled) { mEnabled = enabled; }
  bool isEnabled() const { return mEnabled; }

private:
  unsigned int mSampleRate;
  unsigned int mChannels;
  bool mEnabled;
  bool mIsLoaded;

  // TODO: TFLite Interpreter and Delegate members
  // std::unique_ptr<tflite::Interpreter> mInterpreter;
};

#endif // NEURAL_POST_FILTER_H
