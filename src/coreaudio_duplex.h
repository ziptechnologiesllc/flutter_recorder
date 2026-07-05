#ifndef FLUTTER_RECORDER_COREAUDIO_DUPLEX_H
#define FLUTTER_RECORDER_COREAUDIO_DUPLEX_H

// Custom single-unit CoreAudio duplex device (iOS RemoteIO).
//
// WHY: miniaudio opens iOS capture+playback as TWO independent AudioUnits
// bridged by a fixed-rate ring (ma_duplex_rb) with no drift compensation, so
// the reference (capture clock) and the speaker output (playback clock) drift
// apart and the echo canceller loses alignment over time. ONE RemoteIO unit
// with EnableIO on both scopes services mic and speaker in a single render
// cycle off ONE hardware clock — mic↔speaker are sample-locked by construction,
// so the echo path is a constant integer delay (exactly periodic), the LSAEC
// precondition.
//
// The render function receives interleaved Float32 mic + speaker buffers, the
// SAME shape miniaudio's data_callback consumes, so the entire app pipeline
// (capture.cpp data_callback, SoLoud slave mix, AEC, scheduler) is unchanged.

#ifdef __cplusplus
extern "C" {
#endif

// Called once per render cycle (audio thread). `mic` is interleaved Float32
// captured input (channels interleaved); fill `speaker` (interleaved Float32)
// with the output. Both hold frameCount frames.
typedef void (*CADuplexRenderFn)(void *userData, const float *mic,
                                 float *speaker, unsigned int frameCount);

// Start a single-unit RemoteIO duplex device at the given rate/channels.
// Returns true on success. renderFn is invoked on the audio thread each cycle.
bool caDuplexStart(unsigned int sampleRate, unsigned int channels,
                   void *userData, CADuplexRenderFn renderFn);

// Stop and dispose the device. Safe to call if not running.
void caDuplexStop(void);

bool caDuplexIsRunning(void);

// The rate/buffer the device actually negotiated (valid after a successful
// start). Lets the caller reconcile if iOS forced a different rate.
unsigned int caDuplexActualSampleRate(void);

#ifdef __cplusplus
}
#endif

#endif // FLUTTER_RECORDER_COREAUDIO_DUPLEX_H
