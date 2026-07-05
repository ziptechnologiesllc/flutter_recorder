#ifndef SOLOUD_SLAVE_BRIDGE_H
#define SOLOUD_SLAVE_BRIDGE_H

#include <cstddef>

// Export macro for symbol visibility (needed for dlsym to find these)
#if defined(_WIN32) || defined(__CYGWIN__)
#define SLAVE_BRIDGE_EXPORT __declspec(dllexport)
#else
#define SLAVE_BRIDGE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SoLoud Slave Bridge - Enables unified audio device mode for AEC synchronization.
 *
 * On Linux (PipeWire/ALSA), separate audio devices have independent clocks that
 * drift apart, causing AEC reference buffer synchronization failures. This bridge
 * allows SoLoud to run in "slave mode" where the Capture plugin's duplex device
 * drives audio output, ensuring perfect clock synchronization.
 *
 * Usage:
 *   1. SoLoud initializes in slave mode (no audio device created)
 *   2. SoLoud registers its mix callback via soloud_registerSlaveMixCallback()
 *   3. Capture's data_callback() calls the mix callback to get SoLoud output
 *   4. Capture plays the mixed audio through its duplex device's playback
 *   5. AEC reference buffer is written in the same callback = no drift
 */

// Callback type for slave mode mixing
// Parameters:
//   output: buffer to fill with mixed audio (interleaved float samples)
//   frameCount: number of frames to generate
//   channels: number of channels (e.g., 2 for stereo)
// Note: Caller must ensure output buffer has space for frameCount * channels floats
typedef void (*SoloudSlaveMixCallback)(float *output, unsigned int frameCount,
                                       unsigned int channels);

// Global callback pointer (set by flutter_soloud when in slave mode)
extern SLAVE_BRIDGE_EXPORT SoloudSlaveMixCallback g_soloudSlaveMixCallback;

// Register the SoLoud mix callback (called from flutter_soloud during slave init)
SLAVE_BRIDGE_EXPORT void soloud_registerSlaveMixCallback(SoloudSlaveMixCallback callback);

// Unregister the SoLoud mix callback (called during deinit)
SLAVE_BRIDGE_EXPORT void soloud_unregisterSlaveMixCallback();

// Check if slave mode is active
SLAVE_BRIDGE_EXPORT bool soloud_isSlaveMode();

// Check if slave audio is ready (at least one successful callback has run)
// This is used to ensure the audio pipeline is flowing before starting calibration
SLAVE_BRIDGE_EXPORT bool soloud_isSlaveAudioReady();

// Reset the slave audio ready flag (called during deinit)
SLAVE_BRIDGE_EXPORT void soloud_resetSlaveAudioReady();

// Mark slave audio as ready (called from data_callback on first successful mix)
SLAVE_BRIDGE_EXPORT void soloud_setSlaveAudioReady();

// ---------------------------------------------------------------------------
// Transport control bridge (Phase 1 of native-audio-engine work)
// ---------------------------------------------------------------------------
//
// Symmetric to the mix-callback bridge but in the opposite direction:
// flutter_soloud registers thin trampolines that call SoLoud's lock-free
// per-voice setters, and flutter_recorder's `audio_engine` calls them from
// its audio thread when a [PendingAction] fires. This eliminates the Dart
// round-trip on quantized transport changes — the audio thread can apply
// volume / pause / stop directly at the fire frame.
//
// Thread safety: SoLoud's slave mode runs in lock-free mode, so
// `Soloud::setVolume()` / `setPause()` / `stop()` do not acquire mutexes —
// they just modify per-voice atomics. Safe to call from the audio thread.

typedef void (*SoloudSetVolumeCallback)(unsigned int voiceHandle, float volume);
typedef void (*SoloudSetPauseCallback)(unsigned int voiceHandle, bool pause);
typedef void (*SoloudStopCallback)(unsigned int voiceHandle);

extern SLAVE_BRIDGE_EXPORT SoloudSetVolumeCallback g_soloudSetVolume;
extern SLAVE_BRIDGE_EXPORT SoloudSetPauseCallback g_soloudSetPause;
extern SLAVE_BRIDGE_EXPORT SoloudStopCallback g_soloudStop;

// Register transport-control trampolines. Each pointer may be null if the
// corresponding setter isn't needed yet. Called by flutter_soloud at slave
// init via dlsym, mirroring how the mix callback is registered.
SLAVE_BRIDGE_EXPORT void soloud_registerSlaveControlCallbacks(
    SoloudSetVolumeCallback setVolumeCb,
    SoloudSetPauseCallback setPauseCb,
    SoloudStopCallback stopCb);

// Clear all transport-control trampolines (called during slave deinit).
SLAVE_BRIDGE_EXPORT void soloud_unregisterSlaveControlCallbacks();

#ifdef __cplusplus
}
#endif

#endif // SOLOUD_SLAVE_BRIDGE_H
