#include "soloud_slave_bridge.h"
#include <atomic>
#include <stdio.h>

// Note: extern "C" linkage is provided by the header's extern "C" block

// Global callback pointer - initialized to nullptr
SLAVE_BRIDGE_EXPORT SoloudSlaveMixCallback g_soloudSlaveMixCallback = nullptr;

// Atomic flag to track if slave audio has started flowing
// This prevents race conditions during calibration startup
static std::atomic<bool> g_slaveAudioReady{false};

SLAVE_BRIDGE_EXPORT void soloud_registerSlaveMixCallback(SoloudSlaveMixCallback callback) {
  fprintf(stderr,
          "[SoLoud Slave Bridge] Registering slave mix callback: %p\n",
          (void *)callback);
  fflush(stderr);
  g_soloudSlaveMixCallback = callback;
}

SLAVE_BRIDGE_EXPORT void soloud_unregisterSlaveMixCallback() {
  fprintf(stderr, "[SoLoud Slave Bridge] Unregistering slave mix callback\n");
  fflush(stderr);
  g_soloudSlaveMixCallback = nullptr;
  // Also reset the ready flag when unregistering
  g_slaveAudioReady.store(false, std::memory_order_release);
}

SLAVE_BRIDGE_EXPORT bool soloud_isSlaveMode() { return g_soloudSlaveMixCallback != nullptr; }

SLAVE_BRIDGE_EXPORT bool soloud_isSlaveAudioReady() {
  return g_slaveAudioReady.load(std::memory_order_acquire);
}

SLAVE_BRIDGE_EXPORT void soloud_resetSlaveAudioReady() {
  g_slaveAudioReady.store(false, std::memory_order_release);
}

SLAVE_BRIDGE_EXPORT void soloud_setSlaveAudioReady() {
  // Only log the first time
  bool expected = false;
  if (g_slaveAudioReady.compare_exchange_strong(expected, true,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
    fprintf(stderr, "[SoLoud Slave Bridge] Slave audio ready - first callback completed\n");
    fflush(stderr);
  }
}

// ---------------------------------------------------------------------------
// Transport control callbacks (set from flutter_soloud during slave init)
// ---------------------------------------------------------------------------
//
// The pointers themselves are plain globals — registration happens once on
// slave init, never concurrently with calls from the audio thread.
// `Soloud::setVolume` / `setPause` / `stop` are safe to invoke under SoLoud's
// lock-free mode (no mutex acquisition); see lockAudioMutex_internal in
// soloud.cpp where lock-free mode short-circuits to a no-op.

SLAVE_BRIDGE_EXPORT SoloudSetVolumeCallback g_soloudSetVolume = nullptr;
SLAVE_BRIDGE_EXPORT SoloudSetPauseCallback g_soloudSetPause = nullptr;
SLAVE_BRIDGE_EXPORT SoloudStopCallback g_soloudStop = nullptr;

SLAVE_BRIDGE_EXPORT void soloud_registerSlaveControlCallbacks(
    SoloudSetVolumeCallback setVolumeCb,
    SoloudSetPauseCallback setPauseCb,
    SoloudStopCallback stopCb) {
  fprintf(stderr,
          "[SoLoud Slave Bridge] Registering control callbacks: "
          "setVolume=%p setPause=%p stop=%p\n",
          (void *)setVolumeCb, (void *)setPauseCb, (void *)stopCb);
  fflush(stderr);
  g_soloudSetVolume = setVolumeCb;
  g_soloudSetPause = setPauseCb;
  g_soloudStop = stopCb;
}

SLAVE_BRIDGE_EXPORT void soloud_unregisterSlaveControlCallbacks() {
  fprintf(stderr, "[SoLoud Slave Bridge] Unregistering control callbacks\n");
  fflush(stderr);
  g_soloudSetVolume = nullptr;
  g_soloudSetPause = nullptr;
  g_soloudStop = nullptr;
}
