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
