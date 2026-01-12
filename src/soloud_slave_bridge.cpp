#include "soloud_slave_bridge.h"
#include <stdio.h>

// Note: extern "C" linkage is provided by the header's extern "C" block

// Global callback pointer - initialized to nullptr
SLAVE_BRIDGE_EXPORT SoloudSlaveMixCallback g_soloudSlaveMixCallback = nullptr;

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
}

SLAVE_BRIDGE_EXPORT bool soloud_isSlaveMode() { return g_soloudSlaveMixCallback != nullptr; }
