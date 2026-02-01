#include "playback_ready_buffer.h"

// Global playback ready buffer instance
// Mirrors the pattern used by g_nativeRingBuffer
PlaybackReadyBuffer* g_playbackReadyBuffer = nullptr;
