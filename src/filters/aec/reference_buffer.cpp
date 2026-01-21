#include "reference_buffer.h"

// Pre-allocate default buffer at static initialization time
// This ensures the buffer is always available from the first callback
// Default: 2 seconds @ 48kHz stereo (max expected configuration)
static AECReferenceBuffer s_defaultAecReferenceBuffer(
    48000 * 2,  // 2 seconds of frames
    2,          // stereo (max channels)
    48000       // 48kHz sample rate
);

// Global reference buffer pointer - points to pre-allocated buffer by default
// The Dart side can reconfigure it via flutter_recorder_aec_createReferenceBuffer
AECReferenceBuffer* g_aecReferenceBuffer = &s_defaultAecReferenceBuffer;