#include "native_scheduler.h"
#include "audio_engine/audio_engine.h"
#include "audio_engine/event.h"
#include "auto_record.h"
#include "capture.h"
#include "common.h"
#include "native_ring_buffer.h"
#include <cstdio>
#include <vector>

// External flag from flutter_recorder.cpp for idempotency
extern std::atomic<bool> g_recordingScheduledOrActive;

// External function from flutter_recorder.cpp to store recorded audio
extern void storeRecordedAudio(float* data, size_t frameCount);

// External WAV path storage - set by scheduler, used by worker thread
extern char g_pendingWavPath[512];
extern std::atomic<bool> g_pendingWavWrite;

#ifdef _IS_ANDROID_
#include <android/log.h>
#define SCHED_LOG(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "NativeScheduler", fmt, ##__VA_ARGS__)
#else
#define SCHED_LOG(fmt, ...) printf("[NativeScheduler] " fmt "\n", ##__VA_ARGS__)
#endif

// Debug logging control - disable for production
#define DEBUG_SCHEDULER 0

#if DEBUG_SCHEDULER
#define SCHED_DEBUG(fmt, ...) SCHED_LOG(fmt, ##__VA_ARGS__)
#else
#define SCHED_DEBUG(fmt, ...) ((void)0)
#endif

// ==================== SINGLETON ====================

NativeScheduler& NativeScheduler::instance() {
    static NativeScheduler instance;
    return instance;
}

NativeScheduler::NativeScheduler() {
    reset();
}

// ==================== CONFIGURATION ====================

void NativeScheduler::setBaseLoop(int64_t loopFrames, int64_t loopStartFrame) {
    int64_t currentFrame =
        flowstate::audio_engine::AudioEngine::instance().getCurrentFrame();
    int64_t ringTotal = g_nativeRingBuffer
        ? (int64_t)g_nativeRingBuffer->getTotalFramesWritten() : -1;
    mBaseLoopFrames.store(loopFrames, std::memory_order_release);
    mBaseLoopStartFrame.store(loopStartFrame, std::memory_order_release);
    SCHED_LOG("setBaseLoop: loopFrames=%lld loopStartFrame=%lld currentGlobalFrame=%lld "
              "ringTotal=%lld divergence(ring-global)=%lld",
              (long long)loopFrames, (long long)loopStartFrame, (long long)currentFrame,
              (long long)ringTotal, (long long)(ringTotal - currentFrame));
}

// Phase 2a: forward to the AudioEngine — single source of truth for the
// global frame counter. Defined out-of-line so the header doesn't have to
// pull in audio_engine.h.
int64_t NativeScheduler::getGlobalFrame() const {
    return flowstate::audio_engine::AudioEngine::instance().getCurrentFrame();
}

void NativeScheduler::clearBaseLoop() {
    mBaseLoopFrames.store(0, std::memory_order_release);
    mBaseLoopStartFrame.store(0, std::memory_order_release);
    SCHED_DEBUG("clearBaseLoop");
}

void NativeScheduler::reset() {
    SCHED_DEBUG("reset");

    // Cancel all events
    cancelAllEvents();

    // Phase 2a: no mGlobalFramePosition to clear — AudioEngine owns the
    // global frame counter and it stays monotonic across session resets
    // (matches the underlying mTotalFramesCaptured timeline). Anything that
    // wants "frames since this session started" must record the start frame
    // and subtract.
    mBaseLoopFrames.store(0, std::memory_order_release);
    mBaseLoopStartFrame.store(0, std::memory_order_release);

    // Reset notification queue
    mNotifyWriteIdx.store(0, std::memory_order_release);
    mNotifyReadIdx.store(0, std::memory_order_release);

    // Reset event ID counter
    mNextEventId.store(1, std::memory_order_release);
}

// ==================== EVENT SCHEDULING ====================

int NativeScheduler::findEmptySlot() {
    for (int i = 0; i < MAX_EVENTS; ++i) {
        EventState expected = EventState::Empty;
        // Try to claim this slot
        if (mEvents[i].state.compare_exchange_strong(
                expected, EventState::Pending,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            // Successfully claimed - but set it back to empty until we're done filling it
            mEvents[i].state.store(EventState::Empty, std::memory_order_release);
            return i;
        }
        // Also try to reclaim cancelled/fired slots
        if (expected == EventState::Cancelled || expected == EventState::Fired) {
            if (mEvents[i].state.compare_exchange_strong(
                    expected, EventState::Pending,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                mEvents[i].state.store(EventState::Empty, std::memory_order_release);
                return i;
            }
        }
    }
    return -1;  // No empty slot
}

uint32_t NativeScheduler::scheduleEvent(SchedulerAction action, int64_t targetFrame,
                                         const char* recordingPath) {
    int slot = findEmptySlot();
    if (slot < 0) {
        SCHED_LOG("scheduleEvent: no empty slot available!");
        return 0;  // Failed
    }

    ScheduledEvent& event = mEvents[slot];

    // Generate event ID
    uint32_t eventId = mNextEventId.fetch_add(1, std::memory_order_relaxed);
    if (eventId == 0) {
        eventId = mNextEventId.fetch_add(1, std::memory_order_relaxed);  // Skip 0
    }

    // Fill in event data
    event.eventId.store(eventId, std::memory_order_relaxed);
    event.action.store(action, std::memory_order_relaxed);
    event.targetFrame.store(targetFrame, std::memory_order_relaxed);

    // Copy recording path if provided
    if (recordingPath != nullptr) {
        strncpy(event.recordingPath, recordingPath, sizeof(event.recordingPath) - 1);
        event.recordingPath[sizeof(event.recordingPath) - 1] = '\0';
    } else {
        event.recordingPath[0] = '\0';
    }

    // Make event visible (pending)
    event.state.store(EventState::Pending, std::memory_order_release);

    SCHED_DEBUG("scheduleEvent: id=%u action=%d targetFrame=%lld slot=%d",
                eventId, (int)action, (long long)targetFrame, slot);

    return eventId;
}

uint32_t NativeScheduler::scheduleQuantizedStart(const char* recordingPath) {
    // CRITICAL: Clear any stale notifications from previous recordings
    // Without this, Dart polling picks up old events and corrupts state
    clearNotifications();

    int64_t currentFrame =
        flowstate::audio_engine::AudioEngine::instance().getCurrentFrame();
    int64_t loopFrames = mBaseLoopFrames.load(std::memory_order_acquire);
    int64_t loopStartFrame = mBaseLoopStartFrame.load(std::memory_order_acquire);

    // Round-trip latency compensation for overdubs. The musician plays along
    // with the loop they HEAR (delayed by output latency) and their input is
    // delayed again (input latency), so their performance lands in the capture
    // stream ~RTL late. Shifting the recording window forward by RTL makes
    // capture-frame (boundary + RTL) become WAV sample 0 — the performance the
    // player intended for the downbeat. The grid (mBaseLoopStartFrame) is NOT
    // moved; only this take's window is offset. Auto/manual stop derive from
    // targetStartFrame so the take stays exactly loopFrames long.
    int64_t latencyComp = mLatencyCompensationFrames.load(std::memory_order_acquire);
    if (latencyComp < 0) latencyComp = 0;
    int64_t targetStartFrame = getNextLoopBoundary() + latencyComp;

    int64_t ringTotal = g_nativeRingBuffer
        ? (int64_t)g_nativeRingBuffer->getTotalFramesWritten() : -1;
    SCHED_LOG("scheduleQuantizedStart: currentFrame=%lld loopFrames=%lld loopStart=%lld latencyComp=%lld "
              "-> targetFrame=%lld ringTotal=%lld divergence(ring-global)=%lld path=%s",
              (long long)currentFrame, (long long)loopFrames, (long long)loopStartFrame,
              (long long)latencyComp, (long long)targetStartFrame, (long long)ringTotal,
              (long long)(ringTotal - currentFrame),
              recordingPath ? recordingPath : "(null)");

    // Schedule the START event
    uint32_t startEventId = scheduleEvent(SchedulerAction::StartRecording, targetStartFrame, recordingPath);

    // If we have a base loop AND auto-stop is enabled, also schedule the STOP event upfront
    bool autoStop = mAutoStopEnabled.load(std::memory_order_acquire);
    if (startEventId != 0 && loopFrames > 0 && autoStop) {
        // STOP is START + loopFrames. targetStartFrame already carries the
        // latency-comp offset, so STOP inherits it — the take stays exactly
        // loopFrames long, just phase-shifted off the grid by latencyComp.
        int64_t targetStopFrame = targetStartFrame + loopFrames;
        uint32_t stopEventId = scheduleEvent(SchedulerAction::StopRecording, targetStopFrame, recordingPath);
        SCHED_LOG("scheduleQuantizedStart: auto-scheduled STOP at frame %lld (startEventId=%u, stopEventId=%u)",
                  (long long)targetStopFrame, startEventId, stopEventId);
    } else if (startEventId != 0 && loopFrames > 0 && !autoStop) {
        SCHED_LOG("scheduleQuantizedStart: auto-stop DISABLED, no STOP scheduled (startEventId=%u)",
                  startEventId);
    }

    return startEventId;
}

uint32_t NativeScheduler::scheduleQuantizedStop(int64_t recordingStartFrame) {
    int64_t loopFrames = mBaseLoopFrames.load(std::memory_order_acquire);
    int64_t currentFrame =
        flowstate::audio_engine::AudioEngine::instance().getCurrentFrame();

    int64_t targetFrame;
    if (loopFrames <= 0) {
        // No base loop (free mode) - stop immediately (next buffer boundary)
        targetFrame = currentFrame;
        SCHED_DEBUG("scheduleQuantizedStop: no base loop, immediate stop at %lld",
                    (long long)targetFrame);
    } else {
        // Loop mode - find next loop boundary
        // No latency compensation in loop mode (quantized start/stop)
        int64_t framesRecorded = currentFrame - recordingStartFrame;

        // Late-tap grace: a stop tapped just AFTER a loop boundary almost
        // certainly meant THAT boundary — the musician reacted late, they
        // didn't want a whole extra loop. If at least one full loop is down
        // and the tap is within the grace window past the boundary, target
        // the boundary just passed (a past frame): processEvents fires the
        // stop on the next buffer and extraction rounds to the nearest loop
        // multiple, trimming the overhang (with its usual crossfade).
        // Grace = min(15% of the loop, 350 ms) — scales down for short loops,
        // capped so long loops don't swallow deliberate early stops.
        int64_t posInLoop = framesRecorded % loopFrames;
        int64_t graceFrames = loopFrames * 15 / 100;
        if (g_nativeRingBuffer != nullptr) {
            int64_t graceCap =
                (int64_t)g_nativeRingBuffer->sampleRate() * 350 / 1000;
            if (graceCap > 0 && graceFrames > graceCap) graceFrames = graceCap;
        }

        int64_t loops;
        if (framesRecorded >= loopFrames && posInLoop <= graceFrames) {
            loops = framesRecorded / loopFrames; // boundary just passed
            SCHED_DEBUG("scheduleQuantizedStop: late tap (%lld frames past "
                        "boundary, grace=%lld) - trimming to previous boundary",
                        (long long)posInLoop, (long long)graceFrames);
        } else {
            loops = (framesRecorded / loopFrames) + 1; // next boundary
        }
        int64_t targetDuration = loops * loopFrames;
        targetFrame = recordingStartFrame + targetDuration;

        SCHED_DEBUG("scheduleQuantizedStop: startFrame=%lld currentFrame=%lld framesRecorded=%lld "
                    "loops=%lld targetFrame=%lld",
                    (long long)recordingStartFrame, (long long)currentFrame,
                    (long long)framesRecorded, (long long)loops, (long long)targetFrame);
    }

    return scheduleEvent(SchedulerAction::StopRecording, targetFrame, nullptr);
}

void NativeScheduler::beginAutoRecording(int64_t recordingStartGlobalFrame,
                                          int64_t preRollFrames,
                                          const char* recordingPath,
                                          Capture* capture) {
    if (recordingPath == nullptr || recordingPath[0] == '\0' || capture == nullptr) {
        return;
    }
    // Don't auto-record over a base loop — the quantized path owns that.
    if (mBaseLoopFrames.load(std::memory_order_acquire) > 0) {
        return;
    }
    // Clear stale notifications from any previous recording (mirrors
    // scheduleQuantizedStart).
    clearNotifications();

    strncpy(mActiveRecordingPath, recordingPath, sizeof(mActiveRecordingPath) - 1);
    mActiveRecordingPath[sizeof(mActiveRecordingPath) - 1] = '\0';
    mRecordingStartFrame = recordingStartGlobalFrame;

    if (preRollFrames < 0) preRollFrames = 0;

    if (g_nativeRingBuffer != nullptr) {
        g_nativeRingBuffer->startRecording((size_t)preRollFrames);
        mRecordingStartTotalFrame = g_nativeRingBuffer->getRecordingStartFrame();
        // Queued: FFI trampolines must not run on the RT audio thread.
        queueRecordingStartedEvent(recordingStartGlobalFrame, recordingPath);
    } else {
        CaptureErrors err = capture->startRecording(recordingPath);
        if (err != captureNoError) {
            mActiveRecordingPath[0] = '\0';
            mRecordingStartFrame = 0;
            return;
        }
        // Queued: FFI trampolines must not run on the RT audio thread.
        queueRecordingStartedEvent(recordingStartGlobalFrame, recordingPath);
    }

    // Idempotency flag (mirrors flutter_recorder_startRecording).
    g_recordingScheduledOrActive.store(true, std::memory_order_release);

    // Notify Dart — legacy poll-notification queue ...
    uint32_t eventId = mNextEventId.fetch_add(1, std::memory_order_relaxed);
    if (eventId == 0) eventId = mNextEventId.fetch_add(1, std::memory_order_relaxed);
    EventNotification notif;
    notif.eventId = eventId;
    notif.action = SchedulerAction::StartRecording;
    notif.firedAtFrame = recordingStartGlobalFrame;
    notif.latencyFrames = 0;
    pushNotification(notif);

    // ... and the unified AudioEngine event stream (single-producer invariant
    // preserved: this runs on the same audio thread, sequential with
    // AudioEngine::process(), like processEvents).
    {
        using flowstate::audio_engine::AudioEngine;
        using flowstate::audio_engine::Event;
        Event ev{};
        ev.type            = Event::Type::RecordingStarted;
        ev.id              = eventId;
        ev.frame           = recordingStartGlobalFrame;
        ev.framesProcessed = 0;
        ev.soundHash       = 0;
        ev.code            = 0;
        AudioEngine::instance().emitFromAudioThread(ev);
    }

    SCHED_DEBUG("beginAutoRecording: start=%lld preRoll=%lld path=%s",
                (long long)recordingStartGlobalFrame, (long long)preRollFrames,
                recordingPath);
}

bool NativeScheduler::cancelEvent(uint32_t eventId) {
    if (eventId == 0) return false;

    for (int i = 0; i < MAX_EVENTS; ++i) {
        if (mEvents[i].eventId.load(std::memory_order_acquire) == eventId) {
            EventState expected = EventState::Pending;
            if (mEvents[i].state.compare_exchange_strong(
                    expected, EventState::Cancelled,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                SCHED_DEBUG("cancelEvent: id=%u cancelled", eventId);
                return true;
            }
        }
    }
    SCHED_DEBUG("cancelEvent: id=%u not found or already fired", eventId);
    return false;
}

void NativeScheduler::cancelAllEvents() {
    for (int i = 0; i < MAX_EVENTS; ++i) {
        EventState state = mEvents[i].state.load(std::memory_order_acquire);
        if (state == EventState::Pending) {
            mEvents[i].state.store(EventState::Cancelled, std::memory_order_release);
        }
    }
    SCHED_DEBUG("cancelAllEvents");
}

// ==================== AUDIO THREAD PROCESSING ====================

void NativeScheduler::processEvents(int64_t bufferStartFrame, uint32_t frameCount,
                                     Capture* capture) {
    // Phase 2a: AudioEngine::process() (running later in this same audio
    // callback) publishes the global frame for us — no local store needed.
    int64_t bufferEndFrame = bufferStartFrame + frameCount;

    // Check each event slot
    for (int i = 0; i < MAX_EVENTS; ++i) {
        EventState state = mEvents[i].state.load(std::memory_order_acquire);
        if (state != EventState::Pending) continue;

        int64_t targetFrame = mEvents[i].targetFrame.load(std::memory_order_acquire);
        SchedulerAction action = mEvents[i].action.load(std::memory_order_acquire);

        // StopRecording must fire on the buffer AFTER the one containing target,
        // so that the buffer covering [target-frameCount, target] has been
        // written to the ring buffer. Otherwise stopRecording extracts up to
        // (E - S) * frameCount frames where E is the firing buffer and S is the
        // recording start — short of the requested loops*loopFrames by up to
        // one full buffer (~256 frames @ 48k), causing per-take drift.
        // Other actions still fire on the buffer containing target.
        const bool shouldFire = (action == SchedulerAction::StopRecording)
            ? (targetFrame < bufferStartFrame)
            : (targetFrame <= bufferEndFrame);
        if (shouldFire) {
            // Try to claim this event for execution
            EventState expected = EventState::Pending;
            if (mEvents[i].state.compare_exchange_strong(
                    expected, EventState::Fired,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {

                // Calculate latency (negative if early, positive if late)
                int32_t latencyFrames = 0;
                if (targetFrame < bufferStartFrame) {
                    latencyFrames = (int32_t)(bufferStartFrame - targetFrame);
                }

                SCHED_DEBUG("processEvents: firing id=%u action=%d targetFrame=%lld "
                            "bufferStart=%lld latency=%d",
                            mEvents[i].eventId.load(std::memory_order_relaxed),
                            (int)mEvents[i].action.load(std::memory_order_relaxed),
                            (long long)targetFrame, (long long)bufferStartFrame, latencyFrames);

                // Execute the event
                executeEvent(mEvents[i], bufferStartFrame, frameCount, capture);

                // Push notification for Dart (legacy path, still polled by
                // flutter_recorder_scheduler_pollNotification).
                EventNotification notif;
                notif.eventId = mEvents[i].eventId.load(std::memory_order_relaxed);
                notif.action = mEvents[i].action.load(std::memory_order_relaxed);
                notif.firedAtFrame = bufferStartFrame;
                notif.latencyFrames = latencyFrames;
                pushNotification(notif);

                // Phase 2c.4 — also emit through the AudioEngine event
                // stream so Dart can consume the unified stream instead of
                // (or alongside) the legacy poll-notification API. Same
                // data_callback, sequential with AudioEngine::process(), so
                // the single-producer invariant on AudioEngine's outbox is
                // preserved.
                {
                  using flowstate::audio_engine::AudioEngine;
                  using flowstate::audio_engine::Event;
                  Event::Type evType = Event::Type::None;
                  switch (notif.action) {
                    case SchedulerAction::StartRecording:
                      evType = Event::Type::RecordingStarted; break;
                    case SchedulerAction::StopRecording:
                      evType = Event::Type::RecordingStopped; break;
                    case SchedulerAction::StartPlayback:
                      evType = Event::Type::PlaybackStarted; break;
                    case SchedulerAction::StopPlayback:
                      evType = Event::Type::PlaybackEnded; break;
                    case SchedulerAction::None:
                      break;
                  }
                  if (evType != Event::Type::None) {
                    Event ev{};
                    ev.type            = evType;
                    ev.id              = notif.eventId;
                    // targetFrame is the sample-accurate intent; firedAtFrame
                    // is the start of the buffer that executed the event.
                    // Prefer the intent so Dart consumers see the same frame
                    // regardless of buffer-size variation.
                    ev.frame           = targetFrame;
                    ev.framesProcessed = 0;
                    ev.soundHash       = 0;
                    ev.code            = static_cast<std::uint32_t>(
                        latencyFrames < 0 ? 0 : latencyFrames);
                    AudioEngine::instance().emitFromAudioThread(ev);
                  }
                }

                // Mark slot as available for reuse
                mEvents[i].reset();
            }
        }
    }
}

void NativeScheduler::executeEvent(ScheduledEvent& event, int64_t currentFrame,
                                    uint32_t audioFrameCount, Capture* capture) {
    if (capture == nullptr) {
        // AUDIO THREAD: No printf allowed
        return;
    }

    SchedulerAction action = event.action.load(std::memory_order_acquire);
    int64_t targetFrame = event.targetFrame.load(std::memory_order_acquire);

    switch (action) {
        case SchedulerAction::StartRecording:
            // AUDIO THREAD: No printf allowed - use SCHED_DEBUG (disabled in production)
            if (event.recordingPath[0] != '\0') {
                SCHED_DEBUG("executeEvent: startRecording at frame %lld (target=%lld)",
                            (long long)currentFrame, (long long)targetFrame);
                // Store path for stop callback
                strncpy(mActiveRecordingPath, event.recordingPath, sizeof(mActiveRecordingPath) - 1);
                mActiveRecordingPath[sizeof(mActiveRecordingPath) - 1] = '\0';
                // Recording start is the actual audio target frame, not the firing
                // buffer's start. The buffer that fires the event covers
                // [currentFrame, currentFrame + audioFrameCount); target lands
                // somewhere inside it.
                mRecordingStartFrame = targetFrame;

                if (g_nativeRingBuffer != nullptr) {
                    int64_t loopFrames = mBaseLoopFrames.load(std::memory_order_acquire);
                    int64_t preRollFrames = 0;
                    if (loopFrames > 0) {
                        // Loop mode: the firing buffer has already been written to the
                        // ring (capture writes before processEvents). Pull the tail of
                        // that buffer — frames [target, bufferEnd) — out of the wrap
                        // region as pre-roll so linear[0..] represents audio starting
                        // exactly at target. Without this, linear[0..] starts one full
                        // audio buffer late (the ~5.3ms phase shift on every overdub).
                        int64_t offsetInBuffer = targetFrame - currentFrame;
                        if (offsetInBuffer < 0) offsetInBuffer = 0;
                        if (offsetInBuffer > (int64_t)audioFrameCount)
                            offsetInBuffer = (int64_t)audioFrameCount;
                        preRollFrames = (int64_t)audioFrameCount - offsetInBuffer;
                    } else {
                        // Free mode: apply latency compensation for touch/bluetooth latency
                        preRollFrames = mLatencyCompensationFrames.load(std::memory_order_acquire);
                    }
                    g_nativeRingBuffer->startRecording(preRollFrames > 0 ? (size_t)preRollFrames : 0);
                    mRecordingStartTotalFrame = g_nativeRingBuffer->getRecordingStartFrame();
                    SCHED_DEBUG("Ring buffer recording started, preRoll=%lld", (long long)preRollFrames);

                    // Notify Dart with the actual audio start frame (target) so
                    // _recordingStartGlobalSample matches the scheduler's view.
                    // Queued: FFI trampolines must not run on the RT audio thread.
                    queueRecordingStartedEvent(targetFrame, event.recordingPath);
                } else {
                    SCHED_DEBUG("executeEvent: no ring buffer, falling back to capture");
                    CaptureErrors err = capture->startRecording(event.recordingPath);
                    if (err != captureNoError) {
                        SCHED_DEBUG("executeEvent: startRecording failed with error %d", (int)err);
                        mActiveRecordingPath[0] = '\0';
                        g_recordingScheduledOrActive.store(false, std::memory_order_release);
                    } else {
                        // Queued: FFI trampolines must not run on the RT audio thread.
                        queueRecordingStartedEvent(targetFrame, event.recordingPath);
                    }
                }
            } else {
                SCHED_DEBUG("executeEvent: StartRecording but no path!");
            }
            break;

        case SchedulerAction::StopRecording:
            {
                // AUDIO THREAD: No printf/fprintf allowed here!
                // All logging changed to SCHED_DEBUG (disabled in production)
                SCHED_DEBUG("executeEvent: stopRecording at frame %lld", (long long)currentFrame);

                // Determine WAV path: prefer mActiveRecordingPath (loop mode), fallback to event.recordingPath (free mode)
                const char* wavPath = (mActiveRecordingPath[0] != '\0') ? mActiveRecordingPath : event.recordingPath;

                int64_t recordedFrames = 0;

                if (g_nativeRingBuffer != nullptr && g_nativeRingBuffer->isRecording()) {
                    // Calculate expected frame count for sample-accurate loop multiples
                    size_t expectedFrameCount = 0;
                    int64_t loopFrames = mBaseLoopFrames.load(std::memory_order_acquire);
                    if (loopFrames > 0 && mRecordingStartTotalFrame > 0) {
                        // CRITICAL: Use ring buffer's total frames, NOT scheduler's global frame
                        // The scheduler's globalFrame and ring buffer's totalFramesWritten can diverge
                        // if they start counting at different times or have different frame sources.
                        size_t currentRingBufferTotal = g_nativeRingBuffer->getTotalFramesWritten();
                        int64_t framesRecorded = (int64_t)currentRingBufferTotal - (int64_t)mRecordingStartTotalFrame;
                        int64_t loops = (framesRecorded + loopFrames / 2) / loopFrames; // Round to nearest
                        if (loops < 1) loops = 1;
                        expectedFrameCount = (size_t)(loops * loopFrames);
                        SCHED_DEBUG("Loop mode: expecting %zu frames (%lld loops x %lld, ringTotal=%zu, startTotal=%zu)",
                                    expectedFrameCount, (long long)loops, (long long)loopFrames,
                                    currentRingBufferTotal, (size_t)mRecordingStartTotalFrame);
                    }

                    // Extract recorded audio from ring buffer
                    size_t frameCount = 0;
                    float* audioData = g_nativeRingBuffer->stopRecording(&frameCount, expectedFrameCount);

                    if (audioData != nullptr && frameCount > 0) {
                        recordedFrames = (int64_t)frameCount;

                        // Set pending WAV path for worker thread to write
                        // This avoids file I/O on audio thread (glitch-free)
                        if (wavPath[0] != '\0') {
                            strncpy(g_pendingWavPath, wavPath, sizeof(g_pendingWavPath) - 1);
                            g_pendingWavPath[sizeof(g_pendingWavPath) - 1] = '\0';
                            g_pendingWavWrite.store(true, std::memory_order_release);
                            SCHED_DEBUG("WAV path set for worker thread: %s", wavPath);
                        }

                        // Store audio for worker thread (transfers ownership)
                        // Worker will write WAV first, then pass to SoLoud
                        storeRecordedAudio(audioData, frameCount);
                        SCHED_DEBUG("Stored %zu frames for worker thread", frameCount);
                        // Note: audioData ownership transferred - do NOT delete here
                    } else {
                        SCHED_DEBUG("No audio data extracted (frames=%zu)", frameCount);
                        // NOTE: Intentionally NOT calling delete[] on audio thread
                        // Let it leak rather than risk blocking
                    }
                } else {
                    // Fallback: capture was used directly
                    capture->stopRecording();
                    recordedFrames = currentFrame - mRecordingStartFrame;
                }

                SCHED_DEBUG("Recording stopped: %lld frames", (long long)recordedFrames);

                if (wavPath[0] != '\0') {
                    // Queued: delivered by the looper worker AFTER the WAV
                    // write (FFI trampolines must not run on the RT thread).
                    queueRecordingStoppedEvent(recordedFrames, wavPath);
                }

                // Reset idempotency flag
                g_recordingScheduledOrActive.store(false, std::memory_order_release);

                mActiveRecordingPath[0] = '\0';
                mRecordingStartFrame = 0;
                mRecordingStartTotalFrame = 0;

                // If this take was hands-free (auto-record), put the AutoRecorder
                // back to Idle so a stale armed/recording state can't schedule a
                // spurious stop later. No-op when it was already Idle.
                AutoRecorder::instance().onRecordingStopped();
            }
            break;

        case SchedulerAction::StartPlayback:
            // Future: integrate with SoLoud playback
            SCHED_DEBUG("executeEvent: StartPlayback (not implemented)");
            break;

        case SchedulerAction::StopPlayback:
            // Future: integrate with SoLoud playback
            SCHED_DEBUG("executeEvent: StopPlayback (not implemented)");
            break;

        case SchedulerAction::None:
        default:
            break;
    }
}

// ==================== NOTIFICATION QUEUE ====================

void NativeScheduler::pushNotification(const EventNotification& notif) {
    uint32_t writeIdx = mNotifyWriteIdx.load(std::memory_order_relaxed);
    uint32_t nextIdx = (writeIdx + 1) % NOTIFICATION_QUEUE_SIZE;

    uint32_t readIdx = mNotifyReadIdx.load(std::memory_order_acquire);
    if (nextIdx == readIdx) {
        // Queue full - drop oldest notification
        SCHED_DEBUG("pushNotification: queue full, dropping oldest");
        // Move read index forward (losing oldest entry)
        mNotifyReadIdx.store((readIdx + 1) % NOTIFICATION_QUEUE_SIZE,
                             std::memory_order_release);
    }

    // Write notification
    mNotifications[writeIdx] = notif;

    // Make notification visible
    mNotifyWriteIdx.store(nextIdx, std::memory_order_release);
}

bool NativeScheduler::pollNotification(EventNotification* out) {
    uint32_t readIdx = mNotifyReadIdx.load(std::memory_order_relaxed);
    uint32_t writeIdx = mNotifyWriteIdx.load(std::memory_order_acquire);

    if (readIdx == writeIdx) {
        return false;  // Queue empty
    }

    // Read notification
    *out = mNotifications[readIdx];

    // Move read index forward
    mNotifyReadIdx.store((readIdx + 1) % NOTIFICATION_QUEUE_SIZE,
                         std::memory_order_release);

    return true;
}

bool NativeScheduler::hasNotifications() const {
    uint32_t readIdx = mNotifyReadIdx.load(std::memory_order_acquire);
    uint32_t writeIdx = mNotifyWriteIdx.load(std::memory_order_acquire);
    return readIdx != writeIdx;
}

void NativeScheduler::clearNotifications() {
    // Reset queue indices to effectively clear all pending notifications
    mNotifyWriteIdx.store(0, std::memory_order_release);
    mNotifyReadIdx.store(0, std::memory_order_release);
    SCHED_DEBUG("clearNotifications: queue cleared");
}

// ==================== STATE ACCESSORS ====================

int64_t NativeScheduler::getNextLoopBoundary() const {
    return getNextLoopBoundaryFrom(
        flowstate::audio_engine::AudioEngine::instance().getCurrentFrame());
}

int64_t NativeScheduler::getNextLoopBoundaryFrom(int64_t fromFrame) const {
    int64_t loopFrames = mBaseLoopFrames.load(std::memory_order_acquire);
    int64_t loopStart = mBaseLoopStartFrame.load(std::memory_order_acquire);

    if (loopFrames <= 0) {
        // No base loop - return current frame (immediate)
        return fromFrame;
    }

    // Calculate position within loop cycle
    int64_t positionInCycle = fromFrame - loopStart;

    // Handle negative position (before loop start)
    if (positionInCycle < 0) {
        return loopStart;
    }

    // Calculate frames until next boundary
    int64_t framesIntoCurrentLoop = positionInCycle % loopFrames;
    int64_t framesToNextBoundary = loopFrames - framesIntoCurrentLoop;

    // If exactly on boundary, return next boundary (not current)
    if (framesToNextBoundary == loopFrames) {
        framesToNextBoundary = loopFrames;
    }

    return fromFrame + framesToNextBoundary;
}

// Extern C accessors for looper_bridge.cpp (in flutter_soloud plugin).
// These cross the plugin boundary — called from the worker thread.
extern "C" {
    int64_t looper_getGlobalFrame() {
        return NativeScheduler::instance().getGlobalFrame();
    }
    int64_t looper_getBaseLoopFrames() {
        return NativeScheduler::instance().getBaseLoopFrames();
    }
    int64_t looper_getBaseLoopStartFrame() {
        return NativeScheduler::instance().getBaseLoopStartFrame();
    }
}
