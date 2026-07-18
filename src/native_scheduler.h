#ifndef NATIVE_SCHEDULER_H
#define NATIVE_SCHEDULER_H

#include <atomic>
#include <cstdint>
#include <cstring>

// Forward declaration
class Capture;

// Event actions (matches Dart LaunchAction enum)
enum class SchedulerAction : uint8_t {
    None = 0,
    StartRecording = 1,
    StopRecording = 2,
    StartPlayback = 3,
    StopPlayback = 4,
};

// Event slot state
enum class EventState : uint8_t {
    Empty = 0,      // Slot available
    Pending = 1,    // Event scheduled, waiting
    Fired = 2,      // Event executed, pending Dart notification
    Cancelled = 3,  // Event cancelled
};

// Fixed-size event slot (no dynamic allocation)
struct ScheduledEvent {
    std::atomic<EventState> state{EventState::Empty};
    std::atomic<int64_t> targetFrame{0};
    std::atomic<SchedulerAction> action{SchedulerAction::None};
    std::atomic<uint32_t> eventId{0};
    char recordingPath[256];  // For StartRecording - copied atomically during schedule

    // Reset slot to empty state
    void reset() {
        state.store(EventState::Empty, std::memory_order_release);
        targetFrame.store(0, std::memory_order_relaxed);
        action.store(SchedulerAction::None, std::memory_order_relaxed);
        eventId.store(0, std::memory_order_relaxed);
        recordingPath[0] = '\0';
    }
};

// Notification for Dart (event fired)
struct EventNotification {
    uint32_t eventId;
    SchedulerAction action;
    int64_t firedAtFrame;
    int32_t latencyFrames;  // How many frames late (0 = perfect, positive = late)
};

// Singleton native scheduler for sample-accurate timing
// Designed for lock-free operation in audio thread
class NativeScheduler {
public:
    static NativeScheduler& instance();

    // ===================== CONFIGURATION =====================
    // Called from Dart before audio starts or when loop changes

    /// Set the base loop length in frames for quantization
    void setBaseLoop(int64_t loopFrames, int64_t loopStartFrame = 0);

    /// Clear base loop (free recording mode)
    void clearBaseLoop();

    /// Set latency compensation in frames (applied at recording start)
    void setLatencyCompensationFrames(int64_t frames) {
        mLatencyCompensationFrames.store(frames, std::memory_order_release);
    }

    /// Get latency compensation in frames
    int64_t getLatencyCompensationFrames() const {
        return mLatencyCompensationFrames.load(std::memory_order_acquire);
    }

    /// Set auto-stop enabled (when true, STOP is scheduled upfront with START)
    void setAutoStopEnabled(bool enabled) {
        mAutoStopEnabled.store(enabled, std::memory_order_release);
    }

    /// Get auto-stop enabled state
    bool isAutoStopEnabled() const {
        return mAutoStopEnabled.load(std::memory_order_acquire);
    }

    /// How many base-loop cycles an overdub records before the upfront STOP.
    /// 1 = one cycle (default), N = N cycles, 0 = no upfront stop — record
    /// until the user taps stop (the manual stop still quantizes to a loop
    /// boundary, so the on-grid length invariant holds either way).
    void setRecordCycles(int32_t cycles) {
        mRecordCycles.store(cycles, std::memory_order_release);
    }

    /// Get the configured record-cycles multiplier
    int32_t getRecordCycles() const {
        return mRecordCycles.load(std::memory_order_acquire);
    }

    /// Reset all state (call on session end)
    void reset();

    // ===================== EVENT SCHEDULING =====================
    // Called from Dart, must be lock-free

    /// Schedule recording start at next loop boundary
    /// Returns event ID (0 if failed to schedule)
    uint32_t scheduleQuantizedStart(const char* recordingPath);

    /// Schedule recording stop at loop boundary
    /// [recordingStartFrame] is when recording started (for multi-loop calculation)
    /// Returns event ID (0 if failed to schedule)
    uint32_t scheduleQuantizedStop(int64_t recordingStartFrame);

    /// Schedule event at specific frame
    /// Returns event ID (0 if failed to schedule)
    uint32_t scheduleEvent(SchedulerAction action, int64_t targetFrame,
                           const char* recordingPath = nullptr);

    /// Begin a recording right now (audio thread), at [recordingStartGlobalFrame]
    /// — the downbeat the auto-recorder (AutoRecorder) detected — rewinding the
    /// native ring buffer by [preRollFrames] so linear[0..] starts exactly there
    /// and the lead-in silence is trimmed. Mirrors the StartRecording branch of
    /// executeEvent so the recording bookkeeping (active path, ring-buffer start
    /// total, the Dart RecordingStarted notification + AudioEngine event, the
    /// idempotency flag) is identical to the scheduled/quantized path.
    /// Call only when no base loop exists (free / first-loop mode).
    void beginAutoRecording(int64_t recordingStartGlobalFrame, int64_t preRollFrames,
                            const char* recordingPath, Capture* capture);

    /// Cancel a scheduled event by ID
    /// Returns true if event was found and cancelled
    bool cancelEvent(uint32_t eventId);

    /// Cancel all pending events
    void cancelAllEvents();

    // ===================== AUDIO THREAD PROCESSING =====================
    // Called from data_callback(), executes events at buffer boundaries

    /// Process scheduled events for this buffer
    /// [bufferStartFrame] - global frame at start of this buffer
    /// [frameCount] - frames in this buffer
    /// [capture] - Capture instance for start/stop recording
    void processEvents(int64_t bufferStartFrame, uint32_t frameCount, Capture* capture);

    // ===================== DART NOTIFICATION =====================
    // Called from Dart main thread to get fired event info

    /// Poll for next notification. Returns true if notification available.
    bool pollNotification(EventNotification* out);

    /// Check if there are pending notifications
    bool hasNotifications() const;

    /// Clear all pending notifications (call before starting new recording)
    void clearNotifications();

    // ===================== STATE ACCESSORS =====================

    /// Get current global frame position. Phase 2a: forwards to the
    /// AudioEngine which is now the canonical source — NativeScheduler's
    /// own counter has been retired. Defined out-of-line so the include
    /// graph stays scoped to the .cpp.
    int64_t getGlobalFrame() const;

    /// Get base loop length in frames (0 if no base loop)
    int64_t getBaseLoopFrames() const { return mBaseLoopFrames.load(std::memory_order_acquire); }

    /// Get the global frame where the base loop started
    int64_t getBaseLoopStartFrame() const { return mBaseLoopStartFrame.load(std::memory_order_acquire); }

    /// Get frame of next loop boundary from current position
    int64_t getNextLoopBoundary() const;

    /// Calculate next loop boundary from given frame
    int64_t getNextLoopBoundaryFrom(int64_t fromFrame) const;

    /// Get recording start frame (set when StartRecording event fires)
    int64_t getRecordingStartFrame() const { return mRecordingStartFrame; }

    /// Check if recording is active via scheduler
    bool isRecordingActive() const { return mActiveRecordingPath[0] != '\0'; }

private:
    NativeScheduler();
    ~NativeScheduler() = default;

    // Non-copyable
    NativeScheduler(const NativeScheduler&) = delete;
    NativeScheduler& operator=(const NativeScheduler&) = delete;

    // ===================== CONSTANTS =====================
    static constexpr int MAX_EVENTS = 8;
    static constexpr int NOTIFICATION_QUEUE_SIZE = 16;

    // ===================== EVENT QUEUE =====================
    ScheduledEvent mEvents[MAX_EVENTS];
    std::atomic<uint32_t> mNextEventId{1};

    // ===================== TIMING STATE =====================
    // Phase 2a: mGlobalFramePosition retired — AudioEngine::getCurrentFrame()
    // is now the single source of truth. getGlobalFrame() forwards to it.
    std::atomic<int64_t> mBaseLoopFrames{0};
    std::atomic<int64_t> mBaseLoopStartFrame{0};
    std::atomic<int64_t> mLatencyCompensationFrames{0};  // Frames to rewind at recording start
    std::atomic<bool> mAutoStopEnabled{false};  // When true, auto-schedule STOP with START
    std::atomic<int32_t> mRecordCycles{1};      // Loop cycles per overdub take (0 = manual stop)

    // ===================== NOTIFICATION QUEUE (SPSC) =====================
    // Single-Producer (audio thread) / Single-Consumer (Dart poll)
    EventNotification mNotifications[NOTIFICATION_QUEUE_SIZE];
    std::atomic<uint32_t> mNotifyWriteIdx{0};
    std::atomic<uint32_t> mNotifyReadIdx{0};

    // ===================== ACTIVE RECORDING TRACKING =====================
    // For firing Dart callback when recording stops
    char mActiveRecordingPath[256] = {0};
    int64_t mRecordingStartFrame = 0;
    size_t mRecordingStartTotalFrame = 0;  // Ring buffer total frame at start (for fork extraction)

    // ===================== INTERNAL HELPERS =====================

    /// Find an empty event slot
    int findEmptySlot();

    /// Push notification to queue (called from audio thread)
    void pushNotification(const EventNotification& notif);

    /// Execute a single event (called from processEvents).
    /// [audioFrameCount] is the size in frames of the audio buffer that fired
    /// the event — used to compute the sub-buffer offset of the target frame
    /// so loop-mode StartRecording can capture the post-target tail of the
    /// firing buffer as pre-roll (eliminates the one-buffer phase shift).
    void executeEvent(ScheduledEvent& event, int64_t currentFrame,
                      uint32_t audioFrameCount, Capture* capture);
};

#endif // NATIVE_SCHEDULER_H
