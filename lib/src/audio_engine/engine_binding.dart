// Dart-side binding to the C++ AudioEngine.
//
// Phase 1 surface:
//   - [EngineBinding.snapshots]: a broadcast stream of [Snapshot]s polled at
//     ~120 Hz from the audio thread's seqlock. UI subscribes for live state.
//   - [EngineBinding.events]: a broadcast stream of [Event]s drained from the
//     audio thread's SPSC outbox at ~120 Hz. Used by AppBloc to fold into
//     AppState. (Phase 1: native emits no events yet.)
//   - [EngineBinding.postCommand]: enqueue a [Command] for the audio thread.
//     (Phase 1: native ignores all commands.)
//
// This file deliberately has zero business logic. It is a mechanical
// translator between FFI and Dart-typed records. AppBloc lives upstream.

import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'native_structs.dart';

// ---------------------------------------------------------------------------
// Public immutable Dart types
// ---------------------------------------------------------------------------

/// Active sync source kind. Mirrors C++ `SyncSourceKind`.
enum SyncSourceKind { none, local, abletonLink, midiClock }

/// Lock-free snapshot of audio engine state.
///
/// Published by the audio thread once per buffer (~5.3 ms @ 256 frames /
/// 48 kHz) via a seqlock; Dart polls at 120 Hz.
class Snapshot {
  const Snapshot({
    required this.currentFrame,
    required this.nextDownbeatFrame,
    required this.tempoBPM,
    required this.quantum,
    required this.currentBeat,
    required this.phaseInBar,
    required this.sigNumerator,
    required this.sigDenominator,
    required this.syncSourceKind,
    required this.baseLoopStart,
    required this.baseLoopFrames,
    required this.sampleRate,
    required this.channels,
    required this.bufferSize,
    required this.activeRecordingMask,
    required this.activeVoiceMask,
    required this.captureLevelDb,
    required this.playbackLevelDb,
  });

  /// Empty snapshot; engine hasn't published yet.
  static const Snapshot empty = Snapshot(
    currentFrame: 0,
    nextDownbeatFrame: 0,
    tempoBPM: 0.0,
    quantum: 0,
    currentBeat: 0,
    phaseInBar: 0.0,
    sigNumerator: 4,
    sigDenominator: 4,
    syncSourceKind: SyncSourceKind.local,
    baseLoopStart: 0,
    baseLoopFrames: 0,
    sampleRate: 0,
    channels: 0,
    bufferSize: 0,
    activeRecordingMask: 0,
    activeVoiceMask: 0,
    captureLevelDb: -100.0,
    playbackLevelDb: -100.0,
  );

  // --- Transport (musical time) ---

  /// Monotonically increasing sample counter from the audio thread.
  final int currentFrame;

  /// Global frame of the next downbeat (bar boundary). Meaningful only when
  /// [hasTempo] is true.
  final int nextDownbeatFrame;

  /// Tempo in BPM. 0 means no tempo set (free mode).
  final double tempoBPM;

  /// Beats per loop / bar. 0 means no tempo set.
  final int quantum;

  /// Total beats elapsed since the active sync source's anchor.
  final int currentBeat;

  /// Phase within the current bar, in [0, 1). 0 == downbeat.
  final double phaseInBar;

  final int sigNumerator;
  final int sigDenominator;

  final SyncSourceKind syncSourceKind;

  // --- Legacy base-loop fields (parallel-run during 2 → 2c) ---

  /// Anchor frame of the base loop. Meaningless unless [baseLoopFrames] > 0.
  final int baseLoopStart;

  /// Base loop length in frames (samples per channel). 0 = free mode.
  final int baseLoopFrames;

  // --- Device ---

  final int sampleRate;
  final int channels;
  final int bufferSize;

  // --- Activity ---

  /// Bit i set ⇔ recording slot i is active.
  final int activeRecordingMask;

  /// Bit i set ⇔ voice slot i is active.
  final int activeVoiceMask;

  final double captureLevelDb;
  final double playbackLevelDb;

  bool get hasBaseLoop => baseLoopFrames > 0;
  bool get hasTempo => tempoBPM > 0.0 && quantum > 0;

  @override
  String toString() => 'Snapshot(frame=$currentFrame, '
      'tempo=${tempoBPM.toStringAsFixed(1)}, q=$quantum, '
      'phase=${phaseInBar.toStringAsFixed(3)}, beat=$currentBeat, '
      'sync=${syncSourceKind.name}, '
      'baseLoop=$baseLoopFrames@$baseLoopStart, '
      'sr=$sampleRate, ch=$channels, buf=$bufferSize)';
}

/// Event emitted by the audio thread. Drained in FIFO order.
class Event {
  const Event({
    required this.type,
    required this.id,
    required this.frame,
    required this.framesProcessed,
    required this.soundHash,
    required this.code,
  });

  final EventType type;
  final int id;
  final int frame;
  final int framesProcessed;
  final int soundHash;
  final int code;

  @override
  String toString() => 'Event(${type.name}, id=$id, frame=$frame, '
      'framesProcessed=$framesProcessed, soundHash=$soundHash, code=$code)';
}

enum EventType {
  none,
  recordingStarted,
  recordingStopped,
  playbackStarted,
  playbackEnded,
  baseLoopSet,        // legacy
  baseLoopCleared,    // legacy
  tempoSet,
  tempoCleared,
  syncSourceChanged,
  downbeatFired,
  beatFired,
  tempoInferred,      // worker thread, after recording analysis
  keyInferred,        // worker thread, after recording analysis
  // Phase 2c: tap-to-mute / MIDI Performance. Fired sample-accurately when
  // a queued mute/unmute/gain change reaches its scheduled boundary.
  voiceMuted,
  voiceUnmuted,
  gainChanged,
  // Phase 1 native transport. Fired on the audio thread the moment a queued
  // pause / unpause reaches its launch boundary, alongside the bridged
  // SoLoud setter call. UI consumes for transport state; the audible
  // change is already in flight.
  voicePaused,
  voiceUnpaused,
  error,
}

/// User-driven action posted to the audio thread.
class Command {
  const Command({
    required this.type,
    this.id = 0,
    this.targetFrame = 0,
    this.lengthFrames = 0,
    this.soundHash = 0,
    this.flags = 0,
  });

  final CommandType type;
  final int id;
  final int targetFrame;
  final int lengthFrames;
  final int soundHash;
  final int flags;
}

/// Convert a `KeyInferred` event to a human-readable label like "C", "F#m",
/// or "—" for unknown. Decodes the wire format set by the C++ side.
String keyEventLabel(Event e) {
  if (e.type != EventType.keyInferred) return '—';
  if (e.id == 255) return '—';
  const names = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
  ];
  if (e.id >= names.length) return '—';
  final isMinor = (e.code & 0x1) != 0;
  return isMinor ? '${names[e.id]}m' : names[e.id];
}

/// Unpack the float confidence packed into `soundHash` by the C++ side.
double keyEventConfidence(Event e) {
  if (e.type != EventType.keyInferred) return 0.0;
  final bd = ByteData(4)..setUint32(0, e.soundHash, Endian.host);
  return bd.getFloat32(0, Endian.host);
}

/// Pitch class encoded in the event's `id` field (0–11, or 255 = unknown).
int keyEventPitchClass(Event e) {
  if (e.type != EventType.keyInferred) return 255;
  return e.id;
}

/// Whether the inferred key is minor mode.
bool keyEventIsMinor(Event e) {
  if (e.type != EventType.keyInferred) return false;
  return (e.code & 0x1) != 0;
}

enum CommandType {
  none,
  startRecording,
  stopRecording,
  setBaseLoop,         // legacy, prefer [setTempo]
  clearBaseLoop,       // legacy, prefer [clearTempo]
  startPlayback,
  stopPlayback,
  setLatencyComp,
  setTempo,
  clearTempo,
  setSyncSource,
  setMetronome,
  reportKeyInferred,   // worker → audio (internal)
  // Phase 2c: tap-to-mute / MIDI Performance.
  queueMute,
  queueUnmute,
  setTrackGain,
  cancelPendingQueue,
  // Phase 1 native transport.
  queuePause,
  queueUnpause,
  queueStop,
  registerTrackHandle,
  unregisterTrackHandle,
}

/// Launch quantize value in 1/16 units. Names mirror Ableton's clip-launch
/// dropdown so the vocabulary is familiar.
class LaunchQuantize {
  static const int free    = 0;   // immediate / stab mode
  static const int s16th   = 1;
  static const int s8th    = 2;
  static const int quarter = 4;
  static const int half    = 8;
  static const int bar     = 16;
}

// ---------------------------------------------------------------------------
// FFI signatures
// ---------------------------------------------------------------------------

typedef _LoadSnapshotNative = ffi.Void Function(ffi.Pointer<NativeSnapshot>);
typedef _LoadSnapshotDart = void Function(ffi.Pointer<NativeSnapshot>);

typedef _DrainEventNative = ffi.Bool Function(ffi.Pointer<NativeEvent>);
typedef _DrainEventDart = bool Function(ffi.Pointer<NativeEvent>);

typedef _PostCommandNative = ffi.Bool Function(ffi.Pointer<NativeCommand>);
typedef _PostCommandDart = bool Function(ffi.Pointer<NativeCommand>);

// ---------------------------------------------------------------------------
// EngineBinding
// ---------------------------------------------------------------------------

/// Polling-based bridge to the native AudioEngine.
///
/// Construct once at app startup, dispose at shutdown. Subscribers may come
/// and go; the underlying polling timer runs only when at least one
/// subscriber is attached (broadcast stream semantics).
class EngineBinding {
  EngineBinding._({
    required _LoadSnapshotDart loadSnapshot,
    required _DrainEventDart drainEvent,
    required _PostCommandDart postCommandFn,
    Duration pollInterval = const Duration(microseconds: 8333),
  })  : _loadSnapshotFn = loadSnapshot,
        _drainEventFn = drainEvent,
        _postCommandFn = postCommandFn,
        _pollInterval = pollInterval {
    _snapshotPtr = calloc<NativeSnapshot>();
    _eventPtr = calloc<NativeEvent>();
    _commandPtr = calloc<NativeCommand>();
  }

  /// Builds the binding by resolving symbols from the given [DynamicLibrary].
  /// Caller is responsible for ensuring the library is loaded.
  factory EngineBinding.fromDylib(
    ffi.DynamicLibrary dylib, {
    Duration pollInterval = const Duration(microseconds: 8333),
  }) {
    final loadSnapshot = dylib
        .lookup<ffi.NativeFunction<_LoadSnapshotNative>>(
            'flutter_recorder_engine_loadSnapshot')
        .asFunction<_LoadSnapshotDart>();
    final drainEvent = dylib
        .lookup<ffi.NativeFunction<_DrainEventNative>>(
            'flutter_recorder_engine_drainEvent')
        .asFunction<_DrainEventDart>();
    final postCommand = dylib
        .lookup<ffi.NativeFunction<_PostCommandNative>>(
            'flutter_recorder_engine_postCommand')
        .asFunction<_PostCommandDart>();
    return EngineBinding._(
      loadSnapshot: loadSnapshot,
      drainEvent: drainEvent,
      postCommandFn: postCommand,
      pollInterval: pollInterval,
    );
  }

  /// Convenience factory matching the platform-aware loading used by the
  /// rest of flutter_recorder. Throws on unsupported platforms.
  factory EngineBinding.platformDefault({
    Duration pollInterval = const Duration(microseconds: 8333),
  }) =>
      EngineBinding.fromDylib(_openDylib(), pollInterval: pollInterval);

  static ffi.DynamicLibrary _openDylib() {
    const libName = 'flutter_recorder';
    if (Platform.isMacOS || Platform.isIOS) {
      try {
        return ffi.DynamicLibrary.open('$libName.framework/$libName');
      } catch (_) {
        return ffi.DynamicLibrary.process();
      }
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return ffi.DynamicLibrary.open('lib$libName.so');
    }
    if (Platform.isWindows) {
      return ffi.DynamicLibrary.open('$libName.dll');
    }
    throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
  }

  // Function pointers, looked up once at construction.
  final _LoadSnapshotDart _loadSnapshotFn;
  final _DrainEventDart _drainEventFn;
  final _PostCommandDart _postCommandFn;

  final Duration _pollInterval;

  // Pre-allocated FFI buffers reused on every poll to avoid GC pressure.
  late final ffi.Pointer<NativeSnapshot> _snapshotPtr;
  late final ffi.Pointer<NativeEvent> _eventPtr;
  late final ffi.Pointer<NativeCommand> _commandPtr;

  late final StreamController<Snapshot> _snapshotCtrl =
      StreamController<Snapshot>.broadcast(
    onListen: _ensurePolling,
    onCancel: _maybeStopPolling,
  );
  late final StreamController<Event> _eventCtrl =
      StreamController<Event>.broadcast(
    onListen: _ensurePolling,
    onCancel: _maybeStopPolling,
  );

  Timer? _pollTimer;
  Snapshot _lastSnapshot = Snapshot.empty;

  /// Broadcast stream of engine snapshots, ~120 Hz.
  Stream<Snapshot> get snapshots => _snapshotCtrl.stream;

  /// Broadcast stream of engine events. Phase 1: native emits none.
  Stream<Event> get events => _eventCtrl.stream;

  /// Read the snapshot synchronously without going through the stream.
  Snapshot loadSnapshotSync() {
    _loadSnapshotFn(_snapshotPtr);
    return _readSnapshot(_snapshotPtr.ref);
  }

  /// Send a command to the audio thread. Returns true on success, false if
  /// the inbox is full (very rare; would indicate a bug or extreme burst).
  bool postCommand(Command cmd) {
    _commandPtr.ref
      ..type = _commandTypeToNative(cmd.type)
      ..reserved0_0 = 0
      ..reserved0_1 = 0
      ..reserved0_2 = 0
      ..id = cmd.id
      ..targetFrame = cmd.targetFrame
      ..lengthFrames = cmd.lengthFrames
      ..soundHash = cmd.soundHash
      ..flags = cmd.flags;
    return _postCommandFn(_commandPtr);
  }

  /// Disposes timers, frees FFI buffers, closes streams. The binding is
  /// unusable after this returns.
  Future<void> dispose() async {
    _pollTimer?.cancel();
    _pollTimer = null;
    await _snapshotCtrl.close();
    await _eventCtrl.close();
    calloc.free(_snapshotPtr);
    calloc.free(_eventPtr);
    calloc.free(_commandPtr);
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  void _ensurePolling() {
    if (_pollTimer != null) return;
    if (!_snapshotCtrl.hasListener && !_eventCtrl.hasListener) return;
    _pollTimer = Timer.periodic(_pollInterval, _onPoll);
  }

  void _maybeStopPolling() {
    if (_snapshotCtrl.hasListener || _eventCtrl.hasListener) return;
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  void _onPoll(Timer _) {
    // Drain all queued events before reading the snapshot so subscribers see
    // events strictly before the snapshot that supersedes them.
    if (_eventCtrl.hasListener) {
      // Bounded loop — outbox capacity is 256, so this terminates fast even
      // if native is bursting. The audio thread is the only producer.
      for (int i = 0; i < 256; ++i) {
        if (!_drainEventFn(_eventPtr)) break;
        _eventCtrl.add(_readEvent(_eventPtr.ref));
      }
    }
    if (_snapshotCtrl.hasListener) {
      _loadSnapshotFn(_snapshotPtr);
      final s = _readSnapshot(_snapshotPtr.ref);
      // Suppress duplicate emissions when the audio thread hasn't moved.
      // (Cheap identity check on currentFrame; if engine state changed in
      // other fields without frame advancing, we'd miss it — but the audio
      // thread bumps currentFrame every buffer, so this is safe.)
      if (s.currentFrame != _lastSnapshot.currentFrame ||
          !identical(s, _lastSnapshot)) {
        _lastSnapshot = s;
        _snapshotCtrl.add(s);
      }
    }
  }

  Snapshot _readSnapshot(NativeSnapshot s) => Snapshot(
        currentFrame: s.currentFrame,
        nextDownbeatFrame: s.nextDownbeatFrame,
        tempoBPM: s.tempoBPM,
        quantum: s.quantum,
        currentBeat: s.currentBeat,
        phaseInBar: s.phaseInBar,
        sigNumerator: s.sigNumerator,
        sigDenominator: s.sigDenominator,
        syncSourceKind: _syncSourceKindFromNative(s.syncSourceKind),
        baseLoopStart: s.baseLoopStart,
        baseLoopFrames: s.baseLoopFrames,
        sampleRate: s.sampleRate,
        channels: s.channels,
        bufferSize: s.bufferSize,
        activeRecordingMask: s.activeRecordingMask,
        activeVoiceMask: s.activeVoiceMask,
        captureLevelDb: s.captureLevelDb,
        playbackLevelDb: s.playbackLevelDb,
      );

  static SyncSourceKind _syncSourceKindFromNative(int v) {
    switch (v) {
      case NativeSyncSourceKind.none:
        return SyncSourceKind.none;
      case NativeSyncSourceKind.abletonLink:
        return SyncSourceKind.abletonLink;
      case NativeSyncSourceKind.midiClock:
        return SyncSourceKind.midiClock;
      case NativeSyncSourceKind.local:
      default:
        return SyncSourceKind.local;
    }
  }

  Event _readEvent(NativeEvent e) => Event(
        type: _eventTypeFromNative(e.type),
        id: e.id,
        frame: e.frame,
        framesProcessed: e.framesProcessed,
        soundHash: e.soundHash,
        code: e.code,
      );

  static EventType _eventTypeFromNative(int v) {
    switch (v) {
      case NativeEventType.recordingStarted:
        return EventType.recordingStarted;
      case NativeEventType.recordingStopped:
        return EventType.recordingStopped;
      case NativeEventType.playbackStarted:
        return EventType.playbackStarted;
      case NativeEventType.playbackEnded:
        return EventType.playbackEnded;
      case NativeEventType.baseLoopSet:
        return EventType.baseLoopSet;
      case NativeEventType.baseLoopCleared:
        return EventType.baseLoopCleared;
      case NativeEventType.tempoSet:
        return EventType.tempoSet;
      case NativeEventType.tempoCleared:
        return EventType.tempoCleared;
      case NativeEventType.syncSourceChanged:
        return EventType.syncSourceChanged;
      case NativeEventType.downbeatFired:
        return EventType.downbeatFired;
      case NativeEventType.beatFired:
        return EventType.beatFired;
      case NativeEventType.tempoInferred:
        return EventType.tempoInferred;
      case NativeEventType.keyInferred:
        return EventType.keyInferred;
      case NativeEventType.voiceMuted:
        return EventType.voiceMuted;
      case NativeEventType.voiceUnmuted:
        return EventType.voiceUnmuted;
      case NativeEventType.gainChanged:
        return EventType.gainChanged;
      case NativeEventType.voicePaused:
        return EventType.voicePaused;
      case NativeEventType.voiceUnpaused:
        return EventType.voiceUnpaused;
      case NativeEventType.error:
        return EventType.error;
      default:
        return EventType.none;
    }
  }

  static int _commandTypeToNative(CommandType t) {
    switch (t) {
      case CommandType.none:
        return NativeCommandType.none;
      case CommandType.startRecording:
        return NativeCommandType.startRecording;
      case CommandType.stopRecording:
        return NativeCommandType.stopRecording;
      case CommandType.setBaseLoop:
        return NativeCommandType.setBaseLoop;
      case CommandType.clearBaseLoop:
        return NativeCommandType.clearBaseLoop;
      case CommandType.startPlayback:
        return NativeCommandType.startPlayback;
      case CommandType.stopPlayback:
        return NativeCommandType.stopPlayback;
      case CommandType.setLatencyComp:
        return NativeCommandType.setLatencyComp;
      case CommandType.setTempo:
        return NativeCommandType.setTempo;
      case CommandType.clearTempo:
        return NativeCommandType.clearTempo;
      case CommandType.setSyncSource:
        return NativeCommandType.setSyncSource;
      case CommandType.setMetronome:
        return NativeCommandType.setMetronome;
      case CommandType.reportKeyInferred:
        return NativeCommandType.reportKeyInferred;
      case CommandType.queueMute:
        return NativeCommandType.queueMute;
      case CommandType.queueUnmute:
        return NativeCommandType.queueUnmute;
      case CommandType.setTrackGain:
        return NativeCommandType.setTrackGain;
      case CommandType.cancelPendingQueue:
        return NativeCommandType.cancelPendingQueue;
      case CommandType.queuePause:
        return NativeCommandType.queuePause;
      case CommandType.queueUnpause:
        return NativeCommandType.queueUnpause;
      case CommandType.queueStop:
        return NativeCommandType.queueStop;
      case CommandType.registerTrackHandle:
        return NativeCommandType.registerTrackHandle;
      case CommandType.unregisterTrackHandle:
        return NativeCommandType.unregisterTrackHandle;
    }
  }

  /// Typed helper for `SetTempo`. Encodes the tempo (double) into the
  /// command's `lengthFrames` slot via bit-cast, mirroring the C++ side.
  bool setTempo({
    required double bpm,
    required int quantum,
    required int anchorFrame,
  }) {
    // Bit-cast double → int64 to fit Command's `lengthFrames` slot.
    final tempoBytes = ByteData(8)..setFloat64(0, bpm, Endian.host);
    final asInt64 = tempoBytes.getInt64(0, Endian.host);
    return postCommand(Command(
      type: CommandType.setTempo,
      lengthFrames: asInt64,
      targetFrame: anchorFrame,
      flags: quantum,
    ));
  }

  /// Clear tempo (back to free mode).
  bool clearTempo() {
    return postCommand(const Command(type: CommandType.clearTempo));
  }

  /// Switch the active sync source. Phase 2 only implements [SyncSourceKind.local].
  bool setSyncSource(SyncSourceKind kind) {
    return postCommand(Command(
      type: CommandType.setSyncSource,
      id: _syncSourceKindToNative(kind),
    ));
  }

  /// Enable/disable the metronome. (Phase 3 implements the click voice; in
  /// Phase 2 this only toggles a flag.)
  bool setMetronome({required bool enabled, bool downbeatOnly = false}) {
    int flags = 0;
    if (enabled) flags |= 0x1;
    if (downbeatOnly) flags |= 0x2;
    return postCommand(Command(
      type: CommandType.setMetronome,
      flags: flags,
    ));
  }

  // ── Phase 2c: tap-to-mute helpers ───────────────────────────────────────

  /// Queue a mute on `trackIndex`. Fires sample-accurately at the next launch
  /// boundary defined by `quantize` (a [LaunchQuantize] value). The audio
  /// thread emits a [EventType.voiceMuted] event at the fire frame.
  bool queueMute({
    required int trackIndex,
    int quantize = LaunchQuantize.bar,
  }) {
    return postCommand(Command(
      type: CommandType.queueMute,
      id: trackIndex,
      flags: quantize,
    ));
  }

  /// Queue an unmute on `trackIndex` with `velocity` ∈ [0, 1]. Fires at the
  /// next launch boundary defined by `quantize`. The audio thread emits a
  /// [EventType.voiceUnmuted] event carrying the velocity at the fire frame.
  bool queueUnmute({
    required int trackIndex,
    double velocity = 1.0,
    int quantize = LaunchQuantize.bar,
  }) {
    // Bit-cast float32 → uint32 to fit the command's `soundHash` slot.
    final bd = ByteData(4)..setFloat32(0, velocity, Endian.host);
    final velBits = bd.getUint32(0, Endian.host);
    return postCommand(Command(
      type: CommandType.queueUnmute,
      id: trackIndex,
      soundHash: velBits,
      flags: quantize,
    ));
  }

  /// Queue a gain change on `trackIndex`. `gain` is typically in [0, 1+].
  /// Fires at the next launch boundary defined by `quantize`. The audio
  /// thread emits a [EventType.gainChanged] event at the fire frame.
  bool setTrackGain({
    required int trackIndex,
    required double gain,
    int quantize = LaunchQuantize.free,
  }) {
    final bd = ByteData(4)..setFloat32(0, gain, Endian.host);
    final gainBits = bd.getUint32(0, Endian.host);
    return postCommand(Command(
      type: CommandType.setTrackGain,
      id: trackIndex,
      soundHash: gainBits,
      flags: quantize,
    ));
  }

  /// Remove any pending mute/unmute/gain entry for `trackIndex`. Used to
  /// "unqueue" a tap before its fire boundary arrives. No-op if nothing is
  /// pending.
  bool cancelPendingQueue({required int trackIndex}) {
    return postCommand(Command(
      type: CommandType.cancelPendingQueue,
      id: trackIndex,
    ));
  }

  // ── Phase 1 native transport ─────────────────────────────────────────────
  //
  // The audio thread owns the transport change: when the native scheduler
  // hits the fire frame it calls into SoLoud's lock-free per-voice setter
  // directly (no Dart round-trip). Dart still sees the resulting
  // [EventType.voicePaused] / [voiceUnpaused] / [playbackEnded] for UI
  // bookkeeping, but those events arrive _after_ the audio has already
  // transitioned — they're informational, not load-bearing for timing.
  //
  // [registerTrackHandle] must be called once (per loop player) before any
  // queue* call referencing the same trackIndex. The audio thread looks up
  // the SoLoud handle through the trackIndex→handle table; a missing
  // registration silently no-ops the bridged setter (the queued event still
  // fires so any legacy Dart listener can react).

  /// Announce the SoLoud voice handle backing `trackIndex` so the audio
  /// thread can apply transport changes directly. Call once per player on
  /// `AudioPlayerPlay(paused: true)`; call [unregisterTrackHandle] on
  /// `AudioPlayerStop`.
  bool registerTrackHandle({
    required int trackIndex,
    required int soloudHandle,
  }) {
    return postCommand(Command(
      type: CommandType.registerTrackHandle,
      id: trackIndex,
      soundHash: soloudHandle,
    ));
  }

  bool unregisterTrackHandle({required int trackIndex}) {
    return postCommand(Command(
      type: CommandType.unregisterTrackHandle,
      id: trackIndex,
    ));
  }

  /// Queue a sample-accurate `setPause(true)` on `trackIndex` at the next
  /// `quantize` boundary. Fires [EventType.voicePaused] when applied.
  bool queuePause({
    required int trackIndex,
    int quantize = LaunchQuantize.bar,
  }) {
    return postCommand(Command(
      type: CommandType.queuePause,
      id: trackIndex,
      flags: quantize,
    ));
  }

  /// Queue a sample-accurate `setPause(false)` — the workhorse for
  /// "play this take in sync on the next downbeat". Fires
  /// [EventType.voiceUnpaused] when applied.
  bool queueUnpause({
    required int trackIndex,
    int quantize = LaunchQuantize.bar,
  }) {
    return postCommand(Command(
      type: CommandType.queueUnpause,
      id: trackIndex,
      flags: quantize,
    ));
  }

  /// Queue a sample-accurate `stop(handle)` on `trackIndex`. SoLoud reclaims
  /// the voice; the audio thread also drops the trackIndex from its handle
  /// table. Fires [EventType.playbackEnded] when applied.
  bool queueStop({
    required int trackIndex,
    int quantize = LaunchQuantize.free,
  }) {
    return postCommand(Command(
      type: CommandType.queueStop,
      id: trackIndex,
      flags: quantize,
    ));
  }

  /// Unpack the velocity float that the audio thread packed into
  /// [Event.soundHash] for [EventType.voiceUnmuted].
  static double unmuteVelocity(Event e) {
    if (e.type != EventType.voiceUnmuted) return 0.0;
    final bd = ByteData(4)..setUint32(0, e.soundHash, Endian.host);
    return bd.getFloat32(0, Endian.host);
  }

  /// Unpack the gain float that the audio thread packed into
  /// [Event.soundHash] for [EventType.gainChanged].
  static double gainChangedValue(Event e) {
    if (e.type != EventType.gainChanged) return 0.0;
    final bd = ByteData(4)..setUint32(0, e.soundHash, Endian.host);
    return bd.getFloat32(0, Endian.host);
  }

  static int _syncSourceKindToNative(SyncSourceKind kind) {
    switch (kind) {
      case SyncSourceKind.none:
        return NativeSyncSourceKind.none;
      case SyncSourceKind.local:
        return NativeSyncSourceKind.local;
      case SyncSourceKind.abletonLink:
        return NativeSyncSourceKind.abletonLink;
      case SyncSourceKind.midiClock:
        return NativeSyncSourceKind.midiClock;
    }
  }
}
