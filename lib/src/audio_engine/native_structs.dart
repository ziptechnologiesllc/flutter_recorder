// FFI struct mirrors of audio_engine/{snapshot,command,event}.h.
//
// IMPORTANT: field order, types, and padding bytes here must match the C++
// definitions exactly. Both sides have static_asserts (C++) / known sizes
// (Dart) so a layout drift will surface at load time, not at runtime.
//
// All three structs are POD on the C++ side and trivially copyable. Dart
// allocates them, hands a pointer to native, and native writes into them.

import 'dart:ffi' as ffi;

import 'package:meta/meta.dart';

// ---------------------------------------------------------------------------
// Snapshot (matches Snapshot in C++). Sized <= 64 bytes (one cache line).
// ---------------------------------------------------------------------------

@internal
final class NativeSnapshot extends ffi.Struct {
  // Transport — must match the C++ Snapshot field order exactly.
  @ffi.Int64()
  external int currentFrame;

  @ffi.Int64()
  external int nextDownbeatFrame;

  @ffi.Double()
  external double tempoBPM;

  @ffi.Uint32()
  external int quantum;

  @ffi.Uint32()
  external int currentBeat;

  @ffi.Float()
  external double phaseInBar;

  @ffi.Uint8()
  external int sigNumerator;

  @ffi.Uint8()
  external int sigDenominator;

  @ffi.Uint8()
  external int syncSourceKind;

  @ffi.Uint8()
  external int reserved0;

  // Legacy base-loop fields (parallel-run during 2 → 2c).
  @ffi.Int64()
  external int baseLoopStart;

  @ffi.Int64()
  external int baseLoopFrames;

  @ffi.Uint32()
  external int sampleRate;

  @ffi.Uint16()
  external int channels;

  @ffi.Uint16()
  external int bufferSize;

  @ffi.Uint8()
  external int activeRecordingMask;

  @ffi.Uint8()
  external int reserved1;

  @ffi.Uint16()
  external int activeVoiceMask;

  @ffi.Float()
  external double captureLevelDb;

  @ffi.Float()
  external double playbackLevelDb;

  @ffi.Uint32()
  external int reserved2;
}

/// Mirrors C++ `SyncSourceKind`.
class NativeSyncSourceKind {
  static const int none = 0;
  static const int local = 1;
  static const int abletonLink = 2;
  static const int midiClock = 3;
}

// ---------------------------------------------------------------------------
// Command (matches Command in C++). Exactly 32 bytes.
// ---------------------------------------------------------------------------

/// Numeric constants for `Command.type`. Must match the C++ `Command::Type`
/// enum order in audio_engine/command.h.
@internal
class NativeCommandType {
  static const int none = 0;
  static const int startRecording = 1;
  static const int stopRecording = 2;
  static const int setBaseLoop = 3;       // legacy, prefer setTempo
  static const int clearBaseLoop = 4;     // legacy, prefer clearTempo
  static const int startPlayback = 5;
  static const int stopPlayback = 6;
  static const int setLatencyComp = 7;
  static const int setTempo = 8;
  static const int clearTempo = 9;
  static const int setSyncSource = 10;
  static const int setMetronome = 11;
}

@internal
final class NativeCommand extends ffi.Struct {
  @ffi.Uint8()
  external int type;

  @ffi.Uint8()
  external int reserved0_0;
  @ffi.Uint8()
  external int reserved0_1;
  @ffi.Uint8()
  external int reserved0_2;

  @ffi.Uint32()
  external int id;

  @ffi.Int64()
  external int targetFrame;

  @ffi.Int64()
  external int lengthFrames;

  @ffi.Uint32()
  external int soundHash;

  @ffi.Uint32()
  external int flags;
}

// ---------------------------------------------------------------------------
// Event (matches Event in C++). Exactly 32 bytes.
// ---------------------------------------------------------------------------

/// Numeric constants for `Event.type`. Must match the C++ `Event::Type` enum
/// order in audio_engine/event.h.
@internal
class NativeEventType {
  static const int none = 0;
  static const int recordingStarted = 1;
  static const int recordingStopped = 2;
  static const int playbackStarted = 3;
  static const int playbackEnded = 4;
  static const int baseLoopSet = 5;
  static const int baseLoopCleared = 6;
  static const int tempoSet = 7;
  static const int tempoCleared = 8;
  static const int syncSourceChanged = 9;
  static const int downbeatFired = 10;
  static const int beatFired = 11;
  static const int tempoInferred = 12;
  static const int keyInferred = 13;
  static const int error = 14;
}

@internal
final class NativeEvent extends ffi.Struct {
  @ffi.Uint8()
  external int type;

  @ffi.Uint8()
  external int reserved0_0;
  @ffi.Uint8()
  external int reserved0_1;
  @ffi.Uint8()
  external int reserved0_2;

  @ffi.Uint32()
  external int id;

  @ffi.Int64()
  external int frame;

  @ffi.Int64()
  external int framesProcessed;

  @ffi.Uint32()
  external int soundHash;

  @ffi.Uint32()
  external int code;
}
