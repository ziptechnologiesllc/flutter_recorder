import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_recorder/flutter_recorder.dart';

/// Feed-forward AEC pseudo-track for the metronome (M2 of the convergence
/// work).
///
/// The engine's click is fully deterministic — a Hann-windowed sine
/// (downbeat 1500 Hz / 0.55, beat 800 Hz / 0.40, 30 ms; MUST stay in
/// lockstep with `MetronomeVoice::generateClickInto` in
/// src/audio_engine/metronome_voice.cpp) mixed at transport-computed frames
/// that Dart already receives as beat/downbeat events. So instead of the
/// linear canceller LEARNING the click's echo (slow, and its transient trips
/// the E3 freeze — the "metronome ghost"), we SYNTHESIZE one loop period of
/// the click pattern from observed beat frames and register it as an
/// ordinary per-track contribution: the template then carries the exact
/// click echo, toggled on/off with the metronome like any muted/unmuted
/// track. The HF suppressor only has to clean what physics leaves behind.
///
/// Collection strategy: while armed, record (frame, isDownbeat) from the
/// event stream until one full loop period is covered, verify the beat
/// spacing divides the period (tempo must be loop-locked, which CloudLoop's
/// bar-locked loops guarantee by construction — if not, registration is
/// skipped and behavior falls back to the reseed path), then place clicks at
/// each beat's loop phase (tails wrap mod P) and register. Re-collects on
/// base-loop / tempo / sync-source changes and on every re-enable.
class MetronomeAecTrack {
  /// Reserved trackIndex for the metronome contribution. Dart trackIndexes
  /// are `<hash>.hashCode & 0x7FFFFFFF`; a fixed high constant makes an
  /// accidental collision astronomically unlikely (and harmless: the slot
  /// table keys exact values).
  static const int kTrackIndex = 0x7FFFFFF0;

  static const double _clickDurationSec = 0.030;
  static const double _downbeatFreqHz = 1500.0;
  static const double _downbeatAmp = 0.55;
  static const double _beatFreqHz = 800.0;
  static const double _beatAmp = 0.40;

  /// Beat spacing must divide the loop period within this tolerance for the
  /// pattern to actually repeat every loop pass.
  static const int _lockToleranceFrames = 3;

  bool _armed = false;
  bool _registered = false;
  bool _active = false;
  int _period = 0;
  int _loopStart = 0;
  int _sampleRate = 0;
  final List<int> _beatFrames = [];
  final List<bool> _beatIsDownbeat = [];

  /// True while the binding's poll loop should keep draining events for us
  /// even with no external listeners.
  bool get wantsEvents => _armed;

  /// Metronome enabled: start (re)collecting a fresh period.
  void arm() {
    _armed = true;
    _reset();
  }

  /// Metronome disabled: remove the contribution from the live template.
  void disable() {
    _armed = false;
    _reset();
    if (_active) {
      Recorder.instance.aecSetTrackActive(kTrackIndex, false);
      _active = false;
    }
  }

  void _reset() {
    _beatFrames.clear();
    _beatIsDownbeat.clear();
    _registered = false;
  }

  /// Feed every drained engine event here (with the freshest snapshot).
  void onEvent(Event event, Snapshot snap) {
    if (!_armed) return;

    switch (event.type) {
      case EventType.baseLoopSet:
      case EventType.baseLoopCleared:
      case EventType.tempoSet:
      case EventType.tempoCleared:
      case EventType.syncSourceChanged:
        // The pattern we collected (or registered) no longer matches the
        // audible one — recollect from scratch. Deactivate a stale live
        // contribution rather than leave it subtracting the wrong phases.
        if (_active) {
          Recorder.instance.aecSetTrackActive(kTrackIndex, false);
          _active = false;
        }
        _reset();
        return;
      case EventType.beatFired:
      case EventType.downbeatFired:
        break;
      default:
        return;
    }

    if (_registered) {
      // Activation retry: aecSetTrackActive is a no-op until the worker has
      // finished computing the contribution, and the FFI is void (we can't
      // see whether it stuck) — so we re-send on every beat. Once applied,
      // the native side short-circuits on active==target, so this settles
      // into a cheap slot-scan no-op.
      if (_active) {
        Recorder.instance.aecSetTrackActive(kTrackIndex, true);
      }
      // Watch for a silent period move (missed event edge cases).
      if (snap.baseLoopFrames != _period || snap.baseLoopStart != _loopStart) {
        _reset();
      }
      return;
    }

    final period = snap.baseLoopFrames;
    if (period <= 0 || snap.sampleRate <= 0) return; // no grid yet

    if (_beatFrames.isEmpty ||
        period != _period ||
        snap.baseLoopStart != _loopStart) {
      _beatFrames.clear();
      _beatIsDownbeat.clear();
      _period = period;
      _loopStart = snap.baseLoopStart;
      _sampleRate = snap.sampleRate;
    }
    _beatFrames.add(event.frame);
    _beatIsDownbeat.add(event.type == EventType.downbeatFired);

    if (_beatFrames.last - _beatFrames.first >= _period) {
      _registerCollectedPeriod();
    }
  }

  void _registerCollectedPeriod() {
    // Beats inside exactly one period window.
    final first = _beatFrames.first;
    final frames = <int>[];
    final downs = <bool>[];
    for (int i = 0; i < _beatFrames.length; ++i) {
      if (_beatFrames[i] - first < _period) {
        frames.add(_beatFrames[i]);
        downs.add(_beatIsDownbeat[i]);
      }
    }
    if (frames.isEmpty) {
      _registered = true; // nothing audible this period; nothing to model
      return;
    }

    // Loop-locked check: observed spacing must divide the period, or the
    // clicks drift through loop phase and a static contribution would be
    // WRONG every later pass. Fall back to the reseed path by not
    // registering (until the next invalidation event re-arms collection).
    if (frames.length >= 2) {
      final spacing = frames[1] - frames[0];
      if (spacing > 0) {
        final r = _period % spacing;
        if (r > _lockToleranceFrames &&
            spacing - r > _lockToleranceFrames) {
          _registered = true; // give up quietly for this grid
          return;
        }
      }
    }

    final mono = Float32List(_period);
    for (int i = 0; i < frames.length; ++i) {
      final phase =
          ((frames[i] - _loopStart) % _period + _period) % _period;
      _addClick(mono, phase, downs[i]);
    }

    Recorder.instance.aecRegisterTrackAudio(kTrackIndex, mono);
    // Activation is retried on every subsequent beat (the contribution
    // computes off-thread in ~tens of ms); _active marks the INTENT so
    // disable() knows to send the deactivation.
    _registered = true;
    _active = true;
    Recorder.instance.aecSetTrackActive(kTrackIndex, true);
  }

  /// Mirror of MetronomeVoice::generateClickInto — keep in lockstep.
  void _addClick(Float32List dst, int phase, bool isDownbeat) {
    final clickFrames = (_clickDurationSec * _sampleRate).floor();
    if (clickFrames < 2) return;
    final freq = isDownbeat ? _downbeatFreqHz : _beatFreqHz;
    final amp = isDownbeat ? _downbeatAmp : _beatAmp;
    final dt = 1.0 / _sampleRate;
    for (int i = 0; i < clickFrames; ++i) {
      final t = i * dt;
      final s = math.sin(2.0 * math.pi * freq * t);
      final w = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (clickFrames - 1)));
      dst[(phase + i) % _period] += amp * s * w;
    }
  }
}
