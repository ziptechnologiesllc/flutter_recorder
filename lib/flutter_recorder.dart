/// Flutter low level audio recorder plugin using miniaudio library and FFI
library flutter_recorder;

export 'src/audio_data_container.dart';
export 'src/audio_engine/engine_binding.dart'
    show
        Snapshot,
        Event,
        EventType,
        Command,
        CommandType,
        EngineBinding,
        LaunchQuantize,
        SyncSourceKind,
        keyEventLabel,
        keyEventConfidence,
        keyEventPitchClass,
        keyEventIsMinor;
export 'src/audio_engine/pitch_class_colors.dart'
    show
        kPitchClassColors,
        kPitchClassNames,
        colorForPitchClass,
        namePitchClass,
        formatKey;
export 'src/bindings/recorder.dart' show CallbackStats;
export 'src/enums.dart'
    show
        AndroidInputPreset,
        CalibrationSignalType,
        CaptureDevice,
        CaptureErrors,
        LooperPlaybackStartedEvent,
        PCMFormat,
        RecorderChannels,
        RecordingStartedEvent,
        RecordingStoppedEvent;
export 'src/filters/autogain.dart';
export 'src/filters/echo_cancellation.dart';
export 'src/filters/filters.dart' show RecorderFilterType;
export 'src/flutter_recorder.dart';
