/// Flutter low level audio recorder plugin using miniaudio library and FFI
library flutter_recorder;

export 'src/audio_data_container.dart';
export 'src/enums.dart'
    show PCMFormat, RecorderChannels, CaptureDevice, CaptureErrors, CalibrationSignalType, RecordingStoppedEvent;
export 'src/filters/autogain.dart';
export 'src/filters/echo_cancellation.dart';
export 'src/filters/filters.dart' show RecorderFilterType;
export 'src/flutter_recorder.dart';
