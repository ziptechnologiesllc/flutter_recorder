// Relative import to be able to reuse the C sources.
// See the comment in ../flutter_recorder.podspec for more information.
#include "../../src/flutter_recorder.cpp"

#include "../../src/analyzer.cpp"
#include "../../src/audio_engine/ableton_link_clock.cpp"
#include "../../src/audio_engine/audio_engine.cpp"
#include "../../src/audio_engine/inference.cpp"
#include "../../src/audio_engine/local_clock.cpp"
#include "../../src/audio_engine/metronome_voice.cpp"
#include "../../src/capture.cpp"
#include "../../src/fft/soloud_fft.cpp"
#include "../../src/filters/aec/adaptive_echo_cancellation.cpp"
#include "../../src/filters/aec/calibration.cpp"
#include "../../src/filters/aec/reference_buffer.cpp"
#include "../../src/filters/aec/aec_test.cpp"
#include "../../src/filters/aec/neural_post_filter.cpp"
#include "../../src/filters/aec/delay_estimator.cpp"
#include "../../src/filters/aec/drift_aligner.cpp"
#include "../../src/filters/aec/spectral_governor.cpp"
#include "../../src/filters/aec/synchronous_echo_template.cpp"
#include "../../src/filters/aec/vss_nlms_filter.cpp"
#include "../../src/coreaudio_duplex.mm"
#include "../../src/filters/autogain.cpp"
#include "../../src/filters/echo_cancellation.cpp"
#include "../../src/filters/filters.cpp"
#include "../../src/native_ring_buffer.cpp"
#include "../../src/native_scheduler.cpp"
#include "../../src/auto_record.cpp"
#include "../../src/soloud_slave_bridge.cpp"
#include "../../src/stubs/signpost_profiler_stub.cpp"

#import <AVFoundation/AVFoundation.h>

extern "C" {
void flutter_recorder_ios_force_speaker_output(bool enabled) {
  NSError *error = nil;
  AVAudioSession *session = [AVAudioSession sharedInstance];
  if (enabled) {
    [session overrideOutputAudioPort:AVAudioSessionPortOverrideSpeaker
                               error:&error];
    NSLog(@"[FlutterRecorder] Forced speaker output: YES");
  } else {
    [session overrideOutputAudioPort:AVAudioSessionPortOverrideNone
                               error:&error];
    NSLog(@"[FlutterRecorder] Forced speaker output: NO");
  }
  if (error) {
    NSLog(@"[FlutterRecorder] Error setting speaker output: %@", error);
  }
}
}
