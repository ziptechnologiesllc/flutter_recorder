#include "coreaudio_duplex.h"

#include <TargetConditionals.h>

// Single-clock CoreAudio duplex.
//   iOS   : one RemoteIO unit, EnableIO both scopes (built-in mic+speaker are
//           one device → one hardware clock by construction).
//   macOS : one HALOutput (AUHAL) unit bound to a device that exposes BOTH
//           input and output. Built-in Mac mic and speakers are SEPARATE
//           AudioDevices, so we build a PRIVATE aggregate device (output =
//           clock master, input = drift-compensated) and bind one AUHAL to it,
//           giving a single coherent clock. A USB interface already exposes
//           both directions on one device, so the aggregate wraps a single sub.
// Other platforms: no-op stubs → caller falls back to miniaudio.

#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
#define CA_DUPLEX_IMPL 1
#endif

#if defined(CA_DUPLEX_IMPL)

#import <AudioToolbox/AudioToolbox.h>
#if TARGET_OS_IPHONE
#import <AVFoundation/AVFoundation.h>
#else
#import <CoreAudio/CoreAudio.h>
#endif
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

extern void aecLog(const char *fmt, ...);

namespace {

// RemoteIO/AUHAL bus numbering: output (to speaker) = bus 0, input (mic) = bus 1.
constexpr AudioUnitElement kOutputBus = 0;
constexpr AudioUnitElement kInputBus = 1;

AudioUnit g_unit = nullptr;
CADuplexRenderFn g_renderFn = nullptr;
void *g_userData = nullptr;
unsigned int g_channels = 2;
unsigned int g_actualSampleRate = 0;
bool g_running = false;
#if TARGET_OS_OSX
AudioDeviceID g_aggregateDevice = kAudioObjectUnknown; // 0 if none created
#endif

AudioBufferList g_micABL;
float *g_micBuf = nullptr;
unsigned int g_micBufFrames = 0;

// Shared render callback (output bus): pull the mic for THIS cycle off the same
// clock, then hand interleaved mic+speaker to the app — one render cycle, one
// clock, mic↔speaker sample-locked.
OSStatus renderCallback(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags,
                        const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber,
                        UInt32 inNumberFrames, AudioBufferList *ioData) {
  (void)inRefCon;
  (void)inBusNumber;
  (void)ioActionFlags;
  float *speaker = ioData ? static_cast<float *>(ioData->mBuffers[0].mData)
                          : nullptr;
  // One-time diagnostics: confirm the callback fires and the output ABL shape.
  static int s_cbLog = 0;
  if (s_cbLog < 3) {
    ++s_cbLog;
    aecLog("[CADuplex] render#%d frames=%u outBufs=%u ch=%u bytes=%u\n", s_cbLog,
           (unsigned)inNumberFrames, ioData ? ioData->mNumberBuffers : 0,
           ioData ? ioData->mBuffers[0].mNumberChannels : 0,
           ioData ? ioData->mBuffers[0].mDataByteSize : 0);
  }
  if (!speaker || inNumberFrames == 0)
    return noErr;

  bool haveMic = false;
  if (g_micBuf && inNumberFrames <= g_micBufFrames) {
    g_micABL.mNumberBuffers = 1;
    g_micABL.mBuffers[0].mNumberChannels = g_channels;
    g_micABL.mBuffers[0].mDataByteSize =
        inNumberFrames * g_channels * sizeof(float);
    g_micABL.mBuffers[0].mData = g_micBuf;
    AudioUnitRenderActionFlags micFlags = 0;
    OSStatus st = AudioUnitRender(g_unit, &micFlags, inTimeStamp, kInputBus,
                                  inNumberFrames, &g_micABL);
    haveMic = (st == noErr);
  }
  if (!haveMic && g_micBuf)
    std::memset(g_micBuf, 0,
                (size_t)inNumberFrames * g_channels * sizeof(float));

  if (g_renderFn)
    g_renderFn(g_userData, g_micBuf, speaker, inNumberFrames);
  else
    std::memset(speaker, 0, inNumberFrames * g_channels * sizeof(float));

  // Diagnostic (~1/s): peak of the mic we pulled and the speaker we produced
  // AFTER the app filled it. Tells us if the SoLoud mix actually reaches the
  // output buffer (non-zero spk) vs being silent before the speaker.
  static int s_ampLog = 0;
  if ((++s_ampLog % 375) == 0) {
    float micPk = 0.0f, spkPk = 0.0f;
    const unsigned n = inNumberFrames * g_channels;
    for (unsigned i = 0; i < n; ++i) {
      float m = g_micBuf ? fabsf(g_micBuf[i]) : 0.0f;
      float s = fabsf(speaker[i]);
      if (m > micPk) micPk = m;
      if (s > spkPk) spkPk = s;
    }
    aecLog("[CADuplex] amp micPk=%.4f spkPk=%.4f\n", micPk, spkPk);
  }
  return noErr;
}

AudioStreamBasicDescription interleavedFloatASBD(double sampleRate,
                                                 unsigned int channels) {
  AudioStreamBasicDescription asbd;
  std::memset(&asbd, 0, sizeof(asbd));
  asbd.mSampleRate = sampleRate;
  asbd.mFormatID = kAudioFormatLinearPCM;
  // Canonical native-endian packed interleaved float (no NonInterleaved flag →
  // one interleaved buffer in the render ABL).
  asbd.mFormatFlags = kAudioFormatFlagsNativeFloatPacked;
  asbd.mFramesPerPacket = 1;
  asbd.mChannelsPerFrame = channels;
  asbd.mBitsPerChannel = 32;
  asbd.mBytesPerFrame = channels * sizeof(float);
  asbd.mBytesPerPacket = asbd.mBytesPerFrame;
  return asbd;
}

// Common post-create wiring shared by iOS + macOS once g_unit exists and IO is
// enabled / device is bound: formats, render callback, mic scratch, start.
bool finishStart(double rate, unsigned int channels) {
  AudioStreamBasicDescription asbd = interleavedFloatASBD(rate, channels);
  AudioUnitSetProperty(g_unit, kAudioUnitProperty_StreamFormat,
                       kAudioUnitScope_Output, kInputBus, &asbd, sizeof(asbd));
  AudioUnitSetProperty(g_unit, kAudioUnitProperty_StreamFormat,
                       kAudioUnitScope_Input, kOutputBus, &asbd, sizeof(asbd));

  AURenderCallbackStruct cb;
  cb.inputProc = renderCallback;
  cb.inputProcRefCon = nullptr;
  AudioUnitSetProperty(g_unit, kAudioUnitProperty_SetRenderCallback,
                       kAudioUnitScope_Input, kOutputBus, &cb, sizeof(cb));

  g_micBufFrames = 4096;
  g_micBuf = static_cast<float *>(
      std::malloc((size_t)g_micBufFrames * channels * sizeof(float)));
  std::memset(&g_micABL, 0, sizeof(g_micABL));

  if (AudioUnitInitialize(g_unit) != noErr) {
    aecLog("[CADuplex] AudioUnitInitialize failed\n");
    return false;
  }
  if (AudioOutputUnitStart(g_unit) != noErr) {
    aecLog("[CADuplex] AudioOutputUnitStart failed\n");
    return false;
  }
  return true;
}

#if TARGET_OS_IPHONE
bool configureSession(unsigned int sampleRate) {
  NSError *err = nil;
  AVAudioSession *session = [AVAudioSession sharedInstance];
  // Default mode (NOT Measurement): Measurement minimizes output processing and
  // routes/attenuates the speaker so playback goes silent while recording still
  // works (the "tuner works, playback broken" symptom). Default applies NO
  // system echo cancellation either, so it's safe for our own AEC, with normal
  // playback. Default-to-speaker + allow Bluetooth.
  [session setCategory:AVAudioSessionCategoryPlayAndRecord
                  mode:AVAudioSessionModeDefault
               options:AVAudioSessionCategoryOptionDefaultToSpeaker |
                       AVAudioSessionCategoryOptionAllowBluetoothA2DP
                 error:&err];
  err = nil;
  [session setPreferredSampleRate:(double)sampleRate error:&err];
  err = nil;
  [session setPreferredIOBufferDuration:128.0 / (double)sampleRate error:&err];
  err = nil;
  [session setActive:YES error:&err];
  if (err) {
    aecLog("[CADuplex] setActive failed: %s\n",
           err.localizedDescription.UTF8String);
    return false;
  }
  return true;
}

bool startIOS(unsigned int sampleRate, unsigned int channels) {
  if (!configureSession(sampleRate))
    return false;
  g_actualSampleRate =
      (unsigned int)([AVAudioSession sharedInstance].sampleRate + 0.5);

  AudioComponentDescription desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.componentType = kAudioUnitType_Output;
  desc.componentSubType = kAudioUnitSubType_RemoteIO;
  desc.componentManufacturer = kAudioUnitManufacturer_Apple;
  AudioComponent comp = AudioComponentFindNext(nullptr, &desc);
  if (!comp || AudioComponentInstanceNew(comp, &g_unit) != noErr || !g_unit) {
    aecLog("[CADuplex] failed to create RemoteIO unit\n");
    return false;
  }
  UInt32 one = 1;
  AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_EnableIO,
                       kAudioUnitScope_Input, kInputBus, &one, sizeof(one));
  AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_EnableIO,
                       kAudioUnitScope_Output, kOutputBus, &one, sizeof(one));
  return finishStart((double)g_actualSampleRate, channels);
}

// ---------------------------------------------------------------------------
// AVAudioSession lifecycle (interruptions + route changes).
//
// configureSession() above is the app's ONE AVAudioSession owner, which means
// it must also own recovery: without these observers nothing re-activates the
// session or restarts the unit after a phone call / Siri / route change, and
// the duplex stays permanently silent. All handling runs on the MAIN queue via
// NSNotificationCenter block observers — the render callback is never touched.
// ---------------------------------------------------------------------------
#if !__has_feature(objc_arc)
#error "coreaudio_duplex.mm relies on ARC for the observer tokens on iOS"
#endif

id g_interruptionObserver = nil; // NSNotificationCenter block-observer tokens
id g_routeChangeObserver = nil;
std::atomic<bool> g_suspended{false};   // interruption began → unit stopped
std::atomic<bool> g_configuring{false}; // inside our own (re)configure window
unsigned int g_requestedSampleRate = 0; // caDuplexStart arg, kept for rebuilds

// Route changes triggered by our OWN setCategory/setActive are posted
// asynchronously and land on the main queue AFTER the code that caused them
// returns, so the guard must outlive the configure call itself.
void endConfiguringSoon() {
  dispatch_after(
      dispatch_time(DISPATCH_TIME_NOW, (int64_t)(500 * NSEC_PER_MSEC)),
      dispatch_get_main_queue(), ^{ g_configuring = false; });
}

// Full teardown + rebuild through the exact same public stop/start paths used
// everywhere else, so the RemoteIO ASBD is re-derived from the freshly
// negotiated session sample rate. Main queue only.
bool rebuildDuplex(const char *why) {
  const unsigned int rate = g_requestedSampleRate;
  const unsigned int channels = g_channels;
  void *userData = g_userData;
  CADuplexRenderFn renderFn = g_renderFn;
  aecLog("[CADuplex] %s: full unit rebuild (requested %u Hz)\n", why, rate);
  caDuplexStop();
  const bool ok = caDuplexStart(rate, channels, userData, renderFn);
  aecLog("[CADuplex] %s: rebuild %s (actual %u Hz)\n", why,
         ok ? "OK" : "FAILED — duplex stopped, engine restart required",
         g_actualSampleRate);
  return ok;
}

// Shared recovery for interruption-ended and actionable route changes:
// re-own the session (configureSession → setActive:YES), re-read the
// negotiated hardware rate, then either just restart the unit (rate unchanged)
// or rebuild it so the ASBD matches the new rate.
void reactivateAndRestart(const char *why) {
  if (!g_unit) {
    aecLog("[CADuplex] %s: no unit, nothing to restart\n", why);
    return;
  }
  g_configuring = true; // configureSession below fires its own route change
  if (!configureSession(g_requestedSampleRate)) {
    g_configuring = false;
    aecLog("[CADuplex] %s: session reactivation FAILED, staying suspended\n",
           why);
    return; // stay down; the next interruption/route event retries
  }
  const unsigned int sessionRate =
      (unsigned int)([AVAudioSession sharedInstance].sampleRate + 0.5);
  if (sessionRate == g_actualSampleRate) {
    const OSStatus st = AudioOutputUnitStart(g_unit);
    aecLog("[CADuplex] %s: rate unchanged (%u Hz), unit restart %s (%d)\n",
           why, sessionRate, st == noErr ? "OK" : "FAILED", (int)st);
    if (st == noErr) {
      g_suspended = false;
      endConfiguringSoon();
    } else {
      // A unit that refuses to start is dead anyway — rebuilding through the
      // full stop/start path is the only remaining recovery.
      if (rebuildDuplex(why)) // stop/start manage g_configuring themselves
        g_suspended = false;
    }
  } else {
    aecLog("[CADuplex] %s: session rate %u != running %u\n", why, sessionRate,
           g_actualSampleRate);
    if (rebuildDuplex(why)) // stop/start manage g_configuring themselves
      g_suspended = false;
  }
}

void handleInterruption(NSNotification *note) {
  NSNumber *typeNum = note.userInfo[AVAudioSessionInterruptionTypeKey];
  if (!typeNum)
    return;
  const NSUInteger type = typeNum.unsignedIntegerValue;
  if (type == AVAudioSessionInterruptionTypeBegan) {
    aecLog("[CADuplex] interruption BEGAN -> stopping unit\n");
    if (g_unit)
      AudioOutputUnitStop(g_unit);
    g_suspended = true;
  } else if (type == AVAudioSessionInterruptionTypeEnded) {
    NSNumber *optNum = note.userInfo[AVAudioSessionInterruptionOptionKey];
    const bool shouldResume =
        optNum && (optNum.unsignedIntegerValue &
                   AVAudioSessionInterruptionOptionShouldResume) != 0;
    // Attempt resume even without ShouldResume: this is the app's only audio
    // engine, and if iOS still objects setActive:YES fails and we stay
    // suspended until the next event.
    aecLog("[CADuplex] interruption ENDED (shouldResume=%d) -> reactivating\n",
           shouldResume ? 1 : 0);
    reactivateAndRestart("interruption-ended");
  }
}

void handleRouteChange(NSNotification *note) {
  NSNumber *reasonNum = note.userInfo[AVAudioSessionRouteChangeReasonKey];
  const NSUInteger reason =
      reasonNum ? reasonNum.unsignedIntegerValue
                : AVAudioSessionRouteChangeReasonUnknown;
  if (g_configuring) {
    aecLog("[CADuplex] route change (reason=%lu) ignored: self-inflicted "
           "during configure\n",
           (unsigned long)reason);
    return;
  }
  if (g_suspended) {
    aecLog("[CADuplex] route change (reason=%lu) deferred: interrupted, "
           "waiting for interruption end\n",
           (unsigned long)reason);
    return;
  }
  switch (reason) {
  case AVAudioSessionRouteChangeReasonOldDeviceUnavailable:
  case AVAudioSessionRouteChangeReasonNewDeviceAvailable:
  case AVAudioSessionRouteChangeReasonCategoryChange:
    aecLog("[CADuplex] route change (reason=%lu) -> rate check\n",
           (unsigned long)reason);
    reactivateAndRestart("route-change");
    break;
  default:
    aecLog("[CADuplex] route change (reason=%lu) ignored\n",
           (unsigned long)reason);
    break;
  }
}

void registerSessionObservers() {
  if (g_interruptionObserver || g_routeChangeObserver)
    return;
  NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
  AVAudioSession *session = [AVAudioSession sharedInstance];
  g_interruptionObserver =
      [nc addObserverForName:AVAudioSessionInterruptionNotification
                      object:session
                       queue:[NSOperationQueue mainQueue]
                  usingBlock:^(NSNotification *note) {
                    handleInterruption(note);
                  }];
  g_routeChangeObserver =
      [nc addObserverForName:AVAudioSessionRouteChangeNotification
                      object:session
                       queue:[NSOperationQueue mainQueue]
                  usingBlock:^(NSNotification *note) {
                    handleRouteChange(note);
                  }];
  aecLog("[CADuplex] session lifecycle observers registered\n");
}

void unregisterSessionObservers() {
  NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
  if (g_interruptionObserver) {
    [nc removeObserver:g_interruptionObserver];
    g_interruptionObserver = nil;
  }
  if (g_routeChangeObserver) {
    [nc removeObserver:g_routeChangeObserver];
    g_routeChangeObserver = nil;
  }
}
#endif // TARGET_OS_IPHONE

#if TARGET_OS_OSX
AudioDeviceID defaultDevice(bool input) {
  AudioDeviceID dev = kAudioObjectUnknown;
  UInt32 sz = sizeof(dev);
  AudioObjectPropertyAddress addr = {
      input ? kAudioHardwarePropertyDefaultInputDevice
            : kAudioHardwarePropertyDefaultOutputDevice,
      kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
  AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &sz,
                             &dev);
  return dev;
}

CFStringRef deviceUID(AudioDeviceID dev) {
  CFStringRef uid = nullptr;
  UInt32 sz = sizeof(uid);
  AudioObjectPropertyAddress addr = {kAudioDevicePropertyDeviceUID,
                                     kAudioObjectPropertyScopeGlobal,
                                     kAudioObjectPropertyElementMain};
  AudioObjectGetPropertyData(dev, &addr, 0, nullptr, &sz, &uid);
  return uid; // caller releases
}

// Build a PRIVATE aggregate device: output = clock master, input subdevice
// drift-compensated onto the master clock. Returns kAudioObjectUnknown on fail.
AudioDeviceID buildAggregate(AudioDeviceID inDev, AudioDeviceID outDev) {
  CFStringRef inUID = deviceUID(inDev);
  CFStringRef outUID = deviceUID(outDev);
  if (!inUID || !outUID) {
    if (inUID) CFRelease(inUID);
    if (outUID) CFRelease(outUID);
    return kAudioObjectUnknown;
  }

  // Sub-device list: output first (master), input second (drift-compensated).
  const void *outSubKeys[] = {CFSTR(kAudioSubDeviceUIDKey)};
  const void *outSubVals[] = {outUID};
  CFDictionaryRef outSub = CFDictionaryCreate(
      nullptr, outSubKeys, outSubVals, 1, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  SInt32 driftOn = 1;
  CFNumberRef drift = CFNumberCreate(nullptr, kCFNumberSInt32Type, &driftOn);
  const void *inSubKeys[] = {CFSTR(kAudioSubDeviceUIDKey),
                             CFSTR(kAudioSubDeviceDriftCompensationKey)};
  const void *inSubVals[] = {inUID, drift};
  CFDictionaryRef inSub = CFDictionaryCreate(
      nullptr, inSubKeys, inSubVals, 2, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  const void *subs[] = {outSub, inSub};
  CFArrayRef subList =
      CFArrayCreate(nullptr, subs, 2, &kCFTypeArrayCallBacks);

  SInt32 isPrivate = 1;
  CFNumberRef priv = CFNumberCreate(nullptr, kCFNumberSInt32Type, &isPrivate);
  CFStringRef aggUID = CFSTR("io.ziptech.cloudloop.aggregate");
  const void *aggKeys[] = {CFSTR(kAudioAggregateDeviceUIDKey),
                           CFSTR(kAudioAggregateDeviceNameKey),
                           CFSTR(kAudioAggregateDeviceSubDeviceListKey),
                           CFSTR(kAudioAggregateDeviceMasterSubDeviceKey),
                           CFSTR(kAudioAggregateDeviceIsPrivateKey)};
  const void *aggVals[] = {aggUID, CFSTR("CloudLoop Duplex"), subList, outUID,
                           priv};
  CFDictionaryRef desc = CFDictionaryCreate(
      nullptr, aggKeys, aggVals, 5, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);

  AudioDeviceID agg = kAudioObjectUnknown;
  OSStatus st = AudioHardwareCreateAggregateDevice(desc, &agg);

  CFRelease(desc); CFRelease(priv); CFRelease(subList);
  CFRelease(inSub); CFRelease(outSub); CFRelease(drift);
  CFRelease(inUID); CFRelease(outUID);
  if (st != noErr) {
    aecLog("[CADuplex] aggregate create failed: %d\n", (int)st);
    return kAudioObjectUnknown;
  }
  return agg;
}

bool startMacOS(unsigned int sampleRate, unsigned int channels) {
  AudioDeviceID inDev = defaultDevice(true);
  AudioDeviceID outDev = defaultDevice(false);
  if (inDev == kAudioObjectUnknown || outDev == kAudioObjectUnknown) {
    aecLog("[CADuplex] no default in/out device\n");
    return false;
  }

  AudioDeviceID bindDev;
  if (inDev == outDev) {
    bindDev = inDev; // one device already does both directions
  } else {
    g_aggregateDevice = buildAggregate(inDev, outDev);
    if (g_aggregateDevice == kAudioObjectUnknown)
      return false; // caller falls back to miniaudio
    bindDev = g_aggregateDevice;
  }

  AudioComponentDescription desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.componentType = kAudioUnitType_Output;
  desc.componentSubType = kAudioUnitSubType_HALOutput;
  desc.componentManufacturer = kAudioUnitManufacturer_Apple;
  AudioComponent comp = AudioComponentFindNext(nullptr, &desc);
  if (!comp || AudioComponentInstanceNew(comp, &g_unit) != noErr || !g_unit) {
    aecLog("[CADuplex] failed to create HALOutput unit\n");
    return false;
  }
  // EnableIO BEFORE binding the device (TN2091).
  UInt32 one = 1, zero = 0;
  AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_EnableIO,
                       kAudioUnitScope_Input, kInputBus, &one, sizeof(one));
  AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_EnableIO,
                       kAudioUnitScope_Output, kOutputBus, &one, sizeof(one));
  (void)zero;
  AudioUnitSetProperty(g_unit, kAudioOutputUnitProperty_CurrentDevice,
                       kAudioUnitScope_Global, 0, &bindDev, sizeof(bindDev));
  g_actualSampleRate = sampleRate; // request; device may negotiate
  return finishStart((double)sampleRate, channels);
}
#endif // TARGET_OS_OSX

} // namespace

extern "C" {

bool caDuplexStart(unsigned int sampleRate, unsigned int channels,
                   void *userData, CADuplexRenderFn renderFn) {
  if (g_running)
    return true;
  g_renderFn = renderFn;
  g_userData = userData;
  g_channels = channels;

  bool ok = false;
#if TARGET_OS_IPHONE
  g_requestedSampleRate = sampleRate;
  g_configuring = true; // ignore route changes our own startup provokes
  ok = startIOS(sampleRate, channels);
#elif TARGET_OS_OSX
  ok = startMacOS(sampleRate, channels);
#endif
  if (!ok) {
    caDuplexStop();
#if TARGET_OS_IPHONE
    g_configuring = false;
#endif
    return false;
  }
  g_running = true;
#if TARGET_OS_IPHONE
  // AudioOutputUnitStart succeeded (finishStart) — arm the main-queue session
  // lifecycle observers, and keep the self-route-change guard up briefly since
  // our own configureSession's route change lands asynchronously.
  registerSessionObservers();
  endConfiguringSoon();
#endif
  aecLog("[CADuplex] STARTED single-clock duplex @ %u Hz, %u ch\n",
         g_actualSampleRate, channels);
  return true;
}

void caDuplexStop(void) {
#if TARGET_OS_IPHONE
  unregisterSessionObservers();
  g_suspended = false;
#endif
  if (g_unit) {
    AudioOutputUnitStop(g_unit);
    AudioUnitUninitialize(g_unit);
    AudioComponentInstanceDispose(g_unit);
    g_unit = nullptr;
  }
#if TARGET_OS_OSX
  if (g_aggregateDevice != kAudioObjectUnknown) {
    AudioHardwareDestroyAggregateDevice(g_aggregateDevice);
    g_aggregateDevice = kAudioObjectUnknown;
  }
#endif
  if (g_micBuf) {
    std::free(g_micBuf);
    g_micBuf = nullptr;
  }
  g_running = false;
}

bool caDuplexIsRunning(void) { return g_running; }
unsigned int caDuplexActualSampleRate(void) { return g_actualSampleRate; }

} // extern "C"

#else // unsupported platform → no-op stubs (caller falls back to miniaudio)

extern "C" {
bool caDuplexStart(unsigned int, unsigned int, void *, CADuplexRenderFn) {
  return false;
}
void caDuplexStop(void) {}
bool caDuplexIsRunning(void) { return false; }
unsigned int caDuplexActualSampleRate(void) { return 0; }
}

#endif
