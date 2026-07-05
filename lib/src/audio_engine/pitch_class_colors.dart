// Pitch-class → color mapping for visual key / chord rendering.
//
// Adopted from Nicolas Melendez and Clint Goss's "Color of Sound" chart
// (Flutopedia), which maps each of the 12 pitch classes to a hue along a
// continuous spectrum derived from the octave-jumped visual wavelength of
// each pitch. The result is musically and perceptually pleasing: warm
// reds/oranges for the F–A bass-rich region and cool greens/blues/violets
// for the C–E treble-rich region.
//
// Use sites:
//   - Status-bar key label (tint)
//   - Snapshot badge (tint)
//   - Chord-progression overlay on the loop tile (segment background)

import 'package:flutter/painting.dart';

const List<Color> kPitchClassColors = <Color>[
  Color(0xFF28FF00),  // 0  C   bright green
  Color(0xFF00FFE8),  // 1  C#  cyan
  Color(0xFF007CFF),  // 2  D   blue
  Color(0xFF0500FF),  // 3  D#  deep blue
  Color(0xFF4500EA),  // 4  E   violet
  Color(0xFF520000),  // 5  F   dark red
  Color(0xFF740000),  // 6  F#  deeper red
  Color(0xFFB30000),  // 7  G   red
  Color(0xFFEE0000),  // 8  G#  bright red
  Color(0xFFFF6300),  // 9  A   orange
  Color(0xFFFFEC00),  // 10 A#  yellow
  Color(0xFF99FF00),  // 11 B   yellow-green
];

const List<String> kPitchClassNames = <String>[
  'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
];

/// Returns the spectral color for the given pitch class (0=C through 11=B).
/// Returns a neutral gray if the pitch class is out of range (e.g. 255 =
/// "unknown").
Color colorForPitchClass(int pitchClass) {
  if (pitchClass < 0 || pitchClass >= kPitchClassColors.length) {
    return const Color(0xFF666666);
  }
  return kPitchClassColors[pitchClass];
}

/// Returns the name of the pitch class as a string ("C", "C#", "D", ...).
/// Returns an empty string for out-of-range input.
String namePitchClass(int pitchClass) {
  if (pitchClass < 0 || pitchClass >= kPitchClassNames.length) return '';
  return kPitchClassNames[pitchClass];
}

/// Convenience: format a (pitchClass, isMinor) as e.g. "C" or "F#m".
String formatKey({required int pitchClass, required bool isMinor}) {
  final name = namePitchClass(pitchClass);
  if (name.isEmpty) return '—';
  return isMinor ? '${name}m' : name;
}
