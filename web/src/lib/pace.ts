/**
 * Speaking pace: one setting drives both how fast the reply streams on screen
 * (ms between words, applied by the server) and how fast it is read aloud.
 *
 * The server's default of 22 ms is "Normal" and keeps the voice at the rate it
 * always had (1.05). Slower settings slow the voice down towards a deliberate
 * speaking pace; faster ones speed it up. The browser's speech engine only
 * takes a new rate per utterance, so a change applies from the next sentence.
 */

export const PACE_DEFAULT_MS = 22;
export const PACE_MIN_MS = 0;
export const PACE_MAX_MS = 400; // the server accepts 0-400 (schemas.ConfigFrame)

export const PACE_PRESETS = [
  { label: "Fast", ms: 8 },
  { label: "Normal", ms: PACE_DEFAULT_MS },
  { label: "Slow", ms: 120 },
  { label: "Speech-like", ms: 360 },
] as const;

const VOICE_RATE_NORMAL = 1.05;
const VOICE_RATE_FASTEST = 1.6; // at 0 ms
const VOICE_RATE_SLOWEST = 0.6; // at PACE_MAX_MS

/** The speech-synthesis rate for a pace setting: 1.05 at the default, 1.6 at 0 ms, 0.6 at 400 ms. */
export function voiceRateFor(ms: number): number {
  const d = Math.min(PACE_MAX_MS, Math.max(PACE_MIN_MS, ms));
  if (d <= PACE_DEFAULT_MS) {
    const t = d / PACE_DEFAULT_MS;
    return round(VOICE_RATE_FASTEST + (VOICE_RATE_NORMAL - VOICE_RATE_FASTEST) * t);
  }
  // Slowing down feels even on a log scale: each doubling of the delay lowers the rate by the same factor.
  const exponent = Math.log(VOICE_RATE_SLOWEST / VOICE_RATE_NORMAL) / Math.log(PACE_DEFAULT_MS / PACE_MAX_MS);
  return round(VOICE_RATE_NORMAL * Math.pow(PACE_DEFAULT_MS / d, exponent));
}

/** The name of the zone a setting falls in, for the label next to the slider. */
export function paceLabel(ms: number): string {
  if (ms < 15) return "Fast";
  if (ms <= 40) return "Normal";
  if (ms < 240) return "Slow";
  return "Speech-like";
}

// The slider is not linear in ms: "Normal" would otherwise sit in the first 5%
// of the track. These anchors put Normal about a third of the way along.
const SLIDER_ANCHORS: ReadonlyArray<readonly [number, number]> = [
  [0, 0],
  [30, PACE_DEFAULT_MS],
  [65, 120],
  [100, PACE_MAX_MS],
];

/** Slider position (0-100) to ms per word, rounded to the slider's 2 ms grain. */
export function sliderToMs(position: number): number {
  const p = Math.min(100, Math.max(0, position));
  for (let i = 1; i < SLIDER_ANCHORS.length; i++) {
    const [p0, m0] = SLIDER_ANCHORS[i - 1];
    const [p1, m1] = SLIDER_ANCHORS[i];
    if (p <= p1) return Math.round((m0 + ((p - p0) / (p1 - p0)) * (m1 - m0)) / 2) * 2;
  }
  return PACE_MAX_MS;
}

/** ms per word to slider position (0-100). */
export function msToSlider(ms: number): number {
  const m = Math.min(PACE_MAX_MS, Math.max(PACE_MIN_MS, ms));
  for (let i = 1; i < SLIDER_ANCHORS.length; i++) {
    const [p0, m0] = SLIDER_ANCHORS[i - 1];
    const [p1, m1] = SLIDER_ANCHORS[i];
    if (m <= m1) return p0 + ((m - m0) / (m1 - m0)) * (p1 - p0);
  }
  return 100;
}

const round = (x: number) => Math.round(x * 100) / 100;
