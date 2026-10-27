/** Data colour ramps from the validated palette (see styles/tokens.css). */

// sequential blue, steps 100..700
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mix(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const c = (x: number, y: number) => Math.round(x + (y - x) * t);
  return `rgb(${c(r1, r2)}, ${c(g1, g2)}, ${c(b1, b2)})`;
}

function along(ramp: string[], t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (ramp.length - 1);
  const i = Math.min(Math.floor(x), ramp.length - 2);
  return mix(ramp[i], ramp[i + 1], x - i);
}

/**
 * Probability (0..1) to colour. Light mode runs light -> dark; dark mode flips the
 * anchor so low values recede into the dark surface and high values glow.
 */
export function probColor(p: number, scheme: "light" | "dark"): string {
  if (scheme === "dark") return along([...BLUE].reverse().slice(0, 12), p);
  return along(BLUE.slice(1), p);
}

/** CSS gradient matching probColor, for legends. */
export function probGradient(scheme: "light" | "dark"): string {
  const stops = Array.from({ length: 11 }, (_, i) => probColor(i / 10, scheme));
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
