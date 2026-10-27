/** The sequential blue ramp used for every "chance of rain" encoding (map, calendar, legend). */
// validated sequential blue, light to dark
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(a, b, t) {
    const [r1, g1, b1] = hexToRgb(a);
    const [r2, g2, b2] = hexToRgb(b);
    const c = (x, y) => Math.round(x + (y - x) * t);
    return `rgb(${c(r1, r2)}, ${c(g1, g2)}, ${c(b1, b2)})`;
}
function along(ramp, t) {
    const x = Math.min(1, Math.max(0, t)) * (ramp.length - 1);
    const i = Math.min(Math.floor(x), ramp.length - 2);
    return mix(ramp[i], ramp[i + 1], x - i);
}
/**
 * Probability (0..1) to colour. In light mode low values are pale and high values deep
 * blue; in dark mode the ramp flips so low values recede into the dark page and high
 * values glow.
 */
export function chanceColor(p, scheme = "light") {
    if (scheme === "dark")
        return along([...BLUE].reverse().slice(0, 12), p);
    return along(BLUE.slice(1), p);
}
/** CSS linear-gradient matching chanceColor, for legends. */
export function chanceGradient(scheme = "light") {
    const stops = Array.from({ length: 11 }, (_, i) => chanceColor(i / 10, scheme));
    return `linear-gradient(to right, ${stops.join(", ")})`;
}
