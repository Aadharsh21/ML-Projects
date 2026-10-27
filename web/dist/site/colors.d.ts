/** The sequential blue ramp used for every "chance of rain" encoding (map, calendar, legend). */
export type ColorScheme = "light" | "dark";
/**
 * Probability (0..1) to colour. In light mode low values are pale and high values deep
 * blue; in dark mode the ramp flips so low values recede into the dark page and high
 * values glow.
 */
export declare function chanceColor(p: number, scheme?: ColorScheme): string;
/** CSS linear-gradient matching chanceColor, for legends. */
export declare function chanceGradient(scheme?: ColorScheme): string;
