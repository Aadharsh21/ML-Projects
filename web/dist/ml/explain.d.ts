import type { Observation } from "./rain.js";
export interface Step {
    label: string;
    value: number;
    from: number;
    to: number;
    other?: boolean;
}
/** Turn grouped SHAP values (log-odds) into a probability waterfall, biggest first. */
export declare function waterfall(base: number, groups: {
    label: string;
    value: number;
}[], top?: number): {
    start: number;
    end: number;
    steps: Step[];
};
/** A short reading for a group, e.g. "85%" for humidity at 3pm. */
export declare function describe(label: string, obs: Observation, compass: string[], stationName: string, date: Date): string | null;
/** One plain-English sentence about the biggest reasons behind a forecast. */
export declare function sentence(steps: Step[], end: number, describeFn: (label: string) => string | null): string;
