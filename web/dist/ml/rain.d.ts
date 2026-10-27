/** Rain-tomorrow data decoding and feature engineering (mirrors ml/rain.py). */
export declare const RAW_NUMERIC: readonly ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"];
export declare const WIND_DIRS: readonly ["WindGustDir", "WindDir9am", "WindDir3pm"];
export type NumericField = (typeof RAW_NUMERIC)[number];
export type WindField = (typeof WIND_DIRS)[number];
export interface DaysJSON {
    start: string;
    stations: string[];
    index: Record<string, [number, number]>;
    columns: Record<string, (number | null)[]>;
}
export interface StationMeta {
    id: string;
    name: string;
    lat: number;
    lon: number;
    missing: Record<string, number>;
}
/** Feature-engineering constants shipped inside model.json at full precision. */
export interface RainPreprocessing {
    stations: Record<string, [number, number]>;
    compass: string[];
    compass_vectors: [number, number][];
}
export interface RainMeta {
    stations: StationMeta[];
    compass: string[];
    features: string[];
    groups: {
        label: string;
        features: string[];
    }[];
    splits: Record<"train" | "val" | "test", {
        rows: number;
        start: string;
        end: string;
        rain_rate: number;
    }>;
    models: Record<string, {
        label: string;
        val: Record<string, number>;
        test: Record<string, number>;
    }>;
    best_iteration: number;
    ci95: Record<string, [number, number]>;
    calibration: {
        lo: number;
        hi: number;
        mean_predicted: number;
        observed: number;
        count: number;
    }[];
    importance: {
        group: string;
        mean_abs_shap: number;
    }[];
    per_station: {
        station: string;
        days: number;
        rain_rate: number;
        accuracy: number;
        roc_auc: number | null;
    }[];
    source: string;
}
/** One day's readings at one station. NaN marks a missing reading; wind directions
 * are compass indices (0 = N, 1 = NNE, ...) or null. */
export interface Observation {
    station: string;
    day: number;
    numeric: Record<NumericField, number>;
    wind: Record<WindField, number | null>;
    rainToday: 0 | 1;
    rainTomorrow: 0 | 1;
}
export declare function dateOf(start: string, day: number): Date;
export declare function dayOfYear(date: Date): number;
export declare class RainDays {
    readonly start: string;
    readonly stations: string[];
    readonly length: number;
    private readonly cols;
    private readonly index;
    /** row lookup: station -> day -> row */
    private readonly lookup;
    private readonly stationIndex;
    readonly lastDay: number;
    constructor(json: DaysJSON);
    rowFor(station: string, day: number): number | undefined;
    rowsFor(station: string): number[];
    observation(row: number): Observation;
    stationOf(row: number): string;
    day(row: number): number;
    rainTomorrow(row: number): 0 | 1;
}
/** Model inputs in the order of `features`, built exactly like ml/rain.py. */
export declare function rainFeatures(obs: Observation, prep: RainPreprocessing, features: string[], start: string): Float64Array;
/** SHAP values summed into the readable groups defined in meta.groups. */
export declare function groupShap(phi: ArrayLike<number>, meta: RainMeta): {
    label: string;
    value: number;
}[];
