/**
 * Everything the rain page needs, in one object: every 2016-2017 test day scored on
 * load, forecasts with explanations for any station and day, "what if" edits to the
 * readings, today's live weather, and the map layout.
 */
import { type Step } from "../ml/explain.js";
import { type LiveReading } from "../ml/live.js";
import { type NumericField, type Observation, type RainMeta, type StationMeta, type WindField } from "../ml/rain.js";
import { type LoadOptions } from "./load.js";
import { type GeoCollection, type RainMapLayout } from "./map.js";
/** Changes to the recorded readings. A wind direction of null means "not recorded". */
export interface RainEdits {
    numeric?: Partial<Record<NumericField, number>>;
    wind?: Partial<Record<WindField, number | null>>;
}
/** A slider for one numeric reading. `primary` readings are shown first. */
export interface ReadingSpec {
    field: NumericField;
    label: string;
    min: number;
    max: number;
    step: number;
    unit: string;
    digits: number;
    primary: boolean;
}
/** The 16 numeric readings, most influential first. */
export declare const RAIN_READINGS: ReadingSpec[];
/** The three wind directions, chosen from `compass` (index) or null. */
export declare const RAIN_WINDS: {
    field: WindField;
    label: string;
}[];
export declare function hasEdits(edits?: RainEdits): boolean;
/** Recorded readings with edits on top. "Rained today" follows the Bureau's definition
 * (more than 1 mm by 9am), so it updates when rainfall is edited. */
export declare function applyEdits(obs: Observation, edits?: RainEdits): Observation;
export interface RainForecast {
    station: StationMeta;
    /** the day the readings are from; the forecast is for rain by 9am the day after */
    date: Date;
    live: boolean;
    recorded: Observation;
    /** the readings the forecast used (recorded plus edits) */
    observation: Observation;
    edited: boolean;
    /** chance of more than 1 mm of rain by 9am the next day */
    probability: number;
    /** the same forecast from the recorded readings, before edits */
    recordedProbability: number;
    /** whether it rained (past days only) */
    actual: 0 | 1 | null;
    /** whether a 50% cut-off called it right (past days only) */
    correct: boolean | null;
    /** from a typical day's chance to this forecast, biggest reasons first */
    waterfall: {
        start: number;
        end: number;
        steps: Step[];
    };
    /** one plain-English sentence about the biggest reasons */
    sentence: string;
    /** the reading behind a waterfall label, e.g. "85%" for "Humidity at 3pm" */
    describe: (label: string) => string | null;
}
export interface RainStationDay {
    station: StationMeta;
    probability: number | null;
    actual: 0 | 1 | null;
    correct: boolean | null;
}
export interface RainDayView {
    stations: RainStationDay[];
    /** stations with a forecast and a known outcome, and how many of those were right */
    scored: number;
    correct: number;
}
export interface RainLive {
    fetchedAt: Date;
    readings: Map<string, LiveReading>;
    dayView: RainDayView;
    forecast(station: string, edits?: RainEdits): RainForecast | null;
}
export interface RainExplorer {
    meta: RainMeta;
    geo: GeoCollection;
    /** stations in A-Z order of name */
    stations: StationMeta[];
    compass: string[];
    /** first test day as YYYY-MM-DD; days are counted from it */
    start: string;
    lastDay: number;
    dateOf(day: number): Date;
    /** day number of a date (YYYY-MM-DD or Date), clamped to the test period */
    dayOf(date: string | Date): number;
    /** per day: share of stations where it rained the next day */
    rainShare: Float32Array;
    dayView(day: number): RainDayView;
    history(station: string): {
        day: number;
        date: Date;
        probability: number;
        actual: 0 | 1;
    }[];
    stationStats(station: string): RainMeta["per_station"][number] | undefined;
    forecast(station: string, day: number, edits?: RainEdits): RainForecast | null;
    /** today's weather for all stations from Open-Meteo, forecast by the same model */
    fetchLive(): Promise<RainLive>;
    nearest(lat: number, lon: number): {
        station: StationMeta;
        km: number;
    };
    /** SVG layout of the map (default viewBox 760 x 600) */
    map(width?: number, height?: number): RainMapLayout;
}
/** Download the rain model and test days, then score all 25,710 station-days. */
export declare function loadRain(options?: LoadOptions): Promise<RainExplorer>;
