/**
 * Live mode: today's weather for every station from the Open-Meteo forecast API
 * (free, no key, CORS-enabled), mapped onto the station readings the model was
 * trained on. Open-Meteo values come from weather models rather than the stations'
 * own instruments, so live forecasts are approximate.
 */
import type { Observation } from "./rain.js";
declare const HOURLY: readonly ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation"];
declare const DAILY: readonly ["sunshine_duration"];
interface OpenMeteoResponse {
    latitude: number;
    longitude: number;
    utc_offset_seconds: number;
    timezone: string;
    hourly: Record<(typeof HOURLY)[number] | "time", (number | null)[] & string[]>;
    daily: Record<(typeof DAILY)[number] | "time", (number | null)[] & string[]>;
}
export interface LiveReading {
    station: string;
    /** local date at the station, YYYY-MM-DD */
    date: string;
    timezone: string;
    obs: Observation;
    /** readings later than the current hour are forecasts */
    forecastFields: string[];
    /** Open-Meteo's own expected rain in the same window (9am today to 9am tomorrow) */
    modelRainMm: number | null;
}
export declare function openMeteoUrl(stations: {
    lat: number;
    lon: number;
}[]): string;
/** Map one Open-Meteo location onto the 22 readings the model expects. */
export declare function toReading(station: string, r: OpenMeteoResponse, now: Date, startDate: string): LiveReading;
export declare function fetchLive(stations: {
    id: string;
    lat: number;
    lon: number;
}[], startDate: string): Promise<Map<string, LiveReading>>;
export {};
