/** Rain-tomorrow data decoding and feature engineering (mirrors ml/rain.py). */

export const RAW_NUMERIC = [
  "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
  "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
  "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
] as const;
export const WIND_DIRS = ["WindGustDir", "WindDir9am", "WindDir3pm"] as const;

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
  groups: { label: string; features: string[] }[];
  splits: Record<"train" | "val" | "test", { rows: number; start: string; end: string; rain_rate: number }>;
  models: Record<string, { label: string; val: Record<string, number>; test: Record<string, number> }>;
  best_iteration: number;
  ci95: Record<string, [number, number]>;
  calibration: { lo: number; hi: number; mean_predicted: number; observed: number; count: number }[];
  importance: { group: string; mean_abs_shap: number }[];
  per_station: { station: string; days: number; rain_rate: number; accuracy: number; roc_auc: number | null }[];
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

const MS_PER_DAY = 86_400_000;

export function dateOf(start: string, day: number): Date {
  const [y, m, d] = start.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d + day));
}

export function dayOfYear(date: Date): number {
  return (date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 1)) / MS_PER_DAY + 1;
}

export class RainDays {
  readonly start: string;
  readonly stations: string[];
  readonly length: number;
  private readonly cols: Record<string, (number | null)[]>;
  private readonly index: Record<string, [number, number]>;
  /** row lookup: station -> day -> row */
  private readonly lookup = new Map<string, Map<number, number>>();
  private readonly stationIndex: Uint8Array;
  readonly lastDay: number;

  constructor(json: DaysJSON) {
    this.start = json.start;
    this.stations = json.stations;
    this.cols = json.columns;
    this.index = json.index;
    this.length = json.columns.day.length;
    let last = 0;
    this.stationIndex = new Uint8Array(this.length);
    for (const [si, s] of this.stations.entries()) {
      const [from, count] = json.index[s];
      const m = new Map<number, number>();
      for (let r = from; r < from + count; r++) {
        this.stationIndex[r] = si;
        const d = json.columns.day[r] as number;
        m.set(d, r);
        if (d > last) last = d;
      }
      this.lookup.set(s, m);
    }
    this.lastDay = last;
  }

  rowFor(station: string, day: number): number | undefined {
    return this.lookup.get(station)?.get(day);
  }

  rowsFor(station: string): number[] {
    const [from, count] = this.index[station];
    return Array.from({ length: count }, (_, i) => from + i);
  }

  observation(row: number): Observation {
    const numeric = {} as Record<NumericField, number>;
    for (const f of RAW_NUMERIC) {
      const v = this.cols[f][row];
      numeric[f] = v === null ? NaN : v;
    }
    const wind = {} as Record<WindField, number | null>;
    for (const f of WIND_DIRS) wind[f] = this.cols[f][row];
    return {
      station: this.stationOf(row),
      day: this.cols.day[row] as number,
      numeric,
      wind,
      rainToday: this.cols.RainToday[row] as 0 | 1,
      rainTomorrow: this.cols.RainTomorrow[row] as 0 | 1,
    };
  }

  stationOf(row: number): string {
    return this.stations[this.stationIndex[row]];
  }

  day(row: number): number {
    return this.cols.day[row] as number;
  }

  rainTomorrow(row: number): 0 | 1 {
    return this.cols.RainTomorrow[row] as 0 | 1;
  }
}

/** Model inputs in the order of `features`, built exactly like ml/rain.py. */
export function rainFeatures(
  obs: Observation,
  prep: RainPreprocessing,
  features: string[],
  start: string,
): Float64Array {
  const latlon = prep.stations[obs.station];
  if (!latlon) throw new Error(`unknown station ${obs.station}`);
  const doy = dayOfYear(dateOf(start, obs.day));
  const v: Record<string, number> = { ...obs.numeric };
  v.RainToday = obs.rainToday;
  v.Latitude = latlon[0];
  v.Longitude = latlon[1];
  v.SeasonSin = Math.sin((2 * Math.PI * doy) / 365.25);
  v.SeasonCos = Math.cos((2 * Math.PI * doy) / 365.25);
  for (const f of WIND_DIRS) {
    const idx = obs.wind[f];
    v[`${f}Sin`] = idx === null ? NaN : prep.compass_vectors[idx][0];
    v[`${f}Cos`] = idx === null ? NaN : prep.compass_vectors[idx][1];
  }
  v.TempRange = v.MaxTemp - v.MinTemp;
  v.HumidityChange = v.Humidity3pm - v.Humidity9am;
  v.PressureChange = v.Pressure3pm - v.Pressure9am;
  v.TempChange = v.Temp3pm - v.Temp9am;
  v.CloudChange = v.Cloud3pm - v.Cloud9am;
  return Float64Array.from(features, (f) => {
    if (!(f in v)) throw new Error(`feature ${f} not computed`);
    return v[f];
  });
}

/** SHAP values summed into the readable groups defined in meta.groups. */
export function groupShap(phi: ArrayLike<number>, meta: RainMeta): { label: string; value: number }[] {
  const col = new Map(meta.features.map((f, i) => [f, i]));
  return meta.groups.map((g) => ({
    label: g.label,
    value: g.features.reduce((s, f) => s + phi[col.get(f)!], 0),
  }));
}
