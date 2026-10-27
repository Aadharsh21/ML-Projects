/** Rain-tomorrow data decoding and feature engineering (mirrors ml/rain.py). */
export const RAW_NUMERIC = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
];
export const WIND_DIRS = ["WindGustDir", "WindDir9am", "WindDir3pm"];
const MS_PER_DAY = 86_400_000;
export function dateOf(start, day) {
    const [y, m, d] = start.split("-").map(Number);
    return new Date(Date.UTC(y, m - 1, d + day));
}
export function dayOfYear(date) {
    return (date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 1)) / MS_PER_DAY + 1;
}
export class RainDays {
    start;
    stations;
    length;
    cols;
    index;
    /** row lookup: station -> day -> row */
    lookup = new Map();
    stationIndex;
    lastDay;
    constructor(json) {
        this.start = json.start;
        this.stations = json.stations;
        this.cols = json.columns;
        this.index = json.index;
        this.length = json.columns.day.length;
        let last = 0;
        this.stationIndex = new Uint8Array(this.length);
        for (const [si, s] of this.stations.entries()) {
            const [from, count] = json.index[s];
            const m = new Map();
            for (let r = from; r < from + count; r++) {
                this.stationIndex[r] = si;
                const d = json.columns.day[r];
                m.set(d, r);
                if (d > last)
                    last = d;
            }
            this.lookup.set(s, m);
        }
        this.lastDay = last;
    }
    rowFor(station, day) {
        return this.lookup.get(station)?.get(day);
    }
    rowsFor(station) {
        const [from, count] = this.index[station];
        return Array.from({ length: count }, (_, i) => from + i);
    }
    observation(row) {
        const numeric = {};
        for (const f of RAW_NUMERIC) {
            const v = this.cols[f][row];
            numeric[f] = v === null ? NaN : v;
        }
        const wind = {};
        for (const f of WIND_DIRS)
            wind[f] = this.cols[f][row];
        return {
            station: this.stationOf(row),
            day: this.cols.day[row],
            numeric,
            wind,
            rainToday: this.cols.RainToday[row],
            rainTomorrow: this.cols.RainTomorrow[row],
        };
    }
    stationOf(row) {
        return this.stations[this.stationIndex[row]];
    }
    day(row) {
        return this.cols.day[row];
    }
    rainTomorrow(row) {
        return this.cols.RainTomorrow[row];
    }
}
/** Model inputs in the order of `features`, built exactly like ml/rain.py. */
export function rainFeatures(obs, prep, features, start) {
    const latlon = prep.stations[obs.station];
    if (!latlon)
        throw new Error(`unknown station ${obs.station}`);
    const doy = dayOfYear(dateOf(start, obs.day));
    const v = { ...obs.numeric };
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
        if (!(f in v))
            throw new Error(`feature ${f} not computed`);
        return v[f];
    });
}
/** SHAP values summed into the readable groups defined in meta.groups. */
export function groupShap(phi, meta) {
    const col = new Map(meta.features.map((f, i) => [f, i]));
    return meta.groups.map((g) => ({
        label: g.label,
        value: g.features.reduce((s, f) => s + phi[col.get(f)], 0),
    }));
}
