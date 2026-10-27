/**
 * Everything the rain page needs, in one object: every 2016-2017 test day scored on
 * load, forecasts with explanations for any station and day, "what if" edits to the
 * readings, today's live weather, and the map layout.
 */
import { describe as describeReading, sentence, waterfall } from "../ml/explain.js";
import { Forest } from "../ml/lgbm.js";
import { fetchLive as fetchLiveWeather } from "../ml/live.js";
import { RainDays, dateOf, groupShap, rainFeatures, } from "../ml/rain.js";
import { loadFiles, nextFrame } from "./load.js";
import { rainMap } from "./map.js";
const spec = (field, label, min, max, step, unit, digits, primary) => ({
    field, label, min, max, step, unit, digits, primary,
});
/** The 16 numeric readings, most influential first. */
export const RAIN_READINGS = [
    spec("Humidity3pm", "Humidity at 3pm", 0, 100, 1, "%", 0, true),
    spec("Pressure3pm", "Pressure at 3pm", 980, 1040, 0.1, " hPa", 1, true),
    spec("WindGustSpeed", "Strongest wind gust", 0, 135, 1, " km/h", 0, true),
    spec("Sunshine", "Sunshine", 0, 14.5, 0.1, " h", 1, true),
    spec("Cloud3pm", "Cloud at 3pm", 0, 8, 1, " eighths", 0, true),
    spec("Rainfall", "Rain so far (24 h to 9am)", 0, 50, 0.2, " mm", 1, true),
    spec("Humidity9am", "Humidity at 9am", 0, 100, 1, "%", 0, false),
    spec("Pressure9am", "Pressure at 9am", 980, 1040, 0.1, " hPa", 1, false),
    spec("Cloud9am", "Cloud at 9am", 0, 8, 1, " eighths", 0, false),
    spec("WindSpeed9am", "Wind speed at 9am", 0, 90, 1, " km/h", 0, false),
    spec("WindSpeed3pm", "Wind speed at 3pm", 0, 90, 1, " km/h", 0, false),
    spec("MaxTemp", "Max temperature", -5, 48, 0.1, "°C", 1, false),
    spec("MinTemp", "Min temperature", -8, 34, 0.1, "°C", 1, false),
    spec("Temp9am", "Temperature at 9am", -7, 41, 0.1, "°C", 1, false),
    spec("Temp3pm", "Temperature at 3pm", -5, 47, 0.1, "°C", 1, false),
    spec("Evaporation", "Evaporation", 0, 20, 0.2, " mm", 1, false),
];
/** The three wind directions, chosen from `compass` (index) or null. */
export const RAIN_WINDS = [
    { field: "WindGustDir", label: "Gust direction" },
    { field: "WindDir9am", label: "Wind at 9am from" },
    { field: "WindDir3pm", label: "Wind at 3pm from" },
];
export function hasEdits(edits) {
    return !!edits && (Object.keys(edits.numeric ?? {}).length > 0 || Object.keys(edits.wind ?? {}).length > 0);
}
/** Recorded readings with edits on top. "Rained today" follows the Bureau's definition
 * (more than 1 mm by 9am), so it updates when rainfall is edited. */
export function applyEdits(obs, edits) {
    if (!hasEdits(edits))
        return obs;
    const numeric = { ...obs.numeric, ...edits.numeric };
    const rainToday = edits.numeric && "Rainfall" in edits.numeric ? (numeric.Rainfall > 1 ? 1 : 0) : obs.rainToday;
    return { ...obs, numeric, wind: { ...obs.wind, ...edits.wind }, rainToday };
}
function haversineKm(a, b) {
    const rad = Math.PI / 180;
    const dLat = (b[0] - a[0]) * rad;
    const dLon = (b[1] - a[1]) * rad;
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLon / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.sqrt(h));
}
/** Download the rain model and test days, then score all 25,710 station-days. */
export async function loadRain(options = {}) {
    const files = await loadFiles({ model: "rain/model.json", days: "rain/days.json", meta: "rain/meta.json", geo: "rain/australia.json" }, options, 0.7, "Downloading the model and 25,710 days of readings");
    const { meta, geo } = files;
    const prep = files.model.preprocessing;
    const forest = new Forest(files.model);
    const days = new RainDays(files.days);
    const stage = "Scoring every station-day in your browser";
    const probabilities = new Float32Array(days.length);
    const chunk = 2000;
    for (let s = 0; s < days.length; s += chunk) {
        for (let r = s; r < Math.min(days.length, s + chunk); r++) {
            probabilities[r] = forest.predictProba(rainFeatures(days.observation(r), prep, forest.features, days.start));
        }
        options.onProgress?.(0.7 + (0.3 * Math.min(days.length, s + chunk)) / days.length, stage);
        await nextFrame();
    }
    const rained = new Float32Array(days.lastDay + 1);
    const counts = new Float32Array(days.lastDay + 1);
    for (let r = 0; r < days.length; r++) {
        counts[days.day(r)]++;
        rained[days.day(r)] += days.rainTomorrow(r);
    }
    const rainShare = rained.map((v, i) => (counts[i] ? v / counts[i] : 0));
    const byId = new Map(meta.stations.map((s) => [s.id, s]));
    const stationOf = (id) => {
        const s = byId.get(id);
        if (!s)
            throw new Error(`unknown station ${id}`);
        return s;
    };
    const explain = (station, recorded, edits, live) => {
        const obs = applyEdits(recorded, edits);
        const edited = hasEdits(edits);
        const x = rainFeatures(obs, prep, forest.features, days.start);
        const probability = forest.predictProba(x);
        const recordedProbability = edited ? forest.predictProba(rainFeatures(recorded, prep, forest.features, days.start)) : probability;
        const phi = forest.shap(x);
        const wf = waterfall(phi[phi.length - 1], groupShap(phi, meta));
        const date = dateOf(days.start, obs.day);
        const describe = (label) => describeReading(label, obs, prep.compass, station.name, date);
        const actual = live ? null : recorded.rainTomorrow;
        return {
            station,
            date,
            live,
            recorded,
            observation: obs,
            edited,
            probability,
            recordedProbability,
            actual,
            correct: actual === null ? null : (probability >= 0.5 ? 1 : 0) === actual,
            waterfall: wf,
            sentence: sentence(wf.steps, wf.end, describe),
            describe,
        };
    };
    const summarize = (stations) => {
        let scored = 0;
        let correct = 0;
        for (const s of stations) {
            if (s.correct === null)
                continue;
            scored++;
            if (s.correct)
                correct++;
        }
        return { stations, scored, correct };
    };
    return {
        meta,
        geo,
        stations: [...meta.stations].sort((a, b) => a.name.localeCompare(b.name)),
        compass: prep.compass,
        start: days.start,
        lastDay: days.lastDay,
        dateOf: (day) => dateOf(days.start, day),
        dayOf(date) {
            const iso = typeof date === "string" ? date : date.toISOString().slice(0, 10);
            const d = Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${days.start}T00:00:00Z`)) / 86_400_000);
            return Number.isFinite(d) ? Math.min(days.lastDay, Math.max(0, d)) : 0;
        },
        rainShare,
        dayView(day) {
            return summarize(meta.stations.map((station) => {
                const row = days.rowFor(station.id, day);
                if (row === undefined)
                    return { station, probability: null, actual: null, correct: null };
                const probability = probabilities[row];
                const actual = days.rainTomorrow(row);
                return { station, probability, actual, correct: (probability >= 0.5 ? 1 : 0) === actual };
            }));
        },
        history(station) {
            return days.rowsFor(stationOf(station).id).map((r) => ({
                day: days.day(r),
                date: dateOf(days.start, days.day(r)),
                probability: probabilities[r],
                actual: days.rainTomorrow(r),
            }));
        },
        stationStats: (station) => meta.per_station.find((s) => s.station === station),
        forecast(station, day, edits) {
            const s = stationOf(station);
            const row = days.rowFor(s.id, day);
            return row === undefined ? null : explain(s, days.observation(row), edits, false);
        },
        async fetchLive() {
            const readings = await fetchLiveWeather(meta.stations, days.start);
            const forecastFor = (id, edits) => {
                const reading = readings.get(id);
                return reading ? explain(stationOf(id), reading.obs, edits, true) : null;
            };
            return {
                fetchedAt: new Date(),
                readings,
                dayView: summarize(meta.stations.map((station) => {
                    const reading = readings.get(station.id);
                    const probability = reading ? forest.predictProba(rainFeatures(reading.obs, prep, forest.features, days.start)) : null;
                    return { station, probability, actual: null, correct: null };
                })),
                forecast: forecastFor,
            };
        },
        nearest(lat, lon) {
            let best = meta.stations[0];
            let km = Infinity;
            for (const s of meta.stations) {
                const d = haversineKm([lat, lon], [s.lat, s.lon]);
                if (d < km) {
                    km = d;
                    best = s;
                }
            }
            return { station: best, km };
        },
        map: (width, height) => rainMap(geo, meta.stations, width, height),
    };
}
