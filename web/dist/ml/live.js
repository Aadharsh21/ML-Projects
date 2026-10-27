const HOURLY = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation",
];
const DAILY = ["sunshine_duration"];
export function openMeteoUrl(stations) {
    const p = new URLSearchParams({
        latitude: stations.map((s) => s.lat).join(","),
        longitude: stations.map((s) => s.lon).join(","),
        hourly: HOURLY.join(","),
        daily: DAILY.join(","),
        timezone: "auto",
        past_days: "1",
        forecast_days: "2",
        wind_speed_unit: "kmh",
    });
    return `https://api.open-meteo.com/v1/forecast?${p}`;
}
const toCompass = (deg) => (deg === null || Number.isNaN(deg) ? null : Math.round((((deg % 360) + 360) % 360) / 22.5) % 16);
const oktas = (pct) => (pct === null ? NaN : Math.min(8, Math.max(0, Math.round(pct / 12.5))));
const num = (v) => (v === null || v === undefined ? NaN : v);
const round1 = (v) => Math.round(v * 10) / 10;
/** Map one Open-Meteo location onto the 22 readings the model expects. */
export function toReading(station, r, now, startDate) {
    const local = new Date(now.getTime() + r.utc_offset_seconds * 1000);
    const today = local.toISOString().slice(0, 10);
    const nowHour = local.getUTCHours();
    const times = r.hourly.time;
    const at = (day, hour) => times.indexOf(`${day}T${String(hour).padStart(2, "0")}:00`);
    const t0 = at(today, 0);
    if (t0 < 24)
        throw new Error(`Open-Meteo returned no data for ${today}`);
    const h = (name, i) => num(r.hourly[name][i]);
    const span = (name, from, to) => Array.from({ length: to - from + 1 }, (_, k) => h(name, from + k)).filter((v) => !Number.isNaN(v));
    const nine = t0 + 9;
    const three = t0 + 15;
    const rainWindow = span("precipitation", nine - 23, nine); // 9am yesterday to 9am today
    const rainfall = rainWindow.length ? round1(rainWindow.reduce((a, b) => a + b, 0)) : NaN;
    const lowest = (xs) => (xs.length ? Math.min(...xs) : NaN);
    const highest = (xs) => (xs.length ? Math.max(...xs) : NaN);
    const minTemp = lowest(span("temperature_2m", nine - 24, nine));
    const maxTemp = highest(span("temperature_2m", nine, nine + 24));
    const gusts = span("wind_gusts_10m", t0, t0 + 23);
    const gustMax = highest(gusts);
    let gustIdx = -1;
    for (let i = t0; i < t0 + 24; i++)
        if (h("wind_gusts_10m", i) === gustMax) {
            gustIdx = i;
            break;
        }
    const dailyIdx = r.daily.time.indexOf(today);
    const sunshine = dailyIdx >= 0 ? num(r.daily.sunshine_duration[dailyIdx]) / 3600 : NaN;
    const ahead = span("precipitation", nine + 1, nine + 24);
    const numeric = {
        MinTemp: round1(minTemp),
        MaxTemp: round1(maxTemp),
        Rainfall: rainfall,
        Evaporation: NaN, // pan evaporation has no Open-Meteo equivalent; the model handles it as missing
        Sunshine: round1(sunshine),
        WindGustSpeed: Math.round(gustMax),
        WindSpeed9am: Math.round(h("wind_speed_10m", nine)),
        WindSpeed3pm: Math.round(h("wind_speed_10m", three)),
        Humidity9am: Math.round(h("relative_humidity_2m", nine)),
        Humidity3pm: Math.round(h("relative_humidity_2m", three)),
        Pressure9am: round1(h("pressure_msl", nine)),
        Pressure3pm: round1(h("pressure_msl", three)),
        Cloud9am: oktas(h("cloud_cover", nine)),
        Cloud3pm: oktas(h("cloud_cover", three)),
        Temp9am: round1(h("temperature_2m", nine)),
        Temp3pm: round1(h("temperature_2m", three)),
    };
    const wind = {
        WindGustDir: gustIdx >= 0 ? toCompass(h("wind_direction_10m", gustIdx)) : null,
        WindDir9am: toCompass(h("wind_direction_10m", nine)),
        WindDir3pm: toCompass(h("wind_direction_10m", three)),
    };
    const forecastFields = [];
    if (nowHour < 15)
        forecastFields.push("Humidity3pm", "Pressure3pm", "Cloud3pm", "Temp3pm", "WindSpeed3pm", "WindDir3pm");
    if (nowHour < 9)
        forecastFields.push("Humidity9am", "Pressure9am", "Cloud9am", "Temp9am", "WindSpeed9am", "WindDir9am");
    forecastFields.push("MaxTemp", "Sunshine", "WindGustSpeed");
    const [y, m, d] = startDate.split("-").map(Number);
    const day = Math.round((Date.parse(`${today}T00:00:00Z`) - Date.UTC(y, m - 1, d)) / 86_400_000);
    return {
        station,
        date: today,
        timezone: r.timezone,
        obs: { station, day, numeric, wind, rainToday: rainfall > 1 ? 1 : 0, rainTomorrow: 0 },
        forecastFields,
        modelRainMm: ahead.length ? round1(ahead.reduce((a, b) => a + b, 0)) : null,
    };
}
export async function fetchLive(stations, startDate) {
    const response = await fetch(openMeteoUrl(stations));
    if (!response.ok)
        throw new Error(`Open-Meteo answered ${response.status}`);
    const json = (await response.json());
    const list = Array.isArray(json) ? json : [json];
    const now = new Date();
    const out = new Map();
    list.forEach((r, i) => {
        try {
            out.set(stations[i].id, toReading(stations[i].id, r, now, startDate));
        }
        catch {
            /* skip a station with incomplete data */
        }
    });
    if (!out.size)
        throw new Error("Open-Meteo returned no usable data");
    return out;
}
