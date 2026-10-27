import { describe, expect, it, vi } from "vitest";
import { fetchLive, openMeteoUrl, toReading } from "../src/ml/live";

/** A synthetic Open-Meteo response: 72 hourly values from yesterday 00:00 local. */
function response(over: Partial<Record<string, number[]>> = {}) {
  const time = Array.from({ length: 72 }, (_, i) => {
    const day = ["2026-10-26", "2026-10-27", "2026-10-28"][Math.floor(i / 24)];
    return `${day}T${String(i % 24).padStart(2, "0")}:00`;
  });
  const fill = (f: (i: number) => number) => Array.from({ length: 72 }, (_, i) => f(i));
  const hourly: Record<string, unknown> = {
    time,
    temperature_2m: fill((i) => 10 + (i % 24) * 0.5),
    relative_humidity_2m: fill((i) => (i === 33 ? 81 : i === 39 ? 64 : 50)),
    pressure_msl: fill((i) => (i === 33 ? 1012.34 : i === 39 ? 1008.96 : 1010)),
    cloud_cover: fill((i) => (i === 33 ? 100 : i === 39 ? 30 : 0)),
    wind_speed_10m: fill((i) => (i === 33 ? 12.4 : i === 39 ? 25.6 : 5)),
    wind_direction_10m: fill((i) => (i === 33 ? 10 : i === 39 ? 350 : i === 30 ? 225 : 90)),
    wind_gusts_10m: fill((i) => (i === 30 ? 55 : 20)),
    precipitation: fill((i) => (i >= 20 && i <= 33 ? 0.5 : i >= 40 && i <= 44 ? 0.2 : 0)),
    ...over,
  };
  return {
    latitude: -33.87,
    longitude: 151.21,
    utc_offset_seconds: 36000,
    timezone: "Australia/Sydney",
    hourly,
    daily: { time: ["2026-10-26", "2026-10-27", "2026-10-28"], sunshine_duration: [0, 18000, 0] },
  };
}

// 06:00 UTC is 4pm in Sydney (UTC+10)
const NOW = new Date("2026-10-27T06:00:00Z");

describe("live readings from Open-Meteo", () => {
  it("maps hourly weather onto the station readings the model expects", () => {
    const r = toReading("Sydney", response() as never, NOW, "2016-01-01");
    expect(r.date).toBe("2026-10-27");
    const n = r.obs.numeric;
    expect(n.Rainfall).toBe(7); // 14 hours x 0.5 mm between 9am yesterday and 9am today
    expect(r.obs.rainToday).toBe(1);
    expect(n.MinTemp).toBe(10); // coldest hour in the 24 h to 9am
    expect(n.MaxTemp).toBe(21.5); // warmest hour in the 24 h from 9am
    expect([n.Humidity9am, n.Humidity3pm]).toEqual([81, 64]);
    expect([n.Pressure9am, n.Pressure3pm]).toEqual([1012.3, 1009]);
    expect([n.Cloud9am, n.Cloud3pm]).toEqual([8, 2]); // percent to eighths of sky
    expect([n.WindSpeed9am, n.WindSpeed3pm]).toEqual([12, 26]);
    expect(n.WindGustSpeed).toBe(55);
    expect(n.Sunshine).toBe(5);
    expect(Number.isNaN(n.Evaporation)).toBe(true);
    expect(r.obs.wind).toEqual({ WindGustDir: 10, WindDir9am: 0, WindDir3pm: 0 }); // SW, N, N
    expect(r.modelRainMm).toBe(1); // Open-Meteo's own rain from 9am today to 9am tomorrow
    expect(r.obs.day).toBe(3952); // days since 2016-01-01, for the time-of-year features
  });

  it("flags readings that are still forecasts", () => {
    const morning = new Date("2026-10-26T21:30:00Z"); // 7:30am in Sydney
    const r = toReading("Sydney", response() as never, morning, "2016-01-01");
    expect(r.forecastFields).toContain("Humidity9am");
    expect(r.forecastFields).toContain("Humidity3pm");
    const evening = toReading("Sydney", response() as never, NOW, "2016-01-01");
    expect(evening.forecastFields).not.toContain("Humidity3pm");
  });

  it("asks for every station in one request", () => {
    const url = new URL(openMeteoUrl([{ lat: -33.87, lon: 151.21 }, { lat: -31.95, lon: 115.86 }]));
    expect(url.searchParams.get("latitude")).toBe("-33.87,-31.95");
    expect(url.searchParams.get("timezone")).toBe("auto");
    expect(url.searchParams.get("past_days")).toBe("1");
  });

  it("fetchLive handles a multi-location answer and skips broken stations", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
    const broken = { ...response(), hourly: { ...response().hourly, time: [] } };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify([response(), broken]))));
    const out = await fetchLive(
      [
        { id: "Sydney", lat: -33.87, lon: 151.21 },
        { id: "Perth", lat: -31.95, lon: 115.86 },
      ],
      "2016-01-01",
    );
    expect([...out.keys()]).toEqual(["Sydney"]);
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });
});
