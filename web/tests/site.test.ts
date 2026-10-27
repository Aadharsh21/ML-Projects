import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { profileFromRow } from "../src/ml/insurance";
import { sigmoid } from "../src/ml/lgbm";
import { loadFraud, type FraudLab, type FraudPolicy } from "../src/site/fraud";
import { loadInsurance, type InsuranceCalculator } from "../src/site/insurance";
import { loadIris, type IrisPlayground } from "../src/site/iris";
import { DATA_URL, REFERENCE_URL } from "../src/site/load";
import { applyEdits, loadRain, type RainExplorer } from "../src/site/rain";
import { selfTest } from "../src/site/selftest";
import { chanceColor, chanceGradient } from "../src/site/colors";
import { formatChance, formatPoints, formatSignedUsd } from "../src/site/format";
import { load } from "./load";

// read the exported files from disk instead of over the network
const fetcher = async (path: string) => load(`../public/data/${path}`);
const referenceFetcher = async (name: string) => load(`./fixtures/${name}`);

describe("file locations", () => {
  it("resolve next to the module, so a CDN copy reads its own release's data", () => {
    expect(DATA_URL.endsWith("/web/public/data/")).toBe(true);
    expect(REFERENCE_URL.endsWith("/web/tests/fixtures/")).toBe(true);
  });
});

const LOAD = 120_000;

describe("rain explorer", () => {
  const progress: number[] = [];
  let rain: RainExplorer;
  let meta: RainExplorer["meta"];
  beforeAll(async () => {
    rain = await loadRain({ fetcher, onProgress: (f) => progress.push(f) });
    meta = rain.meta;
  }, LOAD);
  const fx = load<{ rows: { station: string; date: string }[]; probability: number[] }>("./fixtures/rain.json");

  it("reports progress up to 1", () => {
    expect(progress[progress.length - 1]).toBeCloseTo(1, 12);
    progress.slice(1).forEach((p, i) => expect(p).toBeGreaterThanOrEqual(progress[i]));
  });

  it("forecasts match LightGBM on the reference rows", () => {
    fx.rows.forEach((r, i) => {
      const f = rain.forecast(r.station, rain.dayOf(r.date))!;
      expect(f.probability).toBeCloseTo(fx.probability[i], 9);
      expect(f.waterfall.end).toBeCloseTo(f.probability, 9);
      expect(f.actual === 0 || f.actual === 1).toBe(true);
    });
  });

  it("scoring every day reproduces the published test accuracy", () => {
    let scored = 0;
    let correct = 0;
    for (let d = 0; d <= rain.lastDay; d++) {
      const v = rain.dayView(d);
      scored += v.scored;
      correct += v.correct;
    }
    const t = meta.models.v2_lightgbm.test;
    expect(scored).toBe(meta.splits.test.rows);
    expect(correct).toBe(t.tp + t.tn);
  });

  it("station history matches the per-station results", () => {
    const stats = rain.stationStats("Sydney")!;
    const h = rain.history("Sydney");
    expect(h.length).toBe(stats.days);
    const right = h.filter((d) => (d.probability >= 0.5 ? 1 : 0) === d.actual).length;
    expect(right / h.length).toBeCloseTo(stats.accuracy, 5);
  });

  it("edits change the forecast and keep the recorded one", () => {
    const day = rain.dayOf("2016-07-08");
    const base = rain.forecast("Sydney", day)!;
    const wetter = rain.forecast("Sydney", day, { numeric: { Humidity3pm: 100, Sunshine: 0 } })!;
    expect(wetter.edited).toBe(true);
    expect(wetter.recordedProbability).toBeCloseTo(base.probability, 12);
    expect(wetter.probability).toBeGreaterThan(base.probability);
    expect(wetter.observation.numeric.Humidity3pm).toBe(100);
    expect(base.edited).toBe(false);
    expect(typeof base.sentence).toBe("string");
    expect(base.describe("Humidity at 3pm")).toMatch(/%$/);
    // rainfall over 1 mm by 9am means it rained today
    const dry = { ...base.recorded, numeric: { ...base.recorded.numeric, Rainfall: 0 }, rainToday: 0 as const };
    expect(applyEdits(dry, { numeric: { Rainfall: 5 } }).rainToday).toBe(1);
  });

  it("days and dates round-trip and clamp to the test period", () => {
    expect(rain.dateOf(rain.dayOf("2016-07-08")).toISOString().slice(0, 10)).toBe("2016-07-08");
    expect(rain.dayOf("2010-01-01")).toBe(0);
    expect(rain.dayOf("2030-01-01")).toBe(rain.lastDay);
    expect(rain.rainShare.length).toBe(rain.lastDay + 1);
  });

  it("finds the nearest station", () => {
    const { station, km } = rain.nearest(-33.86, 151.21);
    expect(["Sydney", "SydneyAirport"]).toContain(station.id);
    expect(km).toBeLessThan(15);
  });

  it("lays out the map the right way up", () => {
    const map = rain.map();
    const ids = meta.stations.map((s) => s.id);
    expect(Object.keys(map.points).sort()).toEqual([...ids].sort());
    for (const [x, y] of Object.values(map.points)) {
      expect(x).toBeGreaterThan(0);
      expect(x).toBeLessThan(map.width);
      expect(y).toBeGreaterThan(0);
      expect(y).toBeLessThan(map.height);
    }
    expect(map.points.Perth[0]).toBeLessThan(map.points.Sydney[0]);
    expect(map.points.Darwin[1]).toBeLessThan(map.points.Hobart[1]);
    expect(map.land.startsWith("M")).toBe(true);
    expect(map.inset.land.length).toBeGreaterThan(0);
    expect(map.stationAt(...map.points.Sydney)).toBe("Sydney");
    expect(map.stationAt(1, 1)).toBeNull();
    expect(map.labelled).toContain("Sydney");
  });

  describe("live mode", () => {
    const realFetch = globalThis.fetch;
    afterEach(() => {
      globalThis.fetch = realFetch;
    });

    it("forecasts today's Open-Meteo weather for every station", async () => {
      globalThis.fetch = (async () => new Response(JSON.stringify(meta.stations.map(() => openMeteo())))) as typeof fetch;
      const live = await rain.fetchLive();
      expect(live.readings.size).toBe(meta.stations.length);
      expect(live.dayView.stations.every((s) => s.probability !== null && s.actual === null)).toBe(true);
      const f = live.forecast("Sydney")!;
      expect(f.live).toBe(true);
      expect(f.actual).toBeNull();
      expect(f.correct).toBeNull();
      expect(f.probability).toBeGreaterThan(0);
      expect(f.probability).toBeLessThan(1);
    });
  });
});

/** A plausible Open-Meteo answer for one location: hourly values from yesterday to tomorrow, UTC+10. */
function openMeteo() {
  const offset = 36000;
  const local = new Date(Date.now() + offset * 1000);
  const day = (k: number) => new Date(Date.UTC(local.getUTCFullYear(), local.getUTCMonth(), local.getUTCDate() + k)).toISOString().slice(0, 10);
  const time = Array.from({ length: 72 }, (_, i) => `${day(Math.floor(i / 24) - 1)}T${String(i % 24).padStart(2, "0")}:00`);
  const fill = (f: (h: number) => number) => Array.from({ length: 72 }, (_, i) => f(i % 24));
  return {
    latitude: -33.87,
    longitude: 151.21,
    utc_offset_seconds: offset,
    timezone: "Australia/Sydney",
    hourly: {
      time,
      temperature_2m: fill((h) => 12 + 8 * Math.sin(((h - 9) / 24) * 2 * Math.PI)),
      relative_humidity_2m: fill((h) => 70 - h),
      pressure_msl: fill(() => 1012),
      cloud_cover: fill(() => 50),
      wind_speed_10m: fill(() => 15),
      wind_direction_10m: fill(() => 200),
      wind_gusts_10m: fill((h) => 30 + h),
      precipitation: fill(() => 0),
    },
    daily: { time: [day(-1), day(0), day(1)], sunshine_duration: [20000, 25000, 20000] },
  };
}

describe("fraud lab", () => {
  let lab: FraudLab;
  let meta: FraudLab["meta"];
  let v2: FraudPolicy;
  beforeAll(async () => {
    lab = await loadFraud({ fetcher });
    meta = lab.meta;
    v2 = { model: "v2", rule: "threshold", threshold: 0.5, reviewCost: meta.review_cost };
  }, LOAD);

  it("reproduces the published outcomes", () => {
    for (const [key, model] of [["v2_lightgbm", "v2"], ["v1_undersampled_logistic", "v1"]] as const) {
      const m = meta.models[key].test;
      const o = lab.evaluate({ ...v2, model });
      expect([o.tp, o.fp, o.fn, o.tn]).toEqual([m.tp, m.fp, m.fn, m.tn]);
      expect(o.net).toBeCloseTo(m.net_savings, 1);
      expect(o.alertsPerHour).toBeCloseTo(m.alerts_per_hour, 3);
      expect(o.precision!).toBeCloseTo(m.precision, 5);
      expect(o.recall!).toBeCloseTo(m.recall, 5);
    }
    const rule = meta.models.v2_lightgbm.test.expected_loss_rule;
    const o = lab.evaluate({ model: "v2", rule: "expected_loss", reviewCost: meta.review_cost });
    expect([o.tp, o.fp]).toEqual([rule.tp, rule.fp]);
    expect(o.net).toBeCloseTo(rule.net_savings, 1);
  });

  it("replays the stream in time order", () => {
    expect(lab.count).toBe(meta.splits.test.rows);
    expect(lab.indexAt(0)).toBe(0);
    expect(lab.indexAt(lab.hours + 1)).toBe(lab.count);
    const half = lab.evaluate(v2, lab.indexAt(lab.hours / 2));
    const all = lab.evaluate(v2);
    expect(half.seen).toBeLessThan(all.seen);
    expect(half.tp).toBeLessThanOrEqual(all.tp);
    expect(lab.evaluate(v2, 0).precision).toBeNull();
  });

  it("lists every fraud and alert as an event", () => {
    const events = lab.events(v2, lab.count, 10_000);
    const o = lab.evaluate(v2);
    expect(events.filter((e) => e.kind === "caught").length).toBe(o.tp);
    expect(events.filter((e) => e.kind === "missed").length).toBe(o.fn);
    expect(events.filter((e) => e.kind === "false_alarm").length).toBe(o.fp);
    events.slice(1).forEach((e, i) => expect(e.index).toBeLessThan(events[i].index));
    expect(lab.events(v2, lab.count, 5).length).toBe(5);
  });

  it("sweeps cut-offs from strict to lenient", () => {
    const { points, best } = lab.tradeoff("v2", 5);
    points.slice(1).forEach((p, i) => {
      expect(p.alerts).toBeGreaterThanOrEqual(points[i].alerts);
      expect(p.threshold).toBeLessThan(points[i].threshold);
    });
    expect(best.net).toBeGreaterThanOrEqual(lab.evaluate(v2).net - 1e-6);
    expect(points[points.length - 1].recall).toBeGreaterThan(0.9);
  });

  it("explains a fraud with contributions that add up", async () => {
    const fraudIndex = lab.stream.isFraud.indexOf(1);
    expect(lab.canExplain(fraudIndex)).toBe(true);
    const e = (await lab.explain(fraudIndex))!;
    const z = Math.log(e.probability / (1 - e.probability));
    // stream scores are stored to 0.001 in log-odds, clipped to +-15
    expect(Math.abs(Math.max(-15, Math.min(15, z)) - lab.stream.logit.v2[fraudIndex])).toBeLessThan(0.0011);
    expect(e.waterfall.end).toBeCloseTo(e.probability, 9);
    const base = Math.log(e.waterfall.start / (1 - e.waterfall.start));
    expect(base + e.contributions.reduce((s, c) => s + c.value, 0)).toBeCloseTo(z, 6);
    expect(e.contributions.map((c) => c.label)).toContain("Amount");
    expect(e.contributions.map((c) => c.label)).toContain("Time of day");
    expect(await lab.explain(-1)).toBeNull();
  });
});

describe("insurance calculator", () => {
  let calc: InsuranceCalculator;
  beforeAll(async () => {
    calc = await loadInsurance({ fetcher });
  }, LOAD);
  const fx = load<{ profiles: number[][]; prediction: number[]; shapley: number[][]; baseline: number }>("./fixtures/insurance.json");

  it("quotes match scikit-learn and the exact Shapley values", () => {
    const keys = ["age", "male", "bmi", "children", "smoker", "region"];
    fx.profiles.forEach((row, i) => {
      const q = calc.quote(profileFromRow(row));
      expect(q.charges).toBeCloseTo(fx.prediction[i], 6);
      expect(q.base).toBeCloseTo(fx.baseline, 6);
      for (const c of q.contributions) expect(c.value).toBeCloseTo(fx.shapley[i][keys.indexOf(c.key)], 6);
      expect(q.base + q.contributions.reduce((s, c) => s + c.value, 0)).toBeCloseTo(q.charges, 6);
      expect(q.range.lo).toBeLessThan(q.range.median);
      expect(q.range.median).toBeLessThan(q.range.hi);
    });
  });

  it("puts customers in the three published bands", () => {
    for (const [group, mean] of Object.entries(calc.meta.group_means)) {
      const members = calc.customers.filter((c) => c.group === group);
      expect(members.reduce((s, c) => s + c.charges, 0) / members.length).toBeCloseTo(mean, 0);
    }
    expect(calc.customers.length).toBe(calc.meta.rows);
  });

  it("charges rise with age", () => {
    const curve = calc.byAge({ age: 30, male: 1, bmi: 26, children: 0, smoker: 0, region: 0 });
    expect(curve.length).toBe(47);
    curve.slice(1).forEach((p, i) => expect(p.charges).toBeGreaterThan(curve[i].charges));
    const smoker = calc.quote({ age: 30, male: 1, bmi: 32, children: 0, smoker: 1, region: 2 });
    expect(smoker.group).toBe("smoker_bmi_30_plus");
    expect(smoker.contributions[0].key).toBe("smoker");
    expect(smoker.contributions.find((c) => c.key === "region")!.text).toBe("Southeast");
  });
});

describe("clustering playground", () => {
  let iris: IrisPlayground;
  let methods: IrisPlayground["meta"]["methods"];
  beforeAll(async () => {
    iris = await loadIris({ fetcher });
    methods = iris.meta.methods;
  }, LOAD);

  it("reproduces the published results when keeping the best of several starts", () => {
    for (const seed of [0, 1, 7]) {
      const km = iris.run({ method: "kmeans", k: 3, seed, restarts: 20 });
      expect(km.score.ari).toBeCloseTo(methods.kmeans_raw.ari, 5);
      expect(km.score.accuracy).toBeCloseTo(methods.kmeans_raw.matched_accuracy, 5);
      expect(km.tries.length).toBe(20);
      expect(iris.run({ method: "kmeans", k: 3, seed, scaled: true, restarts: 20 }).score.ari).toBeCloseTo(methods.kmeans_scaled.ari, 5);
      const gmm = iris.run({ method: "gmm", k: 3, seed, restarts: 10 });
      expect(gmm.score.ari).toBeCloseTo(methods.gmm_raw.ari, 5);
      expect(gmm.score.accuracy).toBeCloseTo(methods.gmm_raw.matched_accuracy, 5);
    }
  });

  it("single starts can land in different local optima", () => {
    const found = new Set(Array.from({ length: 8 }, (_, seed) => iris.run({ method: "kmeans", k: 3, seed }).score.ari.toFixed(4)));
    expect(found.size).toBeGreaterThan(1);
  });

  it("records one frame per iteration", () => {
    for (const method of ["kmeans", "gmm"] as const) {
      const run = iris.run({ method, k: 3, seed: 1 });
      expect(run.converged).toBe(true);
      run.frames.forEach((f, i) => {
        expect(f.iteration).toBe(i);
        expect(f.labels.length).toBe(150);
        expect(f.centers.length).toBe(3);
      });
      const last = run.frames[run.frames.length - 1];
      expect(run.score.ari).toBeCloseTo(iris.score(last.labels, 3).ari, 12);
    }
    const gmm = iris.run({ method: "gmm", k: 3, seed: 1 });
    expect(gmm.frames[0].covariances!.length).toBe(3);
    gmm.frames.slice(1).forEach((f, i) => expect(f.logLikelihood!).toBeGreaterThanOrEqual(gmm.frames[i].logLikelihood! - 1e-9));
  });

  it("keeps standardized k-means centres in cm", () => {
    const run = iris.run({ method: "kmeans", k: 3, seed: 2, scaled: true });
    for (const c of run.frames[run.frames.length - 1].centers) {
      expect(c[2]).toBeGreaterThan(1);
      expect(c[2]).toBeLessThan(7);
    }
  });

  it("projects and outlines in the 2-D views", () => {
    const pts = iris.points();
    expect(pts.length).toBe(150);
    const centre = iris.project(iris.data.pca.mean);
    expect(Math.abs(centre[0]) + Math.abs(centre[1])).toBeLessThan(1e-12);
    expect(iris.project(iris.X[0], [2, 3])).toEqual([iris.X[0][2], iris.X[0][3]]);
    const circle = iris.ellipse([0, 0, 1, 1], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]], [2, 3], 1);
    expect(circle.length).toBe(65);
    for (const [x, y] of circle) expect(Math.hypot(x - 1, y - 1)).toBeCloseTo(2, 9);
    const perfect = iris.score(iris.truth, 3);
    expect(perfect.ari).toBe(1);
    expect(perfect.accuracy).toBe(1);
    expect(iris.run({ method: "kmeans", k: 1, seed: 0 }).score.silhouette).toBeNull();
  });
});

describe("self-test", () => {
  it("passes every parity check", async () => {
    const checks = await selfTest({ fetcher, referenceFetcher });
    expect(checks.length).toBe(10);
    for (const c of checks) {
      expect(c.pass, `${c.project}: ${c.label} (max error ${c.maxError})`).toBe(true);
      expect(c.count).toBeGreaterThan(0);
    }
  }, LOAD);
});

describe("formatting and colours", () => {
  it("formats figures consistently", () => {
    expect(formatChance(0.996)).toBe("99.6%");
    expect(formatChance(0.42)).toBe("42%");
    expect(formatChance(0.00001)).toBe("<0.01%");
    expect(formatPoints(-0.123)).toBe("−12 pts");
    expect(formatSignedUsd(7803.39)).toBe("+$7,803");
    expect(formatSignedUsd(-10274.4)).toBe("−$10,274");
  });

  it("maps chances to the blue ramp", () => {
    expect(chanceColor(0)).not.toBe(chanceColor(1));
    expect(chanceColor(1, "dark")).not.toBe(chanceColor(1, "light"));
    expect(chanceGradient()).toMatch(/^linear-gradient\(to right, rgb/);
    expect(sigmoid(0)).toBe(0.5);
  });
});
