import { describe, expect, it } from "vitest";
import { Forest } from "../src/ml/lgbm";
import { RainDays, rainFeatures, type RainMeta } from "../src/ml/rain";
import { expectClose, load } from "./load";

describe("rain model in the browser", () => {
  const modelJson = load("../public/data/rain/model.json");
  const forest = new Forest(modelJson);
  const meta = load<RainMeta>("../public/data/rain/meta.json");
  const days = new RainDays(load("../public/data/rain/days.json"));
  const fx = load<{ rows: { station: string; date: string }[]; features: (number | null)[][]; probability: number[]; shap: number[][] }>("./fixtures/rain.json");
  const startMs = Date.parse(`${days.start}T00:00:00Z`);
  const rows = fx.rows.map((r) => days.rowFor(r.station, (Date.parse(`${r.date}T00:00:00Z`) - startMs) / 86_400_000)!);
  const features = rows.map((r) => rainFeatures(days.observation(r), modelJson.preprocessing, forest.features, days.start));

  it("builds the same features as Python", () => {
    expect(rows.every((r) => r !== undefined)).toBe(true);
    expect(forest.features).toEqual(meta.features);
    features.forEach((f, i) => expectClose(f, fx.features[i], 1e-12));
  });

  it("predicts the same probabilities as LightGBM", () => {
    features.forEach((f, i) => expect(forest.predictProba(f)).toBeCloseTo(fx.probability[i], 9));
  });

  it("reproduces LightGBM's SHAP values", () => {
    features.forEach((f, i) => expectClose(forest.shap(f), fx.shap[i], 1e-8));
  });

  it("SHAP values add up to the raw score", () => {
    const phi = forest.shap(features[0]);
    const sum = phi.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(forest.predictRaw(features[0]), 9);
    expect(phi[phi.length - 1]).toBeCloseTo(forest.baseValue, 12);
  });
});

describe("fraud model in the browser", () => {
  const forest = new Forest(load("../public/data/fraud/model.json"));
  const fx = load<{ features: number[][]; probability: number[]; shap: number[][] }>("./fixtures/fraud.json");

  it("predicts the same probabilities as LightGBM", () => {
    fx.features.forEach((f, i) => expect(forest.predictProba(f)).toBeCloseTo(fx.probability[i], 9));
  });

  it("reproduces LightGBM's SHAP values", () => {
    fx.features.forEach((f, i) => expectClose(forest.shap(f), fx.shap[i], 1e-8));
  });
});
