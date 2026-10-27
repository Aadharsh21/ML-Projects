import { describe, expect, it } from "vitest";
import { InsuranceModel, profileFromRow, quantileSorted, type InsuranceModelJSON } from "../src/ml/insurance";
import { load } from "./load";

describe("insurance model in the browser", () => {
  const model = new InsuranceModel(load<InsuranceModelJSON>("../public/data/insurance/model.json"));
  const data = load<{ rows: number[][] }>("../public/data/insurance/data.json");
  const background = data.rows.map(profileFromRow);
  const fx = load<{ profiles: number[][]; prediction: number[]; shapley: number[][]; baseline: number }>(
    "./fixtures/insurance.json",
  );

  it("matches scikit-learn's predictions", () => {
    fx.profiles.forEach((row, i) => expect(model.predict(profileFromRow(row))).toBeCloseTo(fx.prediction[i], 6));
  });

  it("allocation-free predict agrees with the design matrix", () => {
    const p = profileFromRow(fx.profiles[3]);
    const x = model.design(p);
    const dot = x.reduce((s, v, i) => s + v * model.json.coef[i], model.json.intercept);
    expect(model.predict(p)).toBeCloseTo(dot, 9);
  });

  it("computes the same exact Shapley values", () => {
    fx.profiles.forEach((row, i) => {
      const { phi, base } = model.shapley(profileFromRow(row), background);
      expect(base).toBeCloseTo(fx.baseline, 6);
      phi.forEach((v, j) => expect(v).toBeCloseTo(fx.shapley[i][j], 6));
    });
  });

  it("uses numpy's linear quantiles", () => {
    expect(quantileSorted([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(quantileSorted([0, 10], 0.1)).toBe(1);
    const r = model.range(profileFromRow(fx.profiles[0]));
    expect(r.lo).toBeLessThan(r.median);
    expect(r.median).toBeLessThan(r.hi);
  });
});
