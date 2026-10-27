import { describe, expect, it } from "vitest";
import { Forest } from "../src/ml/lgbm";
import { Stream, fraudFeatures } from "../src/ml/fraud";
import { load } from "./load";

const stream = new Stream(load("../public/data/fraud/stream.json"));
const meta = load<any>("../public/data/fraud/meta.json");

describe("fraud stream", () => {
  it("decodes the test window", () => {
    expect(stream.n).toBe(meta.splits.test.rows);
    expect(stream.fraudCount).toBe(meta.splits.test.frauds);
    expect(stream.fraudAmount).toBeCloseTo(meta.test_fraud_amount, 1);
  });

  for (const [key, model] of [["v2_lightgbm", "v2"], ["v1_undersampled_logistic", "v1"]] as const) {
    it(`reproduces the ${model} confusion matrix and savings at 0.5`, () => {
      const m = meta.models[key].test;
      const o = stream.outcome(stream.flagged(model, { kind: "threshold", probability: 0.5 }), meta.review_cost);
      expect([o.tp, o.fp, o.fn, o.tn]).toEqual([m.tp, m.fp, m.fn, m.tn]);
      expect(o.blocked).toBeCloseTo(m.blocked, 1);
      expect(o.net).toBeCloseTo(m.net_savings, 1);
    });
  }

  it("reproduces the expected-loss rule", () => {
    const m = meta.models.v2_lightgbm.test.expected_loss_rule;
    const o = stream.outcome(stream.flagged("v2", { kind: "expected_loss", reviewCost: meta.review_cost }), meta.review_cost);
    expect([o.tp, o.fp]).toEqual([m.tp, m.fp]);
    expect(o.net).toBeCloseTo(m.net_savings, 1);
  });

  it("curve ends at full recall with every transaction flagged", () => {
    const c = stream.curve("v2", 5);
    const last = c[c.length - 1];
    expect(last.recall).toBe(1);
    expect(last.alerts).toBe(stream.n);
  });

  it("scores inspectable transactions like Python did", () => {
    const forest = new Forest(load("../public/data/fraud/model.json"));
    const inspect = load<{ index: number[]; features: number[][] }>("../public/data/fraud/inspect.json");
    inspect.index.forEach((row, k) => {
      const z = forest.predictRaw(inspect.features[k]);
      // stream scores are stored to 0.001 in log-odds
      expect(Math.abs(Math.max(-15, Math.min(15, z)) - stream.logit.v2[row])).toBeLessThan(0.0011);
    });
  });

  it("rebuilds features from a raw transaction", () => {
    const fx = load<{ raw: { time: number; amount: number }[]; features: number[][] }>("./fixtures/fraud.json");
    fx.raw.forEach((r, i) => {
      const f = fraudFeatures(fx.features[i].slice(0, 28), r.amount, r.time);
      f.forEach((v, j) => expect(v).toBeCloseTo(fx.features[i][j], 12));
    });
  });
});
