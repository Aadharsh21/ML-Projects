import { describe, expect, it } from "vitest";
import { sentence, waterfall } from "../src/ml/explain";
import { sigmoid } from "../src/ml/lgbm";

describe("forecast explanations", () => {
  const groups = [
    { label: "Humidity at 3pm", value: 1.2 },
    { label: "Sunshine", value: 0.4 },
    { label: "Wind gust speed", value: -0.6 },
    { label: "Pressure at 3pm", value: 0.05 },
    { label: "Evaporation", value: -0.02 },
  ];

  it("walks from the base rate to the forecast, biggest effects first", () => {
    const wf = waterfall(-1, groups, 3);
    expect(wf.start).toBeCloseTo(sigmoid(-1), 12);
    expect(wf.end).toBeCloseTo(sigmoid(-1 + 1.2 + 0.4 - 0.6 + 0.05 - 0.02), 12);
    expect(wf.steps.map((s) => s.label)).toEqual(["Humidity at 3pm", "Wind gust speed", "Sunshine", "2 other readings"]);
    wf.steps.slice(1).forEach((s, i) => expect(s.from).toBeCloseTo(wf.steps[i].to, 12));
  });

  it("names the readings that push toward the outcome", () => {
    const wf = waterfall(0, groups, 6);
    const text = sentence(wf.steps, wf.end, (l) => (l === "Humidity at 3pm" ? "85%" : null));
    expect(text).toBe("Rain looks likely mainly because of humidity at 3pm (85%) and sunshine.");
  });
});
