import { describe, expect, it } from "vitest";
import * as built from "../dist/index.js";
import * as source from "../src/index";
import { load } from "./load";

// The committed build in dist/ is what websites import from the CDN, so it must be
// the current source compiled (CI also rebuilds it and fails on any difference).
describe("built package in dist/", () => {
  it("exports the same names as the source", () => {
    expect(Object.keys(built).sort()).toEqual(Object.keys(source).sort());
  });

  it("carries the package version", () => {
    expect(built.VERSION).toBe(load<{ version: string }>("../package.json").version);
  });

  it("runs as plain ES modules and gives the same answers", async () => {
    const fetcher = async (path: string) => load(`../public/data/${path}`);
    const fromDist = await built.loadInsurance({ fetcher });
    const fromSource = await source.loadInsurance({ fetcher });
    const p = { age: 41, male: 0, bmi: 33.5, children: 2, smoker: 1, region: 1 } as const;
    expect(fromDist.quote(p)).toEqual(fromSource.quote(p));
    expect(built.DATA_URL.endsWith("/web/public/data/")).toBe(true);
  });
});
