import { readFileSync } from "node:fs";

export const load = <T = any>(path: string): T =>
  JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8")) as T;

/** Element-wise closeness with NaN == NaN. */
export function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number | null>, tol: number) {
  if (actual.length !== expected.length) throw new Error(`length ${actual.length} != ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    const e = expected[i] === null ? NaN : (expected[i] as number);
    const a = actual[i];
    if (Number.isNaN(e) && Number.isNaN(a)) continue;
    if (!(Math.abs(a - e) <= tol)) throw new Error(`index ${i}: ${a} vs ${e} (tol ${tol})`);
  }
}
