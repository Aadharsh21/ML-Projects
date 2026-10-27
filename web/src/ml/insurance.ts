/** Insurance charges model (mirrors ml/insurance.py). */

export interface InsuranceModelJSON {
  features: string[];
  intercept: number;
  coef: number[];
  obese_bmi: number;
  regions: string[];
  inputs: string[];
  range: [number, number];
  residuals: { smoker: number[]; non_smoker: number[] };
}

export interface Profile {
  age: number;
  male: 0 | 1;
  bmi: number;
  children: number;
  smoker: 0 | 1;
  /** index into regions: northeast, northwest, southeast, southwest */
  region: number;
}

/** The six inputs, in the order Shapley values are reported. */
export const INPUT_KEYS = ["age", "male", "bmi", "children", "smoker", "region"] as const;
export type InputKey = (typeof INPUT_KEYS)[number];

export function profileFromRow(row: number[]): Profile {
  return {
    age: row[0],
    male: row[1] as 0 | 1,
    bmi: row[2],
    children: row[3],
    smoker: row[4] as 0 | 1,
    region: row[5],
  };
}

/** numpy.quantile's default (linear) method on a sorted array. */
export function quantileSorted(sorted: ArrayLike<number>, q: number): number {
  const h = (sorted.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

export class InsuranceModel {
  readonly json: InsuranceModelJSON;
  /** coefficient lookup by feature name (0 when a feature is absent) */
  private readonly c: Record<string, number>;

  constructor(json: InsuranceModelJSON) {
    this.json = json;
    this.c = Object.fromEntries(json.features.map((f, i) => [f, json.coef[i]]));
  }

  /** Design row in the order of json.features. */
  design(p: Profile): Float64Array {
    const obese = p.bmi >= this.json.obese_bmi ? 1 : 0;
    const v: Record<string, number> = {
      age: p.age,
      age_squared: p.age ** 2,
      bmi: p.bmi,
      children: p.children,
      smoker: p.smoker,
      male: p.male,
      region_northwest: p.region === 1 ? 1 : 0,
      region_southeast: p.region === 2 ? 1 : 0,
      region_southwest: p.region === 3 ? 1 : 0,
      obese,
      smoker_obese: p.smoker * obese,
      smoker_bmi: p.smoker * p.bmi,
    };
    return Float64Array.from(this.json.features, (f) => v[f]);
  }

  /** Same value as design() . coef + intercept, without allocating (used in loops). */
  predict(p: Profile): number {
    const c = this.c;
    const obese = p.bmi >= this.json.obese_bmi ? 1 : 0;
    return (
      this.json.intercept +
      c.age * p.age +
      c.age_squared * p.age ** 2 +
      c.bmi * p.bmi +
      c.children * p.children +
      c.smoker * p.smoker +
      c.male * p.male +
      (p.region === 1 ? c.region_northwest : 0) +
      (p.region === 2 ? c.region_southeast : 0) +
      (p.region === 3 ? c.region_southwest : 0) +
      c.obese * obese +
      c.smoker_obese * p.smoker * obese +
      c.smoker_bmi * p.smoker * p.bmi
    );
  }

  residuals(p: Profile): number[] {
    return p.smoker ? this.json.residuals.smoker : this.json.residuals.non_smoker;
  }

  /** Middle-80% range and median of likely charges for this profile. */
  range(p: Profile): { lo: number; median: number; hi: number } {
    const pred = this.predict(p);
    const r = this.residuals(p);
    const [qlo, qhi] = this.json.range;
    return {
      lo: pred + quantileSorted(r, qlo),
      median: pred + quantileSorted(r, 0.5),
      hi: pred + quantileSorted(r, qhi),
    };
  }

  /**
   * Exact Shapley values of the six inputs against the average customer, as in
   * ml/insurance.py: v(S) averages the prediction over the background customers
   * with the inputs in S replaced by this profile's values.
   */
  shapley(p: Profile, background: Profile[]): { phi: number[]; base: number } {
    const n = INPUT_KEYS.length;
    const value = new Float64Array(1 << n);
    const row: Profile = { ...p };
    for (let mask = 0; mask < 1 << n; mask++) {
      let sum = 0;
      for (const b of background) {
        for (let i = 0; i < n; i++) {
          const key = INPUT_KEYS[i];
          (row as unknown as Record<InputKey, number>)[key] = (mask >> i) & 1 ? p[key] : b[key];
        }
        sum += this.predict(row);
      }
      value[mask] = sum / background.length;
    }
    const fact = [1, 1, 2, 6, 24, 120, 720];
    const phi = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let mask = 0; mask < 1 << n; mask++) {
        if ((mask >> i) & 1) continue;
        let size = 0;
        for (let m = mask; m; m >>= 1) size += m & 1;
        const weight = (fact[size] * fact[n - size - 1]) / fact[n];
        phi[i] += weight * (value[mask | (1 << i)] - value[mask]);
      }
    }
    return { phi, base: value[0] };
  }
}
