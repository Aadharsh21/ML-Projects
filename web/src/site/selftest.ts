/**
 * Checks, in the visitor's own browser, that the engine reproduces the Python models:
 * it reruns the reference rows exported by the Python side (web/tests/fixtures) and
 * reports the largest difference for each check.
 */
import { gmmLabels, gmmStart, gmmStep, kmeans, silhouette, adjustedRandIndex } from "../ml/clustering.js";
import { profileFromRow, InsuranceModel, type InsuranceModelJSON } from "../ml/insurance.js";
import { Forest, type ForestJSON } from "../ml/lgbm.js";
import { RainDays, rainFeatures, type DaysJSON, type RainPreprocessing } from "../ml/rain.js";
import { REFERENCE_URL, fetchJSON, loadFiles, type Fetcher, type LoadOptions } from "./load.js";

export interface ParityCheck {
  project: "Rain" | "Fraud" | "Insurance" | "Clustering";
  /** what is being compared, in plain words */
  label: string;
  /** how many values were compared */
  count: number;
  /** largest absolute difference from the Python result */
  maxError: number;
  tolerance: number;
  pass: boolean;
}

export interface SelfTestOptions extends LoadOptions {
  /** base URL of the reference files (defaults to web/tests/fixtures next to this module) */
  referenceUrl?: string;
  /** replaces fetching the reference files, for tests */
  referenceFetcher?: Fetcher;
}

/** Largest |a - b| over paired values, treating NaN/null as equal to each other and infinitely far from numbers. */
function maxDiff(a: ArrayLike<number>, b: ArrayLike<number | null>): number {
  let worst = 0;
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i] === null || b[i] === undefined ? NaN : (b[i] as number);
    if (Number.isNaN(x) && Number.isNaN(y)) continue;
    const d = Math.abs(x - y);
    worst = Number.isNaN(d) ? Infinity : Math.max(worst, d);
  }
  return worst;
}

const check = (project: ParityCheck["project"], label: string, count: number, maxError: number, tolerance: number): ParityCheck => ({
  project, label, count, maxError, tolerance, pass: maxError <= tolerance,
});

/** Run every parity check. Downloads about 4 MB the first time (less if the pages already loaded their models). */
export async function selfTest(options: SelfTestOptions = {}): Promise<ParityCheck[]> {
  const readRef = options.referenceFetcher ?? ((name: string) => fetchJSON(new URL(name, options.referenceUrl ?? REFERENCE_URL).href));
  const stage = "Downloading the models and Python's reference results";
  const [files, ref] = await Promise.all([
    loadFiles<{
      rainModel: ForestJSON & { preprocessing: RainPreprocessing };
      rainDays: DaysJSON;
      fraudModel: ForestJSON;
      insModel: InsuranceModelJSON;
      insData: { rows: number[][] };
      iris: { rows: number[][] };
    }>(
      {
        rainModel: "rain/model.json",
        rainDays: "rain/days.json",
        fraudModel: "fraud/model.json",
        insModel: "insurance/model.json",
        insData: "insurance/data.json",
        iris: "iris/data.json",
      },
      options,
      0.9,
      stage,
    ),
    Promise.all(["rain.json", "fraud.json", "insurance.json", "iris.json"].map((n) => readRef(n))) as Promise<any[]>,
  ]);
  const [rainRef, fraudRef, insRef, irisRef] = ref;
  options.onProgress?.(0.9, "Comparing with Python");
  const out: ParityCheck[] = [];

  // rain: features rebuilt from the raw readings, then LightGBM's probabilities and SHAP values
  const rain = new Forest(files.rainModel);
  const days = new RainDays(files.rainDays);
  const startMs = Date.parse(`${days.start}T00:00:00Z`);
  const rainX = (rainRef.rows as { station: string; date: string }[]).map((r) => {
    const row = days.rowFor(r.station, (Date.parse(`${r.date}T00:00:00Z`) - startMs) / 86_400_000);
    if (row === undefined) throw new Error(`reference row ${r.station} ${r.date} not in the test days`);
    return rainFeatures(days.observation(row), files.rainModel.preprocessing, rain.features, days.start);
  });
  out.push(check("Rain", "Features rebuilt from the raw readings", rainX.length * rain.features.length, Math.max(...rainX.map((x, i) => maxDiff(x, rainRef.features[i]))), 1e-12));
  out.push(check("Rain", "LightGBM probabilities", rainX.length, maxDiff(rainX.map((x) => rain.predictProba(x)), rainRef.probability), 1e-9));
  out.push(check("Rain", "SHAP explanations (TreeSHAP)", rainX.length * (rain.features.length + 1), Math.max(...rainX.map((x, i) => maxDiff(rain.shap(x), rainRef.shap[i]))), 1e-8));

  // fraud: LightGBM probabilities and SHAP values on the reference transactions
  const fraud = new Forest(files.fraudModel);
  const fx = fraudRef.features as number[][];
  out.push(check("Fraud", "LightGBM probabilities", fx.length, maxDiff(fx.map((x) => fraud.predictProba(x)), fraudRef.probability), 1e-9));
  out.push(check("Fraud", "SHAP explanations (TreeSHAP)", fx.length * (fraud.features.length + 1), Math.max(...fx.map((x, i) => maxDiff(fraud.shap(x), fraudRef.shap[i]))), 1e-8));

  // insurance: predictions and exact Shapley values against the 1,338 customers
  const ins = new InsuranceModel(files.insModel);
  const background = files.insData.rows.map(profileFromRow);
  const profiles = (insRef.profiles as number[][]).map(profileFromRow);
  out.push(check("Insurance", "Predicted charges", profiles.length, maxDiff(profiles.map((p) => ins.predict(p)), insRef.prediction), 1e-6));
  const shap = profiles.map((p) => ins.shapley(p, background));
  out.push(
    check(
      "Insurance",
      "Exact Shapley values",
      profiles.length * 7,
      Math.max(...shap.map((s, i) => Math.max(maxDiff(s.phi, insRef.shapley[i]), Math.abs(s.base - insRef.baseline)))),
      1e-6,
    ),
  );

  // clustering: k-means and EM from scikit-learn's starting points, then the scores
  const X = files.iris.rows.map((r) => r.slice(0, 4));
  const species = files.iris.rows.map((r) => r[4]);
  let kmErr = 0;
  let kmCount = 0;
  for (const run of irisRef.kmeans) {
    const s = kmeans(X, run.init);
    s.centers.forEach((c, k) => (kmErr = Math.max(kmErr, maxDiff(c, run.centers[k]))));
    if (Array.from(s.labels).some((l, i) => l !== run.labels[i])) kmErr = Infinity;
    kmCount += s.centers.length * 4;
  }
  out.push(check("Clustering", "k-means centres", kmCount, kmErr, 1e-9));
  let emErr = 0;
  let emCount = 0;
  for (const run of irisRef.gmm) {
    let s = gmmStart(X, run.init_means, run.init_cov);
    for (let i = 0; i < run.steps; i++) s = gmmStep(X, s);
    emErr = Math.max(emErr, maxDiff(s.weights, run.weights));
    s.means.forEach((m, k) => (emErr = Math.max(emErr, maxDiff(m, run.means[k]))));
    s.covariances.forEach((c, k) => c.forEach((row, a) => (emErr = Math.max(emErr, maxDiff(row, run.covariances[k][a])))));
    if (Array.from(gmmLabels(s)).some((l, i) => l !== run.labels[i])) emErr = Infinity;
    emCount += s.weights.length * (1 + 4 + 16);
  }
  out.push(check("Clustering", "Gaussian mixture, step by step", emCount, emErr, 1e-7));
  out.push(
    check(
      "Clustering",
      "Silhouette and adjusted Rand index",
      2,
      Math.max(Math.abs(silhouette(X, irisRef.labels_k3) - irisRef.silhouette_k3), Math.abs(adjustedRandIndex(species, irisRef.labels_k3) - irisRef.ari_k3)),
      1e-10,
    ),
  );
  options.onProgress?.(1, "Done");
  return out;
}
