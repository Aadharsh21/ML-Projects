/**
 * The clustering playground: run k-means or a Gaussian mixture on the 150 iris flowers
 * one iteration at a time, see each step in a 2-D view, and score the result against
 * the species the algorithm never saw.
 */
import {
  adjustedRandIndex,
  gmmLabels,
  gmmStart,
  gmmStep,
  kmeans,
  kmeansPlusPlus,
  kmeansStart,
  kmeansStep,
  inertia,
  matchClusters,
  mStep,
  mulberry32,
  silhouette,
  type GMMState,
  type Matrix,
} from "../ml/clustering.js";
import { loadFiles, type LoadOptions } from "./load.js";

export interface IrisData {
  features: string[];
  species: string[];
  /** four measurements in cm, then the species index */
  rows: number[][];
  pca: { mean: number[]; components: number[][] };
}

export interface IrisSweepPoint {
  k: number;
  inertia: number;
  silhouette: number | null;
  ari: number;
}

export interface IrisMeta {
  features: string[];
  species: string[];
  /** k-means for k = 1..8 on raw and standardized measurements */
  sweep: { raw: IrisSweepPoint[]; scaled: IrisSweepPoint[] };
  gmm_bic: { k: number; bic: number }[];
  /** the published comparison; `matrix` is species (rows) by cluster (columns) */
  methods: Record<string, { label: string; ari: number; matrix: number[][]; matched_accuracy: number }>;
  pca_explained_variance: number[];
}

/** "pca" for the first two principal components, or two measurement indices such as [2, 3]. */
export type IrisView = "pca" | [number, number];

export interface IrisFrame {
  iteration: number;
  /** cluster of each flower */
  labels: number[];
  /** cluster centres in cm */
  centers: number[][];
  /** Gaussian mixture only: covariance (cm²) and weight of each component */
  covariances?: number[][][];
  weights?: number[];
  /** k-means only: within-cluster sum of squares, in the space it clustered in */
  inertia?: number;
  /** Gaussian mixture only: mean log-likelihood per flower */
  logLikelihood?: number;
}

export interface IrisScore {
  /** adjusted Rand index against the species: 1 is perfect, 0 is chance */
  ari: number;
  /** share of flowers in their species' cluster under the best one-to-one matching */
  accuracy: number;
  /** on the raw measurements; null when there is only one cluster */
  silhouette: number | null;
  /** flowers per species (rows) and cluster (columns) */
  counts: number[][];
  /** cluster matched to each species, or -1 */
  mapping: number[];
}

export interface IrisRunOptions {
  method: "kmeans" | "gmm";
  k: number;
  /** seeds the k-means++ starting points (default 0) */
  seed?: number;
  /** k-means only: cluster on standardized measurements instead of cm */
  scaled?: boolean;
  /** run from this many starting points (seed, seed + 1, ...) and keep the best, as
   * scikit-learn's n_init does: lowest inertia for k-means, highest likelihood for a mixture */
  restarts?: number;
  maxIter?: number;
}

export interface IrisRun extends Required<Omit<IrisRunOptions, "maxIter">> {
  /** the starting state, then one frame per iteration (of the best start) */
  frames: IrisFrame[];
  converged: boolean;
  score: IrisScore;
  /** seed of the kept start */
  bestSeed: number;
  /** every start tried: inertia (k-means) or mean log-likelihood (mixture) and ARI */
  tries: { seed: number; objective: number; ari: number }[];
}

export interface IrisPlayground {
  data: IrisData;
  meta: IrisMeta;
  features: string[];
  species: string[];
  /** the 150 flowers' measurements in cm */
  X: number[][];
  /** species index of each flower */
  truth: number[];
  /** share of variance on the two principal components */
  explained: number[];
  /** a point (or centre) in the 2-D view */
  project(x: number[], view?: IrisView): [number, number];
  points(view?: IrisView): [number, number][];
  /** outline of a Gaussian component in the view, `radius` standard deviations out */
  ellipse(mean: number[], cov: number[][], view?: IrisView, radius?: number, segments?: number): [number, number][];
  run(options: IrisRunOptions): IrisRun;
  score(labels: ArrayLike<number>, k: number): IrisScore;
}

function standardizer(X: Matrix) {
  const d = X[0].length;
  const mean = new Array<number>(d).fill(0);
  for (const x of X) for (let j = 0; j < d; j++) mean[j] += x[j] / X.length;
  const std = mean.map((m, j) => Math.sqrt(X.reduce((s, x) => s + (x[j] - m) ** 2, 0) / X.length));
  return {
    to: (x: number[]) => x.map((v, j) => (v - mean[j]) / std[j]),
    from: (z: number[]) => z.map((v, j) => v * std[j] + mean[j]),
  };
}

const oneHot = (labels: ArrayLike<number>, k: number): Matrix => Array.from(labels, (l) => Array.from({ length: k }, (_, c) => (c === l ? 1 : 0)));

export async function loadIris(options: LoadOptions = {}): Promise<IrisPlayground> {
  const { data, meta } = await loadFiles<{ data: IrisData; meta: IrisMeta }>(
    { data: "iris/data.json", meta: "iris/meta.json" },
    options,
    1,
    "Downloading the flowers",
  );
  const X = data.rows.map((r) => r.slice(0, 4));
  const truth = data.rows.map((r) => r[4]);
  const { mean: pcaMean, components } = data.pca;

  const project = (x: number[], view: IrisView = "pca"): [number, number] => {
    if (view !== "pca") return [x[view[0]], x[view[1]]];
    const c = x.map((v, j) => v - pcaMean[j]);
    return [components[0].reduce((s, w, j) => s + w * c[j], 0), components[1].reduce((s, w, j) => s + w * c[j], 0)];
  };

  const project2x2 = (cov: number[][], view: IrisView): [number, number, number] => {
    if (view !== "pca") return [cov[view[0]][view[0]], cov[view[0]][view[1]], cov[view[1]][view[1]]];
    const [w0, w1] = components;
    const q = (a: number[], b: number[]) => a.reduce((s, ai, i) => s + ai * b.reduce((t, bj, j) => t + cov[i][j] * bj, 0), 0);
    return [q(w0, w0), q(w0, w1), q(w1, w1)];
  };

  const score = (labels: ArrayLike<number>, k: number): IrisScore => {
    const m = matchClusters(truth, labels, data.species.length, k);
    const s = k > 1 ? silhouette(X, labels) : NaN;
    return { ari: adjustedRandIndex(truth, labels), accuracy: m.accuracy, silhouette: Number.isNaN(s) ? null : s, counts: m.counts, mapping: m.mapping };
  };

  // greedy k-means++ with scikit-learn's number of candidates per centre
  const seeds = (space: Matrix, k: number, rand: () => number) => kmeansPlusPlus(space, k, rand, 2 + Math.floor(Math.log(k)));

  const runOnce = (method: "kmeans" | "gmm", k: number, seed: number, scaled: boolean, maxIter: number) => {
    const rand = mulberry32(seed);
    const frames: IrisFrame[] = [];
    let converged = false;
    let labels: Int32Array;
    let objective: number;
    if (method === "kmeans") {
      const z = scaled ? standardizer(X) : null;
      const space = z ? X.map(z.to) : X;
      const back = (c: number[]) => (z ? z.from(c) : c);
      let s = kmeansStart(space, seeds(space, k, rand));
      const push = () =>
        frames.push({ iteration: s.iteration, labels: Array.from(s.labels), centers: s.centers.map(back), inertia: inertia(space, s.labels, s.centers) });
      push();
      while (!s.converged && s.iteration < maxIter) {
        s = kmeansStep(space, s);
        push();
      }
      converged = s.converged;
      labels = s.labels;
      objective = frames[frames.length - 1].inertia!;
    } else {
      // initialise like scikit-learn (init_params="kmeans"): parameters from a k-means clustering
      const init = kmeans(X, seeds(X, k, rand));
      const p = mStep(X, oneHot(init.labels, k));
      let s: GMMState = gmmStart(X, p.means, p.covariances, p.weights);
      const push = () =>
        frames.push({
          iteration: s.iteration,
          labels: Array.from(gmmLabels(s)),
          centers: s.means.map((m) => [...m]),
          covariances: s.covariances,
          weights: s.weights,
          logLikelihood: s.logLikelihood,
        });
      push();
      while (s.iteration < maxIter) {
        const next = gmmStep(X, s);
        const change = Math.abs(next.logLikelihood - s.logLikelihood);
        s = next;
        push();
        if (change < 1e-3) {
          converged = true;
          break;
        }
      }
      labels = gmmLabels(s);
      objective = s.logLikelihood;
    }
    return { seed, frames, converged, objective, score: score(labels, k) };
  };

  return {
    data,
    meta,
    features: data.features,
    species: data.species,
    X,
    truth,
    explained: meta.pca_explained_variance,
    project,
    points: (view = "pca") => X.map((x) => project(x, view)),
    ellipse(mean, cov, view = "pca", radius = 2, segments = 64) {
      const [cx, cy] = project(mean, view);
      const [a, b, c] = project2x2(cov, view);
      // eigen-decomposition of the symmetric 2x2 [[a, b], [b, c]]
      const mid = (a + c) / 2;
      const spread = Math.sqrt(((a - c) / 2) ** 2 + b * b);
      const l1 = Math.max(mid + spread, 0);
      const l2 = Math.max(mid - spread, 0);
      const angle = 0.5 * Math.atan2(2 * b, a - c);
      const [cos, sin] = [Math.cos(angle), Math.sin(angle)];
      return Array.from({ length: segments + 1 }, (_, i) => {
        const t = (2 * Math.PI * i) / segments;
        const u = radius * Math.sqrt(l1) * Math.cos(t);
        const v = radius * Math.sqrt(l2) * Math.sin(t);
        return [cx + u * cos - v * sin, cy + u * sin + v * cos] as [number, number];
      });
    },
    run({ method, k, seed = 0, scaled = false, restarts = 1, maxIter = 100 }) {
      let best: ReturnType<typeof runOnce> | null = null;
      const tries: IrisRun["tries"] = [];
      for (let r = 0; r < Math.max(1, restarts); r++) {
        const one = runOnce(method, k, seed + r, method === "kmeans" && scaled, maxIter);
        tries.push({ seed: seed + r, objective: one.objective, ari: one.score.ari });
        const better = !best || (method === "kmeans" ? one.objective < best.objective : one.objective > best.objective);
        if (better) best = one;
      }
      const b = best!;
      return {
        method,
        k,
        seed,
        scaled: method === "kmeans" && scaled,
        restarts: Math.max(1, restarts),
        frames: b.frames,
        converged: b.converged,
        score: b.score,
        bestSeed: b.seed,
        tries,
      };
    },
    score,
  };
}
