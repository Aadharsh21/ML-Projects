import { describe, expect, it } from "vitest";
import {
  adjustedRandIndex,
  gmmLabels,
  gmmStart,
  gmmStep,
  kmeans,
  kmeansPlusPlus,
  matchClusters,
  mulberry32,
  silhouette,
} from "../src/ml/clustering";
import { load } from "./load";

const data = load<{ rows: number[][] }>("../public/data/iris/data.json");
const X = data.rows.map((r) => r.slice(0, 4));
const species = data.rows.map((r) => r[4]);
const fx = load<any>("./fixtures/iris.json");

describe("k-means", () => {
  it("converges to the same centers as scikit-learn from the same start", () => {
    for (const run of fx.kmeans) {
      const s = kmeans(X, run.init);
      s.centers.forEach((c: number[], k: number) => c.forEach((v, j) => expect(v).toBeCloseTo(run.centers[k][j], 9)));
      expect(Array.from(s.labels)).toEqual(run.labels);
    }
  });

  it("k-means++ picks distinct starting points", () => {
    const init = kmeansPlusPlus(X, 5, mulberry32(1));
    expect(new Set(init.map((c) => c.join())).size).toBe(5);
  });
});

describe("Gaussian mixture EM", () => {
  it("matches scikit-learn's GaussianMixture step for step", () => {
    for (const run of fx.gmm) {
      let s = gmmStart(X, run.init_means, run.init_cov);
      for (let i = 0; i < run.steps; i++) s = gmmStep(X, s);
      s.weights.forEach((w, k) => expect(w).toBeCloseTo(run.weights[k], 8));
      s.means.forEach((m, k) => m.forEach((v, j) => expect(v).toBeCloseTo(run.means[k][j], 7)));
      s.covariances.forEach((c, k) =>
        c.forEach((row, a) => row.forEach((v, b) => expect(v).toBeCloseTo(run.covariances[k][a][b], 7))),
      );
      expect(Array.from(gmmLabels(s))).toEqual(run.labels);
    }
  });
});

describe("cluster scores", () => {
  it("silhouette and ARI match scikit-learn", () => {
    expect(silhouette(X, fx.labels_k3)).toBeCloseTo(fx.silhouette_k3, 10);
    expect(adjustedRandIndex(species, fx.labels_k3)).toBeCloseTo(fx.ari_k3, 10);
  });

  it("ARI ignores label names", () => {
    expect(adjustedRandIndex([0, 0, 1, 1], [5, 5, 2, 2])).toBe(1);
  });

  it("matches clusters to species optimally", () => {
    const m = matchClusters([0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1], 3, 2);
    expect(m.accuracy).toBeCloseTo(5 / 7, 12);
    const perfect = matchClusters([0, 1, 2], [2, 0, 1], 3, 3);
    expect(perfect.mapping).toEqual([2, 0, 1]);
    expect(perfect.accuracy).toBe(1);
  });
});
