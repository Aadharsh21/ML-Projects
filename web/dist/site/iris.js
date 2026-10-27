/**
 * The clustering playground: run k-means or a Gaussian mixture on the 150 iris flowers
 * one iteration at a time, see each step in a 2-D view, and score the result against
 * the species the algorithm never saw.
 */
import { adjustedRandIndex, gmmLabels, gmmStart, gmmStep, kmeans, kmeansPlusPlus, kmeansStart, kmeansStep, inertia, matchClusters, mStep, mulberry32, silhouette, } from "../ml/clustering.js";
import { loadFiles } from "./load.js";
function standardizer(X) {
    const d = X[0].length;
    const mean = new Array(d).fill(0);
    for (const x of X)
        for (let j = 0; j < d; j++)
            mean[j] += x[j] / X.length;
    const std = mean.map((m, j) => Math.sqrt(X.reduce((s, x) => s + (x[j] - m) ** 2, 0) / X.length));
    return {
        to: (x) => x.map((v, j) => (v - mean[j]) / std[j]),
        from: (z) => z.map((v, j) => v * std[j] + mean[j]),
    };
}
const oneHot = (labels, k) => Array.from(labels, (l) => Array.from({ length: k }, (_, c) => (c === l ? 1 : 0)));
export async function loadIris(options = {}) {
    const { data, meta } = await loadFiles({ data: "iris/data.json", meta: "iris/meta.json" }, options, 1, "Downloading the flowers");
    const X = data.rows.map((r) => r.slice(0, 4));
    const truth = data.rows.map((r) => r[4]);
    const { mean: pcaMean, components } = data.pca;
    const project = (x, view = "pca") => {
        if (view !== "pca")
            return [x[view[0]], x[view[1]]];
        const c = x.map((v, j) => v - pcaMean[j]);
        return [components[0].reduce((s, w, j) => s + w * c[j], 0), components[1].reduce((s, w, j) => s + w * c[j], 0)];
    };
    const project2x2 = (cov, view) => {
        if (view !== "pca")
            return [cov[view[0]][view[0]], cov[view[0]][view[1]], cov[view[1]][view[1]]];
        const [w0, w1] = components;
        const q = (a, b) => a.reduce((s, ai, i) => s + ai * b.reduce((t, bj, j) => t + cov[i][j] * bj, 0), 0);
        return [q(w0, w0), q(w0, w1), q(w1, w1)];
    };
    const score = (labels, k) => {
        const m = matchClusters(truth, labels, data.species.length, k);
        const s = k > 1 ? silhouette(X, labels) : NaN;
        return { ari: adjustedRandIndex(truth, labels), accuracy: m.accuracy, silhouette: Number.isNaN(s) ? null : s, counts: m.counts, mapping: m.mapping };
    };
    // greedy k-means++ with scikit-learn's number of candidates per centre
    const seeds = (space, k, rand) => kmeansPlusPlus(space, k, rand, 2 + Math.floor(Math.log(k)));
    const runOnce = (method, k, seed, scaled, maxIter) => {
        const rand = mulberry32(seed);
        const frames = [];
        let converged = false;
        let labels;
        let objective;
        if (method === "kmeans") {
            const z = scaled ? standardizer(X) : null;
            const space = z ? X.map(z.to) : X;
            const back = (c) => (z ? z.from(c) : c);
            let s = kmeansStart(space, seeds(space, k, rand));
            const push = () => frames.push({ iteration: s.iteration, labels: Array.from(s.labels), centers: s.centers.map(back), inertia: inertia(space, s.labels, s.centers) });
            push();
            while (!s.converged && s.iteration < maxIter) {
                s = kmeansStep(space, s);
                push();
            }
            converged = s.converged;
            labels = s.labels;
            objective = frames[frames.length - 1].inertia;
        }
        else {
            // initialise like scikit-learn (init_params="kmeans"): parameters from a k-means clustering
            const init = kmeans(X, seeds(X, k, rand));
            const p = mStep(X, oneHot(init.labels, k));
            let s = gmmStart(X, p.means, p.covariances, p.weights);
            const push = () => frames.push({
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
                return [cx + u * cos - v * sin, cy + u * sin + v * cos];
            });
        },
        run({ method, k, seed = 0, scaled = false, restarts = 1, maxIter = 100 }) {
            let best = null;
            const tries = [];
            for (let r = 0; r < Math.max(1, restarts); r++) {
                const one = runOnce(method, k, seed + r, method === "kmeans" && scaled, maxIter);
                tries.push({ seed: seed + r, objective: one.objective, ari: one.score.ari });
                const better = !best || (method === "kmeans" ? one.objective < best.objective : one.objective > best.objective);
                if (better)
                    best = one;
            }
            const b = best;
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
