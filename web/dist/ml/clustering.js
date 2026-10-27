/**
 * k-means (Lloyd's algorithm) and Gaussian mixture EM, written to step one
 * iteration at a time so the playground can animate them. The EM step follows
 * scikit-learn's GaussianMixture (full covariance) so results can be checked
 * against it.
 */
export function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
export function sqDist(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
/* ------------------------------------------------------------------ k-means */
export function assign(X, centers) {
    const labels = new Int32Array(X.length);
    for (let n = 0; n < X.length; n++) {
        let best = 0;
        let bestD = Infinity;
        for (let k = 0; k < centers.length; k++) {
            const d = sqDist(X[n], centers[k]);
            if (d < bestD) {
                bestD = d;
                best = k;
            }
        }
        labels[n] = best;
    }
    return labels;
}
/** Mean of each cluster's points. A cluster that lost all its points keeps its center. */
export function updateCenters(X, labels, previous) {
    const k = previous.length;
    const d = X[0].length;
    const sums = Array.from({ length: k }, () => new Array(d).fill(0));
    const counts = new Array(k).fill(0);
    for (let n = 0; n < X.length; n++) {
        const c = labels[n];
        counts[c]++;
        for (let j = 0; j < d; j++)
            sums[c][j] += X[n][j];
    }
    return sums.map((s, c) => (counts[c] ? s.map((v) => v / counts[c]) : [...previous[c]]));
}
export function inertia(X, labels, centers) {
    let s = 0;
    for (let n = 0; n < X.length; n++)
        s += sqDist(X[n], centers[labels[n]]);
    return s;
}
export function kmeansStart(X, init) {
    return { centers: init.map((c) => [...c]), labels: assign(X, init), iteration: 0, converged: false };
}
/** One Lloyd iteration: move centers to their cluster means, then reassign points. */
export function kmeansStep(X, s) {
    if (s.converged)
        return s;
    const centers = updateCenters(X, s.labels, s.centers);
    const labels = assign(X, centers);
    const converged = labels.every((l, i) => l === s.labels[i]);
    return { centers, labels, iteration: s.iteration + 1, converged };
}
export function kmeans(X, init, maxIter = 300) {
    let s = kmeansStart(X, init);
    while (!s.converged && s.iteration < maxIter)
        s = kmeansStep(X, s);
    return s;
}
/** k-means++ seeding: each new center is drawn with probability proportional to its
 * squared distance from the nearest center chosen so far. With `trials` > 1 it is the
 * greedy variant scikit-learn uses: draw that many candidates and keep the one that
 * leaves the smallest total squared distance. */
export function kmeansPlusPlus(X, k, rand, trials = 1) {
    const centers = [X[Math.floor(rand() * X.length)]];
    let d2 = X.map((x) => sqDist(x, centers[0]));
    while (centers.length < k) {
        const total = d2.reduce((a, b) => a + b, 0);
        let best = { pick: -1, potential: Infinity, d2 };
        for (let t = 0; t < trials; t++) {
            let r = rand() * total;
            let pick = X.length - 1;
            for (let i = 0; i < X.length; i++) {
                r -= d2[i];
                if (r <= 0) {
                    pick = i;
                    break;
                }
            }
            const next = X.map((x, i) => Math.min(d2[i], sqDist(x, X[pick])));
            const potential = next.reduce((a, b) => a + b, 0);
            if (potential < best.potential)
                best = { pick, potential, d2: next };
        }
        centers.push(X[best.pick]);
        d2 = best.d2;
    }
    return centers.map((c) => [...c]);
}
export function randomPoints(X, k, rand) {
    const idx = Array.from(X.keys());
    for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx.slice(0, k).map((i) => [...X[i]]);
}
/* ---------------------------------------------------- Gaussian mixture (EM) */
const EPS = 2.220446049250313e-16;
const LOG_2PI = Math.log(2 * Math.PI);
/** Lower-triangular Cholesky factor L with A = L L^T. */
export function cholesky(A) {
    const n = A.length;
    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let s = A[i][j];
            for (let k = 0; k < j; k++)
                s -= L[i][k] * L[j][k];
            if (i === j) {
                if (s <= 0)
                    throw new Error("matrix is not positive definite");
                L[i][j] = Math.sqrt(s);
            }
            else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return L;
}
/** Upper-triangular U with precision = U U^T, as scikit-learn's precisions_cholesky_. */
export function precisionCholesky(cov) {
    const L = cholesky(cov);
    const n = L.length;
    // invert the lower-triangular factor by forward substitution, then transpose
    const inv = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let col = 0; col < n; col++) {
        for (let i = 0; i < n; i++) {
            let s = i === col ? 1 : 0;
            for (let k = 0; k < i; k++)
                s -= L[i][k] * inv[k][col];
            inv[i][col] = s / L[i][i];
        }
    }
    return inv[0].map((_, i) => inv.map((row) => row[i]));
}
function logSumExp(v) {
    const m = Math.max(...v);
    if (!Number.isFinite(m))
        return m;
    let s = 0;
    for (const x of v)
        s += Math.exp(x - m);
    return m + Math.log(s);
}
/** E-step: responsibilities and mean log-likelihood under the current parameters. */
export function eStep(X, weights, means, covariances) {
    const k = means.length;
    const d = X[0].length;
    const precChol = covariances.map(precisionCholesky);
    const logDet = precChol.map((U) => U.reduce((s, row, i) => s + Math.log(row[i]), 0));
    const resp = [];
    let total = 0;
    for (const x of X) {
        const weighted = new Array(k);
        for (let c = 0; c < k; c++) {
            const U = precChol[c];
            let maha = 0;
            for (let j = 0; j < d; j++) {
                let y = 0;
                for (let i = 0; i < d; i++)
                    y += (x[i] - means[c][i]) * U[i][j];
                maha += y * y;
            }
            weighted[c] = -0.5 * (d * LOG_2PI + maha) + logDet[c] + Math.log(weights[c]);
        }
        const norm = logSumExp(weighted);
        total += norm;
        resp.push(weighted.map((w) => Math.exp(w - norm)));
    }
    return { resp, logLikelihood: total / X.length };
}
/** M-step: weights, means and covariances from responsibilities (+ reg on the diagonal). */
export function mStep(X, resp, regCovar = 1e-6) {
    const n = X.length;
    const d = X[0].length;
    const k = resp[0].length;
    const nk = new Array(k).fill(10 * EPS);
    for (const r of resp)
        for (let c = 0; c < k; c++)
            nk[c] += r[c];
    const means = Array.from({ length: k }, (_, c) => {
        const m = new Array(d).fill(0);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < d; j++)
                m[j] += resp[i][c] * X[i][j];
        return m.map((v) => v / nk[c]);
    });
    const covariances = means.map((mu, c) => {
        const cov = Array.from({ length: d }, () => new Array(d).fill(0));
        for (let i = 0; i < n; i++) {
            const r = resp[i][c];
            for (let a = 0; a < d; a++) {
                const da = X[i][a] - mu[a];
                for (let b = 0; b < d; b++)
                    cov[a][b] += r * da * (X[i][b] - mu[b]);
            }
        }
        for (let a = 0; a < d; a++) {
            for (let b = 0; b < d; b++)
                cov[a][b] /= nk[c];
            cov[a][a] += regCovar;
        }
        return cov;
    });
    const sum = nk.reduce((a, b) => a + b, 0);
    return { weights: nk.map((v) => v / sum), means, covariances };
}
export function gmmStart(X, means, covariances, weights) {
    const w = weights ?? means.map(() => 1 / means.length);
    const { resp, logLikelihood } = eStep(X, w, means, covariances);
    return { weights: w, means, covariances, resp, logLikelihood, iteration: 0 };
}
/** One EM iteration (M-step from the current responsibilities, then a fresh E-step). */
export function gmmStep(X, s, regCovar = 1e-6) {
    const { weights, means, covariances } = mStep(X, s.resp, regCovar);
    const { resp, logLikelihood } = eStep(X, weights, means, covariances);
    return { weights, means, covariances, resp, logLikelihood, iteration: s.iteration + 1 };
}
export function gmmLabels(s) {
    return Int32Array.from(s.resp, (r) => r.indexOf(Math.max(...r)));
}
export function covarianceOf(X) {
    const n = X.length;
    const d = X[0].length;
    const mean = new Array(d).fill(0);
    for (const x of X)
        for (let j = 0; j < d; j++)
            mean[j] += x[j] / n;
    const cov = Array.from({ length: d }, () => new Array(d).fill(0));
    for (const x of X)
        for (let a = 0; a < d; a++)
            for (let b = 0; b < d; b++)
                cov[a][b] += ((x[a] - mean[a]) * (x[b] - mean[b])) / (n - 1);
    return cov;
}
/* ------------------------------------------------------------------ scores */
export function silhouette(X, labels) {
    const n = X.length;
    const k = Math.max(...Array.from(labels)) + 1;
    const sizes = new Array(k).fill(0);
    for (let i = 0; i < n; i++)
        sizes[labels[i]]++;
    if (sizes.filter((s) => s > 0).length < 2)
        return NaN;
    let total = 0;
    const sums = new Array(k);
    for (let i = 0; i < n; i++) {
        sums.fill(0);
        for (let j = 0; j < n; j++)
            if (j !== i)
                sums[labels[j]] += Math.sqrt(sqDist(X[i], X[j]));
        const own = labels[i];
        if (sizes[own] <= 1)
            continue; // silhouette of a singleton is 0
        const a = sums[own] / (sizes[own] - 1);
        let b = Infinity;
        for (let c = 0; c < k; c++)
            if (c !== own && sizes[c] > 0)
                b = Math.min(b, sums[c] / sizes[c]);
        total += (b - a) / Math.max(a, b);
    }
    return total / n;
}
const choose2 = (x) => (x * (x - 1)) / 2;
export function adjustedRandIndex(a, b) {
    const n = a.length;
    const table = new Map();
    const rowSums = new Map();
    const colSums = new Map();
    for (let i = 0; i < n; i++) {
        const key = `${a[i]}|${b[i]}`;
        table.set(key, (table.get(key) ?? 0) + 1);
        rowSums.set(a[i], (rowSums.get(a[i]) ?? 0) + 1);
        colSums.set(b[i], (colSums.get(b[i]) ?? 0) + 1);
    }
    let index = 0;
    for (const v of table.values())
        index += choose2(v);
    let rows = 0;
    for (const v of rowSums.values())
        rows += choose2(v);
    let cols = 0;
    for (const v of colSums.values())
        cols += choose2(v);
    const expected = (rows * cols) / choose2(n);
    const max = (rows + cols) / 2;
    if (max === expected)
        return 1;
    return (index - expected) / (max - expected);
}
/** Best one-to-one matching of clusters to classes, maximizing agreement
 * (the same optimum scipy's linear_sum_assignment finds). */
export function matchClusters(truth, labels, nClasses, k) {
    const counts = Array.from({ length: nClasses }, () => new Array(k).fill(0));
    for (let i = 0; i < truth.length; i++)
        counts[truth[i]][labels[i]]++;
    let best = { score: -1, mapping: [] };
    const mapping = [];
    const used = new Set();
    // each class takes a distinct cluster, or none (-1) when that scores better
    const search = (cls, score) => {
        if (cls === nClasses) {
            if (score > best.score)
                best = { score, mapping: [...mapping] };
            return;
        }
        for (let c = -1; c < k; c++) {
            if (c >= 0 && used.has(c))
                continue;
            if (c >= 0)
                used.add(c);
            mapping.push(c);
            search(cls + 1, score + (c >= 0 ? counts[cls][c] : 0));
            mapping.pop();
            if (c >= 0)
                used.delete(c);
        }
    };
    search(0, 0);
    return { counts, mapping: best.mapping, accuracy: best.score / truth.length };
}
