/**
 * Gradient boosted trees exported from LightGBM (see ml/lgbm_export.py), evaluated
 * in the browser. Split decisions follow LightGBM's NumericalDecision exactly, and
 * `shap` is a port of LightGBM's TreeSHAP, so the browser reproduces the library's
 * predictions and explanations (checked against Python fixtures in tests/).
 */
// missing types: 0 = none (NaN is treated as 0), 1 = zero, 2 = NaN
const MISSING_ZERO = 1;
const MISSING_NAN = 2;
const ZERO_THRESHOLD = 1e-35;
function compile(t) {
    const n = t.split_feature.length;
    const tree = {
        feature: Int32Array.from(t.split_feature),
        threshold: Float64Array.from(t.threshold),
        missingType: Uint8Array.from(t.kind, (k) => k >> 1),
        defaultLeft: Uint8Array.from(t.kind, (k) => k & 1),
        left: Int32Array.from(t.left),
        right: Int32Array.from(t.right),
        count: Float64Array.from(t.count),
        leafValue: Float64Array.from(t.leaf_value),
        leafCount: Float64Array.from(t.leaf_count),
        depth: 0,
        expectedValue: 0,
    };
    if (n === 0) {
        tree.expectedValue = tree.leafValue[0];
        return tree;
    }
    const total = tree.count[0];
    let ev = 0;
    for (let i = 0; i < tree.leafValue.length; i++)
        ev += (tree.leafCount[i] / total) * tree.leafValue[i];
    tree.expectedValue = ev;
    const depthOf = (node) => node < 0 ? 0 : 1 + Math.max(depthOf(tree.left[node]), depthOf(tree.right[node]));
    tree.depth = depthOf(0);
    return tree;
}
/** Child a value is routed to, exactly as LightGBM's Tree::NumericalDecision. */
function decide(tree, node, raw) {
    const missingType = tree.missingType[node];
    let value = raw;
    if (Number.isNaN(value) && missingType !== MISSING_NAN)
        value = 0;
    if ((missingType === MISSING_ZERO && value >= -ZERO_THRESHOLD && value <= ZERO_THRESHOLD) ||
        (missingType === MISSING_NAN && Number.isNaN(value))) {
        return tree.defaultLeft[node] ? tree.left[node] : tree.right[node];
    }
    return value <= tree.threshold[node] ? tree.left[node] : tree.right[node];
}
function leafFor(tree, x) {
    if (tree.feature.length === 0)
        return 0;
    let node = 0;
    while (node >= 0)
        node = decide(tree, node, x[tree.feature[node]]);
    return ~node;
}
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
/** Path buffer for TreeSHAP: parallel arrays standing in for LightGBM's PathElement. */
class PathBuffer {
    f;
    z;
    o;
    w;
    constructor(size) {
        this.f = new Int32Array(size);
        this.z = new Float64Array(size);
        this.o = new Float64Array(size);
        this.w = new Float64Array(size);
    }
}
function extendPath(P, off, depth, zero, one, feature) {
    P.f[off + depth] = feature;
    P.z[off + depth] = zero;
    P.o[off + depth] = one;
    P.w[off + depth] = depth === 0 ? 1 : 0;
    for (let i = depth - 1; i >= 0; i--) {
        P.w[off + i + 1] += (one * P.w[off + i] * (i + 1)) / (depth + 1);
        P.w[off + i] = (zero * P.w[off + i] * (depth - i)) / (depth + 1);
    }
}
function unwindPath(P, off, depth, pathIndex) {
    const one = P.o[off + pathIndex];
    const zero = P.z[off + pathIndex];
    let next = P.w[off + depth];
    for (let i = depth - 1; i >= 0; i--) {
        if (one !== 0) {
            const tmp = P.w[off + i];
            P.w[off + i] = (next * (depth + 1)) / ((i + 1) * one);
            next = tmp - (P.w[off + i] * zero * (depth - i)) / (depth + 1);
        }
        else {
            P.w[off + i] = (P.w[off + i] * (depth + 1)) / (zero * (depth - i));
        }
    }
    for (let i = pathIndex; i < depth; i++) {
        P.f[off + i] = P.f[off + i + 1];
        P.z[off + i] = P.z[off + i + 1];
        P.o[off + i] = P.o[off + i + 1];
    }
}
function unwoundPathSum(P, off, depth, pathIndex) {
    const one = P.o[off + pathIndex];
    const zero = P.z[off + pathIndex];
    let next = P.w[off + depth];
    let total = 0;
    for (let i = depth - 1; i >= 0; i--) {
        if (one !== 0) {
            const tmp = (next * (depth + 1)) / ((i + 1) * one);
            total += tmp;
            next = P.w[off + i] - tmp * zero * ((depth - i) / (depth + 1));
        }
        else {
            total += P.w[off + i] / zero / ((depth - i) / (depth + 1));
        }
    }
    return total;
}
function dataCount(tree, node) {
    return node >= 0 ? tree.count[node] : tree.leafCount[~node];
}
function treeShap(tree, x, phi, P, node, depthIn, parentOff, parentZero, parentOne, parentFeature) {
    let depth = depthIn;
    const off = parentOff + depth;
    if (depth > 0) {
        P.f.copyWithin(off, parentOff, parentOff + depth);
        P.z.copyWithin(off, parentOff, parentOff + depth);
        P.o.copyWithin(off, parentOff, parentOff + depth);
        P.w.copyWithin(off, parentOff, parentOff + depth);
    }
    extendPath(P, off, depth, parentZero, parentOne, parentFeature);
    if (node < 0) {
        const value = tree.leafValue[~node];
        for (let i = 1; i <= depth; i++) {
            const w = unwoundPathSum(P, off, depth, i);
            phi[P.f[off + i]] += w * (P.o[off + i] - P.z[off + i]) * value;
        }
        return;
    }
    const feature = tree.feature[node];
    const hot = decide(tree, node, x[feature]);
    const cold = hot === tree.left[node] ? tree.right[node] : tree.left[node];
    const w = dataCount(tree, node);
    const hotZero = dataCount(tree, hot) / w;
    const coldZero = dataCount(tree, cold) / w;
    let incomingZero = 1;
    let incomingOne = 1;
    let pathIndex = 0;
    for (; pathIndex <= depth; pathIndex++) {
        if (P.f[off + pathIndex] === feature)
            break;
    }
    if (pathIndex !== depth + 1) {
        incomingZero = P.z[off + pathIndex];
        incomingOne = P.o[off + pathIndex];
        unwindPath(P, off, depth, pathIndex);
        depth -= 1;
    }
    treeShap(tree, x, phi, P, hot, depth + 1, off, hotZero * incomingZero, incomingOne, feature);
    treeShap(tree, x, phi, P, cold, depth + 1, off, coldZero * incomingZero, 0, feature);
}
export class Forest {
    features;
    baseValue;
    trees;
    path;
    constructor(json) {
        if (json.format !== "lgbm-flat-1")
            throw new Error(`Unknown model format ${json.format}`);
        this.features = json.features;
        this.trees = json.trees.map(compile);
        this.baseValue = this.trees.reduce((s, t) => s + t.expectedValue, 0);
        const maxDepth = Math.max(0, ...this.trees.map((t) => t.depth));
        this.path = new PathBuffer(((maxDepth + 2) * (maxDepth + 3)) / 2 + 1);
    }
    get size() {
        return this.trees.length;
    }
    /** Sum of leaf outputs: log-odds for a binary model. */
    predictRaw(x) {
        let sum = 0;
        for (const tree of this.trees)
            sum += tree.leafValue[leafFor(tree, x)];
        return sum;
    }
    predictProba(x) {
        return sigmoid(this.predictRaw(x));
    }
    /**
     * SHAP values in log-odds, one per feature, plus the expected value as the last
     * entry (LightGBM's pred_contrib layout). They sum to predictRaw(x).
     */
    shap(x) {
        const nf = this.features.length;
        const phi = new Float64Array(nf + 1);
        for (const tree of this.trees) {
            phi[nf] += tree.expectedValue;
            if (tree.feature.length > 0)
                treeShap(tree, x, phi, this.path, 0, 0, 0, 1, 1, -1);
        }
        return phi;
    }
}
export { sigmoid };
