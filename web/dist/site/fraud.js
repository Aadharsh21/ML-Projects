/**
 * The fraud page's model of the world: the 71,202 test transactions (9.3 hours, 94
 * frauds) with both models' scores, what any flagging policy catches and costs, a
 * clock to replay the stream, and explanations for individual transactions.
 */
import { waterfall } from "../ml/explain.js";
import { Forest, sigmoid } from "../ml/lgbm.js";
import { Stream } from "../ml/fraud.js";
import { loadFiles } from "./load.js";
/** Readable names for the model inputs; V1-V28 are anonymised PCA components of the card data. */
function groupOf(feature) {
    if (feature === "LogAmount")
        return "Amount";
    if (feature === "HourSin" || feature === "HourCos")
        return "Time of day";
    return feature;
}
const lowerBound = (xs, x) => {
    let lo = 0;
    let hi = xs.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
};
export async function loadFraud(options = {}) {
    const { stream: streamJson, meta, inspect } = await loadFiles({ stream: "fraud/stream.json", meta: "fraud/meta.json", inspect: "fraud/inspect.json" }, options, 1, "Downloading 71,202 scored transactions");
    const stream = new Stream(streamJson);
    const t0 = stream.time[0];
    const hourOf = Float64Array.from(stream.time, (t) => (t - t0) / 3600);
    const hours = meta.test_hours;
    const flagCache = new Map();
    const flags = (policy) => {
        const key = `${policy.model}|${policy.rule}|${policy.rule === "threshold" ? policy.threshold ?? 0.5 : policy.reviewCost}`;
        let f = flagCache.get(key);
        if (!f) {
            f =
                policy.rule === "threshold"
                    ? stream.flagged(policy.model, { kind: "threshold", probability: policy.threshold ?? 0.5 })
                    : stream.flagged(policy.model, { kind: "expected_loss", reviewCost: policy.reviewCost });
            if (flagCache.size > 64)
                flagCache.clear();
            flagCache.set(key, f);
        }
        return f;
    };
    // model inputs are saved for every fraud, everything v2 scores at 1% or more, and a sample of the rest
    const rows = new Map(inspect.index.map((row, k) => [row, inspect.features[k]]));
    let forest = null;
    const loadForest = () => (forest ??= loadFiles({ model: "fraud/model.json" }, { ...options, onProgress: undefined }).then(({ model }) => new Forest(model)));
    return {
        meta,
        stream,
        count: stream.n,
        hours,
        hourOf,
        probability: stream.probability,
        flags,
        evaluate(policy, upTo = stream.n) {
            const n = Math.max(0, Math.min(stream.n, Math.round(upTo)));
            const o = stream.outcome(flags(policy), policy.reviewCost, n);
            const covered = n >= stream.n ? hours : n > 0 ? hourOf[n - 1] : 0;
            return {
                ...o,
                precision: o.alerts ? o.tp / o.alerts : null,
                recall: o.tp + o.fn ? o.tp / (o.tp + o.fn) : null,
                hours: covered,
                alertsPerHour: covered > 0 ? o.alerts / covered : 0,
            };
        },
        indexAt: (hour) => lowerBound(hourOf, hour),
        events(policy, upTo = stream.n, limit = 50) {
            const f = flags(policy);
            const p = stream.probability[policy.model];
            const out = [];
            for (let i = Math.min(stream.n, upTo) - 1; i >= 0 && out.length < limit; i--) {
                const fraud = stream.isFraud[i] === 1;
                const flagged = f[i] === 1;
                if (!fraud && !flagged)
                    continue;
                out.push({
                    index: i,
                    hour: hourOf[i],
                    amount: stream.amount[i],
                    probability: p[i],
                    fraud,
                    flagged,
                    kind: fraud ? (flagged ? "caught" : "missed") : "false_alarm",
                });
            }
            return out;
        },
        tradeoff(model, reviewCost, points = 181) {
            // cut-offs evenly spaced in log-odds between 0.01% and 99.99%, strictest first
            const cuts = Array.from({ length: points }, (_, i) => 9.2 - (18.4 * i) / (points - 1));
            const z = stream.logit[model];
            const order = Array.from(z.keys()).sort((a, b) => z[b] - z[a]);
            const out = [];
            let k = 0;
            let tp = 0;
            let fp = 0;
            let blocked = 0;
            for (const cut of cuts) {
                while (k < order.length && z[order[k]] >= cut) {
                    const i = order[k++];
                    if (stream.isFraud[i]) {
                        tp++;
                        blocked += stream.amount[i];
                    }
                    else
                        fp++;
                }
                const alerts = tp + fp;
                out.push({
                    threshold: sigmoid(cut),
                    alerts,
                    tp,
                    fp,
                    precision: alerts ? tp / alerts : null,
                    recall: tp / stream.fraudCount,
                    blocked,
                    net: blocked - alerts * reviewCost,
                });
            }
            const best = out.reduce((a, b) => (b.net > a.net ? b : a));
            return { points: out, best };
        },
        canExplain: (index) => rows.has(index),
        async explain(index) {
            const x = rows.get(index);
            if (!x)
                return null;
            const model = await loadForest();
            const phi = model.shap(x);
            const totals = new Map();
            model.features.forEach((f, j) => totals.set(groupOf(f), (totals.get(groupOf(f)) ?? 0) + phi[j]));
            const contributions = [...totals].map(([label, value]) => ({ label, value })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
            return {
                index,
                probability: model.predictProba(x),
                waterfall: waterfall(phi[phi.length - 1], contributions, 6),
                contributions,
            };
        },
    };
}
