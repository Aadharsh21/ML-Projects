/**
 * The fraud page's model of the world: the 71,202 test transactions (9.3 hours, 94
 * frauds) with both models' scores, what any flagging policy catches and costs, a
 * clock to replay the stream, and explanations for individual transactions.
 */
import { waterfall, type Step } from "../ml/explain.js";
import { Forest, sigmoid, type ForestJSON } from "../ml/lgbm.js";
import { Stream, type ModelKey, type Outcome, type StreamJSON } from "../ml/fraud.js";
import { loadFiles, type LoadOptions } from "./load.js";

export interface FraudModelMetrics {
  accuracy: number;
  roc_auc: number;
  avg_precision: number;
  brier: number;
  log_loss: number;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface FraudMeta {
  features: string[];
  splits: Record<"train" | "val" | "test", { rows: number; frauds: number; start_hour: number; end_hour: number }>;
  test_hours: number;
  test_fraud_amount: number;
  review_cost: number;
  best_iteration: number;
  models: Record<
    string,
    {
      label: string;
      test: FraudModelMetrics & {
        blocked: number;
        missed: number;
        review_cost: number;
        net_savings: number;
        alerts: number;
        alerts_per_hour: number;
        avg_precision_ci95: [number, number];
        expected_loss_rule: { blocked: number; missed: number; review_cost: number; net_savings: number; alerts: number; tp: number; fp: number };
      };
    }
  >;
  /** v1 graded the way v1 graded itself: on a balanced 50/50 sample */
  v1_balanced_view: FraudModelMetrics;
  /** precision-recall curves on the test stream as [recall, precision] pairs */
  pr_curves: Record<ModelKey, [number, number][]>;
  importance: { feature: string; mean_abs_shap: number }[];
  source: string;
}

/** How alerts are raised: above a probability, or when expected loss beats the review cost. */
export interface FraudPolicy {
  model: ModelKey;
  rule: "threshold" | "expected_loss";
  /** probability cut-off for the threshold rule (default 0.5) */
  threshold?: number;
  /** dollars to review one alert; also the bar for the expected-loss rule */
  reviewCost: number;
}

export interface FraudOutcome extends Outcome {
  /** share of alerts that were fraud; null before the first alert */
  precision: number | null;
  /** share of frauds so far that were caught; null before the first fraud */
  recall: number | null;
  /** hours of the stream covered */
  hours: number;
  alertsPerHour: number;
}

export interface FraudEvent {
  index: number;
  /** hours since the start of the test window */
  hour: number;
  amount: number;
  probability: number;
  fraud: boolean;
  flagged: boolean;
  kind: "caught" | "missed" | "false_alarm";
}

export interface TradeoffPoint {
  /** probability cut-off */
  threshold: number;
  alerts: number;
  tp: number;
  fp: number;
  precision: number | null;
  recall: number;
  blocked: number;
  net: number;
}

export interface FraudExplanation {
  index: number;
  probability: number;
  /** from the average transaction's score to this one, biggest reasons first */
  waterfall: { start: number; end: number; steps: Step[] };
  /** every feature group's contribution in log-odds, largest first */
  contributions: { label: string; value: number }[];
}

export interface FraudLab {
  meta: FraudMeta;
  stream: Stream;
  /** transactions in the test window */
  count: number;
  /** length of the test window in hours */
  hours: number;
  /** hours since the start of the window, per transaction */
  hourOf: Float64Array;
  /** P(fraud) per transaction for each model */
  probability: Record<ModelKey, Float64Array>;
  flags(policy: FraudPolicy): Uint8Array;
  /** totals for the first `upTo` transactions (default: all) */
  evaluate(policy: FraudPolicy, upTo?: number): FraudOutcome;
  /** number of transactions before a given hour of the window, for a replay clock */
  indexAt(hour: number): number;
  /** frauds and alerts among the first `upTo` transactions, most recent first */
  events(policy: FraudPolicy, upTo?: number, limit?: number): FraudEvent[];
  /** outcome at a sweep of probability cut-offs, from strict to lenient */
  tradeoff(model: ModelKey, reviewCost: number, points?: number): { points: TradeoffPoint[]; best: TradeoffPoint };
  /** whether explain() has the inputs for this transaction (every fraud, everything v2 scores at 1% or more, and a sample of the rest) */
  canExplain(index: number): boolean;
  /** why the v2 model scored a transaction the way it did (downloads the model on first use) */
  explain(index: number): Promise<FraudExplanation | null>;
}

/** Readable names for the model inputs; V1-V28 are anonymised PCA components of the card data. */
function groupOf(feature: string): string {
  if (feature === "LogAmount") return "Amount";
  if (feature === "HourSin" || feature === "HourCos") return "Time of day";
  return feature;
}

const lowerBound = (xs: Float64Array, x: number) => {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
};

export async function loadFraud(options: LoadOptions = {}): Promise<FraudLab> {
  const { stream: streamJson, meta, inspect } = await loadFiles<{
    stream: StreamJSON;
    meta: FraudMeta;
    inspect: { index: number[]; features: number[][] };
  }>({ stream: "fraud/stream.json", meta: "fraud/meta.json", inspect: "fraud/inspect.json" }, options, 1, "Downloading 71,202 scored transactions");
  const stream = new Stream(streamJson);
  const t0 = stream.time[0];
  const hourOf = Float64Array.from(stream.time, (t) => (t - t0) / 3600);
  const hours = meta.test_hours;

  const flagCache = new Map<string, Uint8Array>();
  const flags = (policy: FraudPolicy) => {
    const key = `${policy.model}|${policy.rule}|${policy.rule === "threshold" ? policy.threshold ?? 0.5 : policy.reviewCost}`;
    let f = flagCache.get(key);
    if (!f) {
      f =
        policy.rule === "threshold"
          ? stream.flagged(policy.model, { kind: "threshold", probability: policy.threshold ?? 0.5 })
          : stream.flagged(policy.model, { kind: "expected_loss", reviewCost: policy.reviewCost });
      if (flagCache.size > 64) flagCache.clear();
      flagCache.set(key, f);
    }
    return f;
  };

  // model inputs are saved for every fraud, everything v2 scores at 1% or more, and a sample of the rest
  const rows = new Map(inspect.index.map((row, k) => [row, inspect.features[k]]));
  let forest: Promise<Forest> | null = null;
  const loadForest = () =>
    (forest ??= loadFiles<{ model: ForestJSON }>({ model: "fraud/model.json" }, { ...options, onProgress: undefined }).then(
      ({ model }) => new Forest(model),
    ));

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
      const out: FraudEvent[] = [];
      for (let i = Math.min(stream.n, upTo) - 1; i >= 0 && out.length < limit; i--) {
        const fraud = stream.isFraud[i] === 1;
        const flagged = f[i] === 1;
        if (!fraud && !flagged) continue;
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
      const out: TradeoffPoint[] = [];
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
          } else fp++;
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
      if (!x) return null;
      const model = await loadForest();
      const phi = model.shap(x);
      const totals = new Map<string, number>();
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
