/** The scored test stream of card transactions, and what a flagging policy costs. */

export interface StreamJSON {
  start_time: number;
  dt: number[];
  amount_cents: number[];
  fraud: number[];
  v2: number[];
  v1: number[];
}

export type ModelKey = "v1" | "v2";

export type Rule =
  | { kind: "threshold"; probability: number }
  | { kind: "expected_loss"; reviewCost: number };

export interface Outcome {
  seen: number;
  alerts: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  blocked: number;
  missed: number;
  reviewCost: number;
  net: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export class Stream {
  readonly n: number;
  /** seconds since the first transaction in the dataset */
  readonly time: Float64Array;
  readonly amount: Float64Array;
  readonly isFraud: Uint8Array;
  readonly logit: Record<ModelKey, Float64Array>;
  readonly probability: Record<ModelKey, Float64Array>;
  readonly fraudCount: number;
  readonly fraudAmount: number;

  constructor(json: StreamJSON) {
    this.n = json.dt.length;
    this.time = new Float64Array(this.n);
    let t = json.start_time;
    for (let i = 0; i < this.n; i++) {
      t += json.dt[i];
      this.time[i] = t;
    }
    this.amount = Float64Array.from(json.amount_cents, (c) => c / 100);
    this.isFraud = new Uint8Array(this.n);
    for (const i of json.fraud) this.isFraud[i] = 1;
    this.logit = {
      v1: Float64Array.from(json.v1, (v) => v / 1000),
      v2: Float64Array.from(json.v2, (v) => v / 1000),
    };
    this.probability = {
      v1: this.logit.v1.map(sigmoid),
      v2: this.logit.v2.map(sigmoid),
    };
    this.fraudCount = json.fraud.length;
    this.fraudAmount = json.fraud.reduce((s, i) => s + this.amount[i], 0);
  }

  flagged(model: ModelKey, rule: Rule): Uint8Array {
    const out = new Uint8Array(this.n);
    if (rule.kind === "threshold") {
      // compare on the log-odds scale the scores are stored in
      const cut = Math.log(rule.probability / (1 - rule.probability));
      const z = this.logit[model];
      for (let i = 0; i < this.n; i++) out[i] = z[i] >= cut ? 1 : 0;
    } else {
      const p = this.probability[model];
      for (let i = 0; i < this.n; i++) out[i] = p[i] * this.amount[i] >= rule.reviewCost ? 1 : 0;
    }
    return out;
  }

  /** Totals over the first `upTo` transactions (all of them by default). */
  outcome(flags: Uint8Array, reviewCost: number, upTo = this.n): Outcome {
    let tp = 0, fp = 0, fn = 0, tn = 0, blocked = 0, missed = 0;
    for (let i = 0; i < upTo; i++) {
      const f = flags[i];
      if (this.isFraud[i]) {
        if (f) {
          tp++;
          blocked += this.amount[i];
        } else {
          fn++;
          missed += this.amount[i];
        }
      } else if (f) fp++;
      else tn++;
    }
    const alerts = tp + fp;
    return { seen: upTo, alerts, tp, fp, fn, tn, blocked, missed, reviewCost: alerts * reviewCost, net: blocked - alerts * reviewCost };
  }

  /**
   * Precision/recall and net savings at every distinct score, highest first: the
   * whole trade-off curve in one sorted pass.
   */
  curve(model: ModelKey, reviewCost: number) {
    const z = this.logit[model];
    const order = Array.from(z.keys()).sort((a, b) => z[b] - z[a]);
    const points: { logit: number; precision: number; recall: number; net: number; alerts: number }[] = [];
    let tp = 0, fp = 0, blocked = 0;
    for (let k = 0; k < order.length; k++) {
      const i = order[k];
      if (this.isFraud[i]) {
        tp++;
        blocked += this.amount[i];
      } else fp++;
      const next = order[k + 1];
      if (next !== undefined && z[next] === z[i]) continue; // ties share one threshold
      points.push({
        logit: z[i],
        precision: tp / (tp + fp),
        recall: tp / this.fraudCount,
        net: blocked - reviewCost * (tp + fp),
        alerts: tp + fp,
      });
    }
    return points;
  }
}

/** Raw transaction to model features (mirrors ml/fraud.py). */
export function fraudFeatures(pca: number[], amount: number, time: number): number[] {
  const hour = (time % 86400) / 3600;
  return [...pca, Math.log1p(amount), Math.sin((2 * Math.PI * hour) / 24), Math.cos((2 * Math.PI * hour) / 24)];
}
