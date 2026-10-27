/**
 * The fraud page's model of the world: the 71,202 test transactions (9.3 hours, 94
 * frauds) with both models' scores, what any flagging policy catches and costs, a
 * clock to replay the stream, and explanations for individual transactions.
 */
import { type Step } from "../ml/explain.js";
import { Stream, type ModelKey, type Outcome } from "../ml/fraud.js";
import { type LoadOptions } from "./load.js";
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
    splits: Record<"train" | "val" | "test", {
        rows: number;
        frauds: number;
        start_hour: number;
        end_hour: number;
    }>;
    test_hours: number;
    test_fraud_amount: number;
    review_cost: number;
    best_iteration: number;
    models: Record<string, {
        label: string;
        test: FraudModelMetrics & {
            blocked: number;
            missed: number;
            review_cost: number;
            net_savings: number;
            alerts: number;
            alerts_per_hour: number;
            avg_precision_ci95: [number, number];
            expected_loss_rule: {
                blocked: number;
                missed: number;
                review_cost: number;
                net_savings: number;
                alerts: number;
                tp: number;
                fp: number;
            };
        };
    }>;
    /** v1 graded the way v1 graded itself: on a balanced 50/50 sample */
    v1_balanced_view: FraudModelMetrics;
    /** precision-recall curves on the test stream as [recall, precision] pairs */
    pr_curves: Record<ModelKey, [number, number][]>;
    importance: {
        feature: string;
        mean_abs_shap: number;
    }[];
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
    waterfall: {
        start: number;
        end: number;
        steps: Step[];
    };
    /** every feature group's contribution in log-odds, largest first */
    contributions: {
        label: string;
        value: number;
    }[];
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
    tradeoff(model: ModelKey, reviewCost: number, points?: number): {
        points: TradeoffPoint[];
        best: TradeoffPoint;
    };
    /** whether explain() has the inputs for this transaction (every fraud, everything v2 scores at 1% or more, and a sample of the rest) */
    canExplain(index: number): boolean;
    /** why the v2 model scored a transaction the way it did (downloads the model on first use) */
    explain(index: number): Promise<FraudExplanation | null>;
}
export declare function loadFraud(options?: LoadOptions): Promise<FraudLab>;
