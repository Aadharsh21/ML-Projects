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
export type Rule = {
    kind: "threshold";
    probability: number;
} | {
    kind: "expected_loss";
    reviewCost: number;
};
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
export declare class Stream {
    readonly n: number;
    /** seconds since the first transaction in the dataset */
    readonly time: Float64Array;
    readonly amount: Float64Array;
    readonly isFraud: Uint8Array;
    readonly logit: Record<ModelKey, Float64Array>;
    readonly probability: Record<ModelKey, Float64Array>;
    readonly fraudCount: number;
    readonly fraudAmount: number;
    constructor(json: StreamJSON);
    flagged(model: ModelKey, rule: Rule): Uint8Array;
    /** Totals over the first `upTo` transactions (all of them by default). */
    outcome(flags: Uint8Array, reviewCost: number, upTo?: number): Outcome;
    /**
     * Precision/recall and net savings at every distinct score, highest first: the
     * whole trade-off curve in one sorted pass.
     */
    curve(model: ModelKey, reviewCost: number): {
        logit: number;
        precision: number;
        recall: number;
        net: number;
        alerts: number;
    }[];
}
/** Raw transaction to model features (mirrors ml/fraud.py). */
export declare function fraudFeatures(pca: number[], amount: number, time: number): number[];
