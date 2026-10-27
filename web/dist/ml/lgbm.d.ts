/**
 * Gradient boosted trees exported from LightGBM (see ml/lgbm_export.py), evaluated
 * in the browser. Split decisions follow LightGBM's NumericalDecision exactly, and
 * `shap` is a port of LightGBM's TreeSHAP, so the browser reproduces the library's
 * predictions and explanations (checked against Python fixtures in tests/).
 */
export interface FlatTreeJSON {
    split_feature: number[];
    threshold: number[];
    kind: number[];
    left: number[];
    right: number[];
    count: number[];
    leaf_value: number[];
    leaf_count: number[];
}
export interface ForestJSON {
    format: string;
    objective: string;
    features: string[];
    trees: FlatTreeJSON[];
    base_value?: number;
}
declare const sigmoid: (z: number) => number;
export declare class Forest {
    readonly features: string[];
    readonly baseValue: number;
    private readonly trees;
    private readonly path;
    constructor(json: ForestJSON);
    get size(): number;
    /** Sum of leaf outputs: log-odds for a binary model. */
    predictRaw(x: ArrayLike<number>): number;
    predictProba(x: ArrayLike<number>): number;
    /**
     * SHAP values in log-odds, one per feature, plus the expected value as the last
     * entry (LightGBM's pred_contrib layout). They sum to predictRaw(x).
     */
    shap(x: ArrayLike<number>): Float64Array;
}
export { sigmoid };
