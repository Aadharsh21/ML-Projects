/** Insurance charges model (mirrors ml/insurance.py). */
export interface InsuranceModelJSON {
    features: string[];
    intercept: number;
    coef: number[];
    obese_bmi: number;
    regions: string[];
    inputs: string[];
    range: [number, number];
    residuals: {
        smoker: number[];
        non_smoker: number[];
    };
}
export interface Profile {
    age: number;
    male: 0 | 1;
    bmi: number;
    children: number;
    smoker: 0 | 1;
    /** index into regions: northeast, northwest, southeast, southwest */
    region: number;
}
/** The six inputs, in the order Shapley values are reported. */
export declare const INPUT_KEYS: readonly ["age", "male", "bmi", "children", "smoker", "region"];
export type InputKey = (typeof INPUT_KEYS)[number];
export declare function profileFromRow(row: number[]): Profile;
/** numpy.quantile's default (linear) method on a sorted array. */
export declare function quantileSorted(sorted: ArrayLike<number>, q: number): number;
export declare class InsuranceModel {
    readonly json: InsuranceModelJSON;
    /** coefficient lookup by feature name (0 when a feature is absent) */
    private readonly c;
    constructor(json: InsuranceModelJSON);
    /** Design row in the order of json.features. */
    design(p: Profile): Float64Array;
    /** Same value as design() . coef + intercept, without allocating (used in loops). */
    predict(p: Profile): number;
    residuals(p: Profile): number[];
    /** Middle-80% range and median of likely charges for this profile. */
    range(p: Profile): {
        lo: number;
        median: number;
        hi: number;
    };
    /**
     * Exact Shapley values of the six inputs against the average customer, as in
     * ml/insurance.py: v(S) averages the prediction over the background customers
     * with the inputs in S replaced by this profile's values.
     */
    shapley(p: Profile, background: Profile[]): {
        phi: number[];
        base: number;
    };
}
