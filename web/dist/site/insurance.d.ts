/**
 * The insurance page's calculator: expected yearly charges for any customer, the range
 * where 8 in 10 similar customers land, and exact Shapley values that split the
 * difference from the average customer across the six inputs.
 */
import { InsuranceModel, type InputKey, type Profile } from "../ml/insurance.js";
import { type LoadOptions } from "./load.js";
export type InsuranceGroup = "non_smoker" | "smoker_bmi_under_30" | "smoker_bmi_30_plus";
interface CvScore {
    mean: number;
    std: number;
}
export interface InsuranceMeta {
    rows: number;
    mean_charges: number;
    group_means: Record<InsuranceGroup, number>;
    /** 5x5 repeated cross-validation, in dollars for mae and rmse */
    models: Record<string, {
        label: string;
        cv: {
            r2: CvScore;
            mae: CvScore;
            rmse: CvScore;
        };
    }>;
    /** how often the middle-80% range contained the real charge on held-out customers */
    interval: {
        target: number;
        coverage: number;
        mean_width: number;
        repeats: number;
    };
    /** the v2 model's coefficients, in dollars */
    coefficients: Record<string, number>;
    source: string;
}
export interface InsuranceCustomer {
    profile: Profile;
    charges: number;
    group: InsuranceGroup;
}
export interface InsuranceContribution {
    key: InputKey;
    label: string;
    /** this profile's value, written out (e.g. "yes", "31.0") */
    text: string;
    /** dollars this input adds to (or takes off) the average customer's charges */
    value: number;
}
export interface InsuranceQuote {
    profile: Profile;
    group: InsuranceGroup;
    /** the model's expected yearly charges */
    charges: number;
    /** where 8 in 10 similar customers land, and the typical (median) charge */
    range: {
        lo: number;
        median: number;
        hi: number;
    };
    /** the average customer's predicted charges: where the contributions start */
    base: number;
    /** largest effect first; they add up to charges - base */
    contributions: InsuranceContribution[];
}
export interface InsuranceCalculator {
    meta: InsuranceMeta;
    model: InsuranceModel;
    /** northeast, northwest, southeast, southwest (Profile.region indexes this) */
    regions: string[];
    customers: InsuranceCustomer[];
    /** labels for the three bands of customers */
    groupLabels: Record<InsuranceGroup, string>;
    groupOf(profile: Profile): InsuranceGroup;
    quote(profile: Profile): InsuranceQuote;
    /** expected charges and range across ages for an otherwise fixed profile */
    byAge(profile: Profile, from?: number, to?: number): {
        age: number;
        charges: number;
        lo: number;
        median: number;
        hi: number;
    }[];
}
export declare const INSURANCE_LABELS: Record<InputKey, string>;
export declare function loadInsurance(options?: LoadOptions): Promise<InsuranceCalculator>;
export {};
