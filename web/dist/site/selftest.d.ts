import { type Fetcher, type LoadOptions } from "./load.js";
export interface ParityCheck {
    project: "Rain" | "Fraud" | "Insurance" | "Clustering";
    /** what is being compared, in plain words */
    label: string;
    /** how many values were compared */
    count: number;
    /** largest absolute difference from the Python result */
    maxError: number;
    tolerance: number;
    pass: boolean;
}
export interface SelfTestOptions extends LoadOptions {
    /** base URL of the reference files (defaults to web/tests/fixtures next to this module) */
    referenceUrl?: string;
    /** replaces fetching the reference files, for tests */
    referenceFetcher?: Fetcher;
}
/** Run every parity check. Downloads about 4 MB the first time (less if the pages already loaded their models). */
export declare function selfTest(options?: SelfTestOptions): Promise<ParityCheck[]>;
