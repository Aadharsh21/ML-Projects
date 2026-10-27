import { type LoadOptions } from "./load.js";
export interface IrisData {
    features: string[];
    species: string[];
    /** four measurements in cm, then the species index */
    rows: number[][];
    pca: {
        mean: number[];
        components: number[][];
    };
}
export interface IrisSweepPoint {
    k: number;
    inertia: number;
    silhouette: number | null;
    ari: number;
}
export interface IrisMeta {
    features: string[];
    species: string[];
    /** k-means for k = 1..8 on raw and standardized measurements */
    sweep: {
        raw: IrisSweepPoint[];
        scaled: IrisSweepPoint[];
    };
    gmm_bic: {
        k: number;
        bic: number;
    }[];
    /** the published comparison; `matrix` is species (rows) by cluster (columns) */
    methods: Record<string, {
        label: string;
        ari: number;
        matrix: number[][];
        matched_accuracy: number;
    }>;
    pca_explained_variance: number[];
}
/** "pca" for the first two principal components, or two measurement indices such as [2, 3]. */
export type IrisView = "pca" | [number, number];
export interface IrisFrame {
    iteration: number;
    /** cluster of each flower */
    labels: number[];
    /** cluster centres in cm */
    centers: number[][];
    /** Gaussian mixture only: covariance (cm²) and weight of each component */
    covariances?: number[][][];
    weights?: number[];
    /** k-means only: within-cluster sum of squares, in the space it clustered in */
    inertia?: number;
    /** Gaussian mixture only: mean log-likelihood per flower */
    logLikelihood?: number;
}
export interface IrisScore {
    /** adjusted Rand index against the species: 1 is perfect, 0 is chance */
    ari: number;
    /** share of flowers in their species' cluster under the best one-to-one matching */
    accuracy: number;
    /** on the raw measurements; null when there is only one cluster */
    silhouette: number | null;
    /** flowers per species (rows) and cluster (columns) */
    counts: number[][];
    /** cluster matched to each species, or -1 */
    mapping: number[];
}
export interface IrisRunOptions {
    method: "kmeans" | "gmm";
    k: number;
    /** seeds the k-means++ starting points (default 0) */
    seed?: number;
    /** k-means only: cluster on standardized measurements instead of cm */
    scaled?: boolean;
    /** run from this many starting points (seed, seed + 1, ...) and keep the best, as
     * scikit-learn's n_init does: lowest inertia for k-means, highest likelihood for a mixture */
    restarts?: number;
    maxIter?: number;
}
export interface IrisRun extends Required<Omit<IrisRunOptions, "maxIter">> {
    /** the starting state, then one frame per iteration (of the best start) */
    frames: IrisFrame[];
    converged: boolean;
    score: IrisScore;
    /** seed of the kept start */
    bestSeed: number;
    /** every start tried: inertia (k-means) or mean log-likelihood (mixture) and ARI */
    tries: {
        seed: number;
        objective: number;
        ari: number;
    }[];
}
export interface IrisPlayground {
    data: IrisData;
    meta: IrisMeta;
    features: string[];
    species: string[];
    /** the 150 flowers' measurements in cm */
    X: number[][];
    /** species index of each flower */
    truth: number[];
    /** share of variance on the two principal components */
    explained: number[];
    /** a point (or centre) in the 2-D view */
    project(x: number[], view?: IrisView): [number, number];
    points(view?: IrisView): [number, number][];
    /** outline of a Gaussian component in the view, `radius` standard deviations out */
    ellipse(mean: number[], cov: number[][], view?: IrisView, radius?: number, segments?: number): [number, number][];
    run(options: IrisRunOptions): IrisRun;
    score(labels: ArrayLike<number>, k: number): IrisScore;
}
export declare function loadIris(options?: LoadOptions): Promise<IrisPlayground>;
