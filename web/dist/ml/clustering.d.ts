/**
 * k-means (Lloyd's algorithm) and Gaussian mixture EM, written to step one
 * iteration at a time so the playground can animate them. The EM step follows
 * scikit-learn's GaussianMixture (full covariance) so results can be checked
 * against it.
 */
export type Matrix = number[][];
export declare function mulberry32(seed: number): () => number;
export declare function sqDist(a: ArrayLike<number>, b: ArrayLike<number>): number;
export declare function assign(X: Matrix, centers: Matrix): Int32Array;
/** Mean of each cluster's points. A cluster that lost all its points keeps its center. */
export declare function updateCenters(X: Matrix, labels: Int32Array, previous: Matrix): Matrix;
export declare function inertia(X: Matrix, labels: Int32Array, centers: Matrix): number;
export interface KMeansState {
    centers: Matrix;
    labels: Int32Array;
    iteration: number;
    converged: boolean;
}
export declare function kmeansStart(X: Matrix, init: Matrix): KMeansState;
/** One Lloyd iteration: move centers to their cluster means, then reassign points. */
export declare function kmeansStep(X: Matrix, s: KMeansState): KMeansState;
export declare function kmeans(X: Matrix, init: Matrix, maxIter?: number): KMeansState;
/** k-means++ seeding: each new center is drawn with probability proportional to its
 * squared distance from the nearest center chosen so far. With `trials` > 1 it is the
 * greedy variant scikit-learn uses: draw that many candidates and keep the one that
 * leaves the smallest total squared distance. */
export declare function kmeansPlusPlus(X: Matrix, k: number, rand: () => number, trials?: number): Matrix;
export declare function randomPoints(X: Matrix, k: number, rand: () => number): Matrix;
/** Lower-triangular Cholesky factor L with A = L L^T. */
export declare function cholesky(A: Matrix): Matrix;
/** Upper-triangular U with precision = U U^T, as scikit-learn's precisions_cholesky_. */
export declare function precisionCholesky(cov: Matrix): Matrix;
export interface GMMState {
    weights: number[];
    means: Matrix;
    covariances: Matrix[];
    /** responsibilities from the latest E-step, one row per point */
    resp: Matrix;
    /** mean log-likelihood per point (scikit-learn's lower bound) */
    logLikelihood: number;
    iteration: number;
}
/** E-step: responsibilities and mean log-likelihood under the current parameters. */
export declare function eStep(X: Matrix, weights: number[], means: Matrix, covariances: Matrix[]): {
    resp: Matrix;
    logLikelihood: number;
};
/** M-step: weights, means and covariances from responsibilities (+ reg on the diagonal). */
export declare function mStep(X: Matrix, resp: Matrix, regCovar?: number): {
    weights: number[];
    means: number[][];
    covariances: number[][][];
};
export declare function gmmStart(X: Matrix, means: Matrix, covariances: Matrix[], weights?: number[]): GMMState;
/** One EM iteration (M-step from the current responsibilities, then a fresh E-step). */
export declare function gmmStep(X: Matrix, s: GMMState, regCovar?: number): GMMState;
export declare function gmmLabels(s: GMMState): Int32Array;
export declare function covarianceOf(X: Matrix): Matrix;
export declare function silhouette(X: Matrix, labels: ArrayLike<number>): number;
export declare function adjustedRandIndex(a: ArrayLike<number>, b: ArrayLike<number>): number;
/** Best one-to-one matching of clusters to classes, maximizing agreement
 * (the same optimum scipy's linear_sum_assignment finds). */
export declare function matchClusters(truth: ArrayLike<number>, labels: ArrayLike<number>, nClasses: number, k: number): {
    counts: number[][];
    mapping: number[];
    accuracy: number;
};
