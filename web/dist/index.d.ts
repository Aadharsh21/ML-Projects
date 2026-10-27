/**
 * Browser inference for the four projects. Every module is dependency-free.
 *
 * ml/   the engines: LightGBM trees and TreeSHAP, feature building, the fraud stream,
 *       the insurance model and the clustering algorithms, each matching Python.
 * site/ page-level helpers built on them: one loader per project that downloads the
 *       exported files (from the same release as this code) and returns everything a
 *       page needs, plus formatting, colours, the rain map layout and a parity self-test.
 */
export declare const VERSION = "2.0.0";
export * from "./ml/lgbm.js";
export * from "./ml/rain.js";
export * from "./ml/explain.js";
export * from "./ml/live.js";
export * from "./ml/fraud.js";
export * from "./ml/insurance.js";
export * from "./ml/clustering.js";
export * from "./site/load.js";
export * from "./site/format.js";
export * from "./site/colors.js";
export * from "./site/map.js";
export * from "./site/rain.js";
export * from "./site/fraud.js";
export * from "./site/insurance.js";
export * from "./site/iris.js";
export * from "./site/selftest.js";
