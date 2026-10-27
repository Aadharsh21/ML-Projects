/**
 * Loading the exported files. Paths resolve relative to this module, so a page that
 * imports the engine from a tagged release on a CDN always gets the data files from
 * the same release.
 */
/** Reads one file given its path below public/data, such as "rain/model.json". */
export type Fetcher = (path: string) => Promise<unknown>;
export interface LoadOptions {
    /** Called with overall progress from 0 to 1 and a short description of the current step. */
    onProgress?: (fraction: number, stage: string) => void;
    /** Base URL of the data files. Defaults to web/public/data next to this module. */
    dataUrl?: string;
    /** Replaces fetching entirely, for tests or for a site that bundles the JSON itself. */
    fetcher?: Fetcher;
}
/** web/public/data, next to this module (works from src/ and dist/, on a CDN or a local server). */
export declare const DATA_URL: string;
/** Reference outputs exported by the Python side, used by selfTest(). */
export declare const REFERENCE_URL: string;
/** Fetch and parse a JSON file once per page; later calls share the same promise. */
export declare function fetchJSON<T = unknown>(url: string): Promise<T>;
/** Load several data files in parallel, reporting progress as each one arrives. */
export declare function loadFiles<T extends Record<string, unknown>>(files: {
    [K in keyof T]: string;
}, options?: LoadOptions, share?: number, stage?: string): Promise<T>;
/** Yield to the browser so long loops keep the page responsive. */
export declare const nextFrame: () => Promise<void>;
