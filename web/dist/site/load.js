/**
 * Loading the exported files. Paths resolve relative to this module, so a page that
 * imports the engine from a tagged release on a CDN always gets the data files from
 * the same release.
 */
/** web/public/data, next to this module (works from src/ and dist/, on a CDN or a local server). */
export const DATA_URL = new URL("../../public/data/", import.meta.url).href;
/** Reference outputs exported by the Python side, used by selfTest(). */
export const REFERENCE_URL = new URL("../../tests/fixtures/", import.meta.url).href;
const cache = new Map();
/** Fetch and parse a JSON file once per page; later calls share the same promise. */
export function fetchJSON(url) {
    let pending = cache.get(url);
    if (!pending) {
        pending = fetch(url).then((response) => {
            if (!response.ok)
                throw new Error(`${url} answered ${response.status}`);
            return response.json();
        });
        pending.catch(() => cache.delete(url));
        cache.set(url, pending);
    }
    return pending;
}
/** Load several data files in parallel, reporting progress as each one arrives. */
export async function loadFiles(files, options = {}, share = 1, stage = "Downloading the model") {
    const read = options.fetcher ?? ((path) => fetchJSON(new URL(path, options.dataUrl ?? DATA_URL).href));
    const keys = Object.keys(files);
    let done = 0;
    options.onProgress?.(0, stage);
    const values = await Promise.all(keys.map(async (key) => {
        const value = await read(files[key]);
        done++;
        options.onProgress?.((share * done) / keys.length, stage);
        return value;
    }));
    return Object.fromEntries(keys.map((key, i) => [key, values[i]]));
}
/** Yield to the browser so long loops keep the page responsive. */
export const nextFrame = () => new Promise((resolve) => setTimeout(resolve, 0));
