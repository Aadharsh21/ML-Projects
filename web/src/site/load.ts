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
export const DATA_URL = new URL("../../public/data/", import.meta.url).href;

/** Reference outputs exported by the Python side, used by selfTest(). */
export const REFERENCE_URL = new URL("../../tests/fixtures/", import.meta.url).href;

const cache = new Map<string, Promise<unknown>>();

/** Fetch and parse a JSON file once per page; later calls share the same promise. */
export function fetchJSON<T = unknown>(url: string): Promise<T> {
  let pending = cache.get(url);
  if (!pending) {
    pending = fetch(url).then((response) => {
      if (!response.ok) throw new Error(`${url} answered ${response.status}`);
      return response.json();
    });
    pending.catch(() => cache.delete(url));
    cache.set(url, pending);
  }
  return pending as Promise<T>;
}

/** Load several data files in parallel, reporting progress as each one arrives. */
export async function loadFiles<T extends Record<string, unknown>>(
  files: { [K in keyof T]: string },
  options: LoadOptions = {},
  share = 1,
  stage = "Downloading the model",
): Promise<T> {
  const read = options.fetcher ?? ((path: string) => fetchJSON(new URL(path, options.dataUrl ?? DATA_URL).href));
  const keys = Object.keys(files) as (keyof T)[];
  let done = 0;
  options.onProgress?.(0, stage);
  const values = await Promise.all(
    keys.map(async (key) => {
      const value = await read(files[key]);
      done++;
      options.onProgress?.((share * done) / keys.length, stage);
      return value;
    }),
  );
  return Object.fromEntries(keys.map((key, i) => [key, values[i]])) as T;
}

/** Yield to the browser so long loops keep the page responsive. */
export const nextFrame = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
