/** Where the site root is relative to the current page (set on <html data-root>). */
export function siteRoot(): string {
  return document.documentElement.dataset.root ?? "./";
}

export function dataUrl(path: string): string {
  return `${siteRoot()}data/${path}`;
}

/** Fetch JSON, reporting download progress (0..1) when the size is known. */
export async function loadJSON<T>(path: string, onProgress?: (fraction: number) => void): Promise<T> {
  const response = await fetch(dataUrl(path));
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status})`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!onProgress || !response.body || !total) {
    const json = (await response.json()) as T;
    onProgress?.(1);
    return json;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(received / total, 1));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

/** Load several files, combining their progress into one fraction. */
export async function loadAll<T extends Record<string, unknown>>(
  files: { [K in keyof T]: string },
  onProgress?: (fraction: number) => void,
): Promise<T> {
  const keys = Object.keys(files) as (keyof T)[];
  const progress = new Map<keyof T, number>(keys.map((k) => [k, 0]));
  const report = () => onProgress?.([...progress.values()].reduce((a, b) => a + b, 0) / keys.length);
  const values = await Promise.all(
    keys.map((k) =>
      loadJSON(files[k], (f) => {
        progress.set(k, f);
        report();
      }),
    ),
  );
  return Object.fromEntries(keys.map((k, i) => [k, values[i]])) as T;
}
