/** Small helpers for keeping page state in the URL hash, so views can be shared. */

export function readHash(): Record<string, string> {
  const out: Record<string, string> = {};
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  params.forEach((v, k) => (out[k] = v));
  return out;
}

export function writeHash(state: Record<string, string | number | undefined | null>) {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(state)) if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  const next = `#${params.toString()}`;
  if (next !== window.location.hash) history.replaceState(null, "", next);
}
