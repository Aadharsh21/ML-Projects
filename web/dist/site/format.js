/** Number formatting shared by the pages, so every page writes figures the same way. */
const pct0 = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 0 });
const pct1 = new Intl.NumberFormat("en-US", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const int0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
/** A probability as a percentage. Values near 0 or 1 keep a decimal, so 99.6% doesn't read as 100%. */
export function formatChance(p) {
    if (Number.isNaN(p))
        return "–";
    if (p > 0.005 && p < 0.995)
        return pct0.format(p);
    if (p < 0.0001)
        return "<0.01%";
    if (p > 0.9999)
        return ">99.99%";
    return pct1.format(p);
}
export const formatPercent = (v, digits = 0) => (digits === 0 ? pct0 : pct1).format(v);
export const formatUsd = (v) => usd0.format(v);
export const formatSignedUsd = (v) => (v >= 0 ? "+" : "−") + usd0.format(Math.abs(v));
export const formatInt = (v) => int0.format(v);
/** A change in probability as percentage points, e.g. "+12 pts". */
export const formatPoints = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(Math.round(v * 100))} pts`;
