/** Number formatting shared by the pages, so every page writes figures the same way. */
/** A probability as a percentage. Values near 0 or 1 keep a decimal, so 99.6% doesn't read as 100%. */
export declare function formatChance(p: number): string;
export declare const formatPercent: (v: number, digits?: number) => string;
export declare const formatUsd: (v: number) => string;
export declare const formatSignedUsd: (v: number) => string;
export declare const formatInt: (v: number) => string;
/** A change in probability as percentage points, e.g. "+12 pts". */
export declare const formatPoints: (v: number) => string;
