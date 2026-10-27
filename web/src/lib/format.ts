const pct0 = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 0 });
const pct1 = new Intl.NumberFormat("en-US", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const usd2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export const fmt = {
  pct: (v: number) => pct0.format(v),
  pct1: (v: number) => pct1.format(v),
  /** probabilities near 0 or 1 keep a decimal so 99.6% doesn't read as 100% */
  prob: (v: number) => (v > 0.005 && v < 0.995 ? pct0.format(v) : v < 0.0001 ? "<0.01%" : v > 0.9999 ? ">99.99%" : pct1.format(v)),
  usd: (v: number) => usd0.format(v),
  usd2: (v: number) => usd2.format(v),
  signedUsd: (v: number) => (v >= 0 ? "+" : "−") + usd0.format(Math.abs(v)),
  int: (v: number) => int.format(v),
  dec: (v: number, digits = 1) => v.toFixed(digits),
  signed: (v: number, digits = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(digits),
  points: (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(Math.round(v * 100))} pts`,
};

const dayFmt = new Intl.DateTimeFormat("en-AU", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const shortFmt = new Intl.DateTimeFormat("en-AU", { day: "numeric", month: "short", timeZone: "UTC" });
const monthFmt = new Intl.DateTimeFormat("en-AU", { month: "short", year: "numeric", timeZone: "UTC" });

export const fmtDate = {
  day: (d: Date) => dayFmt.format(d),
  short: (d: Date) => shortFmt.format(d),
  month: (d: Date) => monthFmt.format(d),
  iso: (d: Date) => d.toISOString().slice(0, 10),
};
