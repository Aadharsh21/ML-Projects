/** Turning rain-model SHAP values into a probability waterfall and a plain-English reason. */
import { sigmoid } from "./lgbm.js";
import type { Observation } from "./rain.js";

const signedFixed = (v: number, digits: number) => (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(digits);

export interface Step {
  label: string;
  value: number; // log-odds contribution
  from: number; // probability before this step
  to: number; // probability after
  other?: boolean;
}

/** Turn grouped SHAP values (log-odds) into a probability waterfall, biggest first. */
export function waterfall(base: number, groups: { label: string; value: number }[], top = 6) {
  const sorted = [...groups].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const shown = sorted.slice(0, top);
  const rest = sorted.slice(top);
  const items = [...shown.map((g) => ({ ...g, other: false }))];
  if (rest.length) items.push({ label: `${rest.length} other readings`, value: rest.reduce((s, g) => s + g.value, 0), other: true });
  let z = base;
  const steps: Step[] = items.map((g) => {
    const from = sigmoid(z);
    z += g.value;
    return { label: g.label, value: g.value, from, to: sigmoid(z), other: g.other };
  });
  return { start: sigmoid(base), end: sigmoid(z), steps };
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** A short reading for a group, e.g. "85%" for humidity at 3pm. */
export function describe(label: string, obs: Observation, compass: string[], stationName: string, date: Date): string | null {
  const n = obs.numeric;
  const v = (x: number, unit: string, digits = 0) => (Number.isNaN(x) ? "not recorded" : `${x.toFixed(digits)}${unit}`);
  const diff = (a: number, b: number, unit: string, digits = 0) =>
    Number.isNaN(a) || Number.isNaN(b) ? "not recorded" : `${signedFixed(a - b, digits)}${unit}`;
  const dir = (i: number | null) => (i === null ? "not recorded" : compass[i]);
  switch (label) {
    case "Humidity at 3pm": return v(n.Humidity3pm, "%");
    case "Humidity at 9am": return v(n.Humidity9am, "%");
    case "Humidity change, 9am to 3pm": return diff(n.Humidity3pm, n.Humidity9am, " pts");
    case "Pressure at 3pm": return v(n.Pressure3pm, " hPa", 1);
    case "Pressure at 9am": return v(n.Pressure9am, " hPa", 1);
    case "Pressure change, 9am to 3pm": return diff(n.Pressure3pm, n.Pressure9am, " hPa", 1);
    case "Sunshine": return v(n.Sunshine, " h", 1);
    case "Cloud at 3pm": return v(n.Cloud3pm, "/8");
    case "Cloud at 9am": return v(n.Cloud9am, "/8");
    case "Cloud change, 9am to 3pm": return diff(n.Cloud3pm, n.Cloud9am, "/8");
    case "Wind gust speed": return v(n.WindGustSpeed, " km/h");
    case "Wind gust direction": return dir(obs.wind.WindGustDir);
    case "Wind speed at 9am": return v(n.WindSpeed9am, " km/h");
    case "Wind speed at 3pm": return v(n.WindSpeed3pm, " km/h");
    case "Wind direction at 9am": return dir(obs.wind.WindDir9am);
    case "Wind direction at 3pm": return dir(obs.wind.WindDir3pm);
    case "Rainfall today": return v(n.Rainfall, " mm", 1);
    case "Rained today": return obs.rainToday ? "yes" : "no";
    case "Max temperature": return v(n.MaxTemp, "°C", 1);
    case "Min temperature": return v(n.MinTemp, "°C", 1);
    case "Temperature range": return diff(n.MaxTemp, n.MinTemp, "°C", 1).replace(/^\+/, "");
    case "Temperature at 9am": return v(n.Temp9am, "°C", 1);
    case "Temperature at 3pm": return v(n.Temp3pm, "°C", 1);
    case "Temperature change, 9am to 3pm": return diff(n.Temp3pm, n.Temp9am, "°C", 1);
    case "Evaporation": return v(n.Evaporation, " mm", 1);
    case "Time of year": return MONTHS[date.getUTCMonth()];
    case "Station location": return stationName;
    default: return null;
  }
}

/** One plain-English sentence about the biggest reasons behind a forecast. */
export function sentence(steps: Step[], end: number, describeFn: (label: string) => string | null): string {
  const raising = end >= 0.5;
  const reasons = steps
    .filter((s) => !s.other && (raising ? s.value > 0 : s.value < 0))
    .slice(0, 2)
    .map((s) => {
      const r = describeFn(s.label);
      return `${s.label.charAt(0).toLowerCase()}${s.label.slice(1)}${r ? ` (${r})` : ""}`;
    });
  if (!reasons.length) return "No single reading dominates this forecast.";
  const lead = raising ? "Rain looks likely mainly because of" : "Rain looks unlikely mainly because of";
  return `${lead} ${reasons.join(" and ")}.`;
}
