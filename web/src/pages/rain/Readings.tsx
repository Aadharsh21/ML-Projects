import { useState } from "react";
import { Slider } from "../../components/controls";
import { ResetIcon } from "../../components/icons";
import type { NumericField, Observation, WindField } from "../../ml/rain";

export interface Edits {
  numeric: Partial<Record<NumericField, number>>;
  wind: Partial<Record<WindField, number | null>>;
}

export const NO_EDITS: Edits = { numeric: {}, wind: {} };

export function hasEdits(e: Edits) {
  return Object.keys(e.numeric).length > 0 || Object.keys(e.wind).length > 0;
}

/** Recorded readings with the user's edits on top. "Rained today" follows the
 * Bureau's definition (more than 1 mm by 9am), so it updates with rainfall. */
export function applyEdits(obs: Observation, e: Edits): Observation {
  const numeric = { ...obs.numeric, ...e.numeric };
  const rainToday = "Rainfall" in e.numeric ? ((numeric.Rainfall > 1 ? 1 : 0) as 0 | 1) : obs.rainToday;
  return { ...obs, numeric, wind: { ...obs.wind, ...e.wind }, rainToday };
}

interface FieldSpec {
  field: NumericField;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  digits: number;
}

const F = (field: NumericField, label: string, min: number, max: number, step: number, unit: string, digits = 0): FieldSpec => ({
  field, label, min, max, step, unit, digits,
});

const MAIN: FieldSpec[] = [
  F("Humidity3pm", "Humidity at 3pm", 0, 100, 1, "%"),
  F("Pressure3pm", "Pressure at 3pm", 980, 1040, 0.1, " hPa", 1),
  F("WindGustSpeed", "Strongest wind gust", 0, 135, 1, " km/h"),
  F("Sunshine", "Sunshine", 0, 14.5, 0.1, " h", 1),
  F("Cloud3pm", "Cloud at 3pm", 0, 8, 1, " eighths"),
  F("Rainfall", "Rain so far (24 h to 9am)", 0, 50, 0.2, " mm", 1),
];

const MORE: FieldSpec[] = [
  F("Humidity9am", "Humidity at 9am", 0, 100, 1, "%"),
  F("Pressure9am", "Pressure at 9am", 980, 1040, 0.1, " hPa", 1),
  F("Cloud9am", "Cloud at 9am", 0, 8, 1, " eighths"),
  F("WindSpeed9am", "Wind speed at 9am", 0, 90, 1, " km/h"),
  F("WindSpeed3pm", "Wind speed at 3pm", 0, 90, 1, " km/h"),
  F("MaxTemp", "Max temperature", -5, 48, 0.1, "°C", 1),
  F("MinTemp", "Min temperature", -8, 34, 0.1, "°C", 1),
  F("Temp9am", "Temperature at 9am", -7, 41, 0.1, "°C", 1),
  F("Temp3pm", "Temperature at 3pm", -5, 47, 0.1, "°C", 1),
  F("Evaporation", "Evaporation", 0, 20, 0.2, " mm", 1),
];

const WINDS: { field: WindField; label: string }[] = [
  { field: "WindGustDir", label: "Gust direction" },
  { field: "WindDir9am", label: "Wind at 9am from" },
  { field: "WindDir3pm", label: "Wind at 3pm from" },
];

export function Readings({
  recorded,
  edits,
  onEdits,
  compass,
  forecastFields,
}: {
  recorded: Observation;
  edits: Edits;
  onEdits: (e: Edits) => void;
  compass: string[];
  forecastFields?: string[];
}) {
  const [more, setMore] = useState(false);
  const current = applyEdits(recorded, edits);
  const field = (s: FieldSpec) => {
    const value = current.numeric[s.field];
    const missing = Number.isNaN(value);
    const changed = s.field in edits.numeric;
    const predicted = forecastFields?.includes(s.field);
    return (
      <div key={s.field} className={changed ? "reading is-changed" : "reading"}>
        <Slider
          label={
            <>
              {s.label}
              {predicted && <span className="tag">forecast</span>}
            </>
          }
          value={missing ? (s.min + s.max) / 2 : value}
          min={s.min}
          max={s.max}
          step={s.step}
          missing={missing}
          mark={changed ? recorded.numeric[s.field] : undefined}
          accent="var(--rain)"
          display={missing ? "not recorded" : `${value.toFixed(s.digits)}${s.unit}`}
          onChange={(v) => onEdits({ ...edits, numeric: { ...edits.numeric, [s.field]: v } })}
        />
      </div>
    );
  };
  return (
    <div className="readings">
      <div className="readings-grid">{MAIN.map(field)}</div>
      {more && (
        <>
          <div className="readings-grid">{MORE.map(field)}</div>
          <div className="readings-grid winds">
            {WINDS.map((w) => {
              const v = current.wind[w.field];
              return (
                <label key={w.field} className={w.field in edits.wind ? "reading is-changed wind" : "reading wind"}>
                  <span className="wind-label">{w.label}</span>
                  <select
                    className="select"
                    value={v === null ? "" : String(v)}
                    onChange={(e) =>
                      onEdits({ ...edits, wind: { ...edits.wind, [w.field]: e.target.value === "" ? null : Number(e.target.value) } })
                    }
                  >
                    <option value="">Not recorded</option>
                    {compass.map((c, i) => (
                      <option key={c} value={i}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              );
            })}
          </div>
        </>
      )}
      <div className="readings-actions">
        <button type="button" className="btn quiet" onClick={() => setMore(!more)} aria-expanded={more}>
          {more ? "Show fewer readings" : "Show all 19 readings"}
        </button>
        <button type="button" className="btn" disabled={!hasEdits(edits)} onClick={() => onEdits(NO_EDITS)}>
          <ResetIcon />
          Reset to recorded
        </button>
        <span className="note">
          Rained today: {current.rainToday ? "yes" : "no"} (more than 1 mm by 9am)
        </span>
      </div>
    </div>
  );
}
