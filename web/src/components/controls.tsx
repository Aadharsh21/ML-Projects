import { useId, type ReactNode } from "react";

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Switch({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="switch">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="track" aria-hidden="true" />
      {children}
    </label>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
  mark,
  missing,
  accent,
  disabled,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display?: ReactNode;
  /** a reference tick, e.g. the recorded value */
  mark?: number;
  missing?: boolean;
  accent?: string;
  disabled?: boolean;
}) {
  const id = useId();
  const clamped = Math.min(max, Math.max(min, value));
  const pct = ((clamped - min) / (max - min)) * 100;
  return (
    <div className="field">
      <div className="field-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id} className={missing ? "missing" : undefined}>
          {display ?? value}
        </output>
      </div>
      <div
        className={`range${missing ? " is-missing" : ""}`}
        style={{ ["--pct" as string]: `${missing ? 0 : pct}%`, ["--track-fill" as string]: accent }}
      >
        {mark !== undefined && Number.isFinite(mark) && (
          <span className="mark" style={{ left: `${((Math.min(max, Math.max(min, mark)) - min) / (max - min)) * 100}%` }} />
        )}
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={clamped}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
    </div>
  );
}
