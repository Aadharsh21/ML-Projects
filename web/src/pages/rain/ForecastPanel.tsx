import { CheckIcon, CrossIcon, DropIcon, SunIcon } from "../../components/icons";
import { fmt } from "../../lib/format";
import type { Step } from "./explain";

export function Waterfall({
  start,
  steps,
  end,
  describe,
  startLabel,
}: {
  start: number;
  steps: Step[];
  end: number;
  describe: (label: string) => string | null;
  startLabel: string;
}) {
  const x = (p: number) => `${p * 100}%`;
  return (
    <figure className="waterfall">
      <figcaption className="visually-hidden">
        How each reading moved the forecast, starting from a typical day at {fmt.prob(start)} and ending at {fmt.prob(end)}.
      </figcaption>
      <div className="wf-row wf-edge">
        <span className="wf-label">{startLabel}</span>
        <span className="wf-track">
          <span className="wf-tick" style={{ left: x(start) }} />
        </span>
        <span className="wf-delta">{fmt.prob(start)}</span>
      </div>
      {steps.map((s) => {
        const lo = Math.min(s.from, s.to);
        const hi = Math.max(s.from, s.to);
        const reading = s.other ? null : describe(s.label);
        return (
          <div className="wf-row" key={s.label}>
            <span className="wf-label">
              {s.label}
              {reading && <span className="wf-reading">{reading}</span>}
            </span>
            <span className="wf-track">
              <span
                className={s.value >= 0 ? "wf-bar up" : "wf-bar down"}
                style={{ left: x(lo), width: `max(2px, ${(hi - lo) * 100}%)` }}
              />
            </span>
            <span className="wf-delta">{fmt.points(s.to - s.from)}</span>
          </div>
        );
      })}
      <div className="wf-row wf-edge wf-end">
        <span className="wf-label">Forecast</span>
        <span className="wf-track">
          <span className="wf-tick strong" style={{ left: x(end) }} />
        </span>
        <span className="wf-delta">{fmt.prob(end)}</span>
      </div>
      <div className="wf-axis" aria-hidden="true">
        <span />
        <span className="wf-scale">
          <i style={{ left: "0%" }}>0%</i>
          <i style={{ left: "50%" }}>50%</i>
          <i style={{ left: "100%" }}>100%</i>
        </span>
        <span />
      </div>
      <div className="legend wf-legend">
        <span className="key">
          <span className="swatch" style={{ background: "var(--pos)" }} />
          Pushes toward rain
        </span>
        <span className="key">
          <span className="swatch" style={{ background: "var(--neg)" }} />
          Pushes toward dry
        </span>
      </div>
    </figure>
  );
}

export function Outcome({ actual, prob }: { actual: 0 | 1; prob: number }) {
  const right = (prob >= 0.5 ? 1 : 0) === actual;
  return (
    <div className="outcome">
      <span className="chip">
        {actual ? <DropIcon style={{ color: "var(--rain)" }} /> : <SunIcon style={{ color: "var(--series-4)" }} />}
        {actual ? "It rained the next day" : "It stayed dry"}
      </span>
      <span className="chip" style={{ color: right ? "var(--good-text)" : "var(--critical)" }}>
        {right ? <CheckIcon /> : <CrossIcon />}
        {right ? "Forecast right" : "Forecast wrong"}
      </span>
    </div>
  );
}
