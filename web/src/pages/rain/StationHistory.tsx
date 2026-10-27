import { useState } from "react";
import { Tooltip, type TipState } from "../../components/Tooltip";
import { probColor } from "../../lib/colors";
import { fmt, fmtDate } from "../../lib/format";
import { dateOf } from "../../ml/rain";

const CELL = 11;
const GAP = 2;
const PITCH = CELL + GAP;
const DAYS = ["Mon", "", "Wed", "", "Fri", "", "Sun"];

export function StationHistory({
  start,
  lastDay,
  values,
  selectedDay,
  onDay,
  scheme,
}: {
  start: string;
  lastDay: number;
  values: Map<number, { prob: number; actual: 0 | 1 }>;
  selectedDay: number;
  onDay: (d: number) => void;
  scheme: "light" | "dark";
}) {
  const [tip, setTip] = useState<TipState | null>(null);
  const offset = (dateOf(start, 0).getUTCDay() + 6) % 7; // weekday of the first day, Monday = 0
  const weeks = Math.ceil((lastDay + 1 + offset) / 7);
  const left = 30;
  const top = 18;
  const width = left + weeks * PITCH;
  const height = top + 7 * PITCH;
  const months: { x: number; label: string }[] = [];
  for (let d = 0; d <= lastDay; d++) {
    const date = dateOf(start, d);
    if (date.getUTCDate() === 1 || d === 0) {
      const col = Math.floor((d + offset) / 7);
      const label = date.getUTCMonth() === 0 || d === 0 ? fmtDate.month(date) : fmtDate.month(date).split(" ")[0];
      months.push({ x: left + col * PITCH, label });
    }
  }

  return (
    <div className="calendar-scroll">
      <svg width={width} height={height} className="calendar" role="img" aria-label="Calendar of daily forecasts at this station">
        {months.map((m) => (
          <text key={m.x} x={m.x} y={11} className="cal-month">
            {m.label}
          </text>
        ))}
        {DAYS.map((d, i) =>
          d ? (
            <text key={d} x={0} y={top + i * PITCH + CELL - 1} className="cal-month">
              {d}
            </text>
          ) : null,
        )}
        {Array.from({ length: lastDay + 1 }, (_, d) => {
          const col = Math.floor((d + offset) / 7);
          const row = (d + offset) % 7;
          const x = left + col * PITCH;
          const y = top + row * PITCH;
          const v = values.get(d);
          const isSel = d === selectedDay;
          return (
            <g
              key={d}
              transform={`translate(${x},${y})`}
              onClick={() => v && onDay(d)}
              onPointerMove={(e) =>
                setTip({
                  x: e.clientX,
                  y: e.clientY,
                  content: v ? (
                    <>
                      <div className="tip-value">{fmt.prob(v.prob)}</div>
                      <div>{fmtDate.day(dateOf(start, d))}</div>
                      <div className="tip-label">{v.actual ? "It rained the next day" : "Dry the next day"}</div>
                    </>
                  ) : (
                    <>
                      <div>{fmtDate.day(dateOf(start, d))}</div>
                      <div className="tip-label">No readings</div>
                    </>
                  ),
                })
              }
              onPointerLeave={() => setTip(null)}
              style={{ cursor: v ? "pointer" : "default" }}
            >
              <rect
                width={CELL}
                height={CELL}
                rx={2}
                className={v ? "cal-cell" : "cal-cell empty"}
                style={v ? { fill: probColor(v.prob, scheme) } : undefined}
              />
              {v?.actual === 1 && (
                <circle
                  cx={CELL / 2}
                  cy={CELL / 2}
                  r={2.2}
                  style={{ fill: v.prob > 0.5 ? "var(--page)" : "var(--ink)" }}
                />
              )}
              {isSel && <rect x={-1.5} y={-1.5} width={CELL + 3} height={CELL + 3} rx={3} className="cal-selected" />}
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}
