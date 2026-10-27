import { scaleLinear } from "d3-scale";
import type { ReactNode } from "react";

/** Horizontal bars with the value at the tip. One series: one colour, no legend. */
export function HBars({
  rows,
  format,
  color = "var(--series-1)",
  width = 520,
  label,
}: {
  rows: { label: string; value: number; highlight?: boolean }[];
  format: (v: number) => string;
  color?: string;
  width?: number;
  label: string;
}) {
  const labelW = 190;
  const valueW = 56;
  const rowH = 26;
  const max = Math.max(...rows.map((r) => r.value));
  const x = scaleLinear().domain([0, max]).range([0, width - labelW - valueW]);
  return (
    <svg viewBox={`0 0 ${width} ${rows.length * rowH}`} className="hbars" role="img" aria-label={label}>
      {rows.map((r, i) => {
        const w = Math.max(2, x(r.value));
        const y = i * rowH;
        return (
          <g key={r.label} transform={`translate(0,${y})`}>
            <text x={labelW - 10} y={rowH / 2 + 4} textAnchor="end" className="hbar-label">
              {r.label}
            </text>
            <path
              d={`M${labelW},${rowH / 2 - 6} h${w - 4} a4,4 0 0 1 4,4 v4 a4,4 0 0 1 -4,4 h${-(w - 4)} z`}
              style={{ fill: r.highlight === false ? "var(--rule-strong)" : color }}
            />
            <text x={labelW + w + 6} y={rowH / 2 + 4} className="hbar-value">
              {format(r.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export interface Column<T> {
  key: string;
  label: ReactNode;
  value: (row: T) => ReactNode;
  num?: boolean;
}

export function DataTable<T>({
  rows,
  columns,
  highlight,
  caption,
}: {
  rows: T[];
  columns: Column<T>[];
  highlight?: (row: T) => boolean;
  caption?: ReactNode;
}) {
  return (
    <div className="table-wrap">
      <table className="data">
        {caption && <caption className="visually-hidden">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.num ? "num" : undefined} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={highlight?.(r) ? "highlight" : undefined}>
              {columns.map((c) => (
                <td key={c.key} className={c.num ? "num" : undefined}>
                  {c.value(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
