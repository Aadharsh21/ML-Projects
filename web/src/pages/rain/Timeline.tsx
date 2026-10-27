import { useEffect } from "react";
import { PauseIcon, PlayIcon } from "../../components/icons";
import { fmtDate } from "../../lib/format";
import { dateOf } from "../../ml/rain";

export function Timeline({
  start,
  lastDay,
  day,
  onDay,
  playing,
  onPlaying,
  rainShare,
}: {
  start: string;
  lastDay: number;
  day: number;
  onDay: (d: number) => void;
  playing: boolean;
  onPlaying: (p: boolean) => void;
  /** share of stations where it rained the next day, per day */
  rainShare: Float32Array;
}) {
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => onDay(day >= lastDay ? 0 : day + 1), 420);
    return () => window.clearInterval(id);
  }, [playing, day, lastDay, onDay]);

  const n = lastDay + 1;
  const months: { day: number; label: string }[] = [];
  for (let d = 0; d < n; d++) {
    const date = dateOf(start, d);
    if (date.getUTCDate() === 1 && date.getUTCMonth() % 3 === 0) months.push({ day: d, label: fmtDate.month(date) });
  }

  return (
    <div className="timeline">
      <div className="timeline-bars" aria-hidden="true">
        <svg viewBox={`0 0 ${n} 100`} preserveAspectRatio="none">
          {Array.from(rainShare, (v, d) => (
            <rect key={d} x={d + 0.1} width={0.8} y={100 - v * 100} height={v * 100} className={d === day ? "is-current" : undefined} />
          ))}
        </svg>
        <span className="timeline-now" style={{ left: `${((day + 0.5) / n) * 100}%` }} />
      </div>
      <input
        type="range"
        className="timeline-range"
        min={0}
        max={lastDay}
        step={1}
        value={day}
        onChange={(e) => {
          onPlaying(false);
          onDay(Number(e.target.value));
        }}
        aria-label="Day"
        aria-valuetext={fmtDate.day(dateOf(start, day))}
      />
      <div className="timeline-months" aria-hidden="true">
        {months.map((m) => (
          <span key={m.day} style={{ left: `${(m.day / n) * 100}%` }}>
            {m.label}
          </span>
        ))}
      </div>
      <div className="timeline-controls">
        <button type="button" className="btn icon" onClick={() => onPlaying(!playing)} aria-label={playing ? "Pause" : "Play through the days"}>
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>
        <button type="button" className="btn quiet" onClick={() => onDay(Math.max(0, day - 1))} aria-label="Previous day">
          Prev
        </button>
        <button type="button" className="btn quiet" onClick={() => onDay(Math.min(lastDay, day + 1))} aria-label="Next day">
          Next
        </button>
        <span className="note">Bars show the share of stations where it rained the next day.</span>
      </div>
    </div>
  );
}
