import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

export interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** A tooltip that follows the pointer and stays inside the viewport. */
export function Tooltip({ tip }: { tip: TipState | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  useLayoutEffect(() => {
    if (!tip || !ref.current) return;
    const { width, height } = ref.current.getBoundingClientRect();
    let left = tip.x + 14;
    let top = tip.y + 14;
    if (left + width > window.innerWidth - 8) left = tip.x - width - 14;
    if (top + height > window.innerHeight - 8) top = tip.y - height - 14;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [tip]);
  if (!tip) return null;
  return (
    <div className="tip" ref={ref} style={pos} role="tooltip">
      {tip.content}
    </div>
  );
}
