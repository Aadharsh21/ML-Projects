import { Delaunay } from "d3-delaunay";
import { geoConicConformal, geoPath } from "d3-geo";
import type { Feature, FeatureCollection } from "geojson";
import { useMemo, useRef, useState } from "react";
import { Tooltip, type TipState } from "../../components/Tooltip";
import { probColor } from "../../lib/colors";
import { fmt } from "../../lib/format";

export interface MapStation {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

export interface MapValue {
  prob: number | null;
  actual: 0 | 1 | null;
}

const W = 760;
const H = 600;
const LABELLED = new Set(["Perth", "Darwin", "Brisbane", "Sydney", "Melbourne", "Adelaide", "Hobart", "AliceSprings", "Cairns", "Canberra"]);
// labels go to the right of the dot unless that would cover neighbours or the inset
type Side = "left" | "right" | "below" | "above";
const LABEL_SIDE: Record<string, Side> = {
  Perth: "left", PerthAirport: "left", PearceRAAF: "left", Witchcliffe: "left", Albany: "below", Walpole: "below",
  Adelaide: "left", Nuriootpa: "above", Brisbane: "left", GoldCoast: "left", CoffsHarbour: "left",
  Canberra: "below", Tuggeranong: "below", MountGinini: "left", Melbourne: "below", MelbourneAirport: "left",
  Watsonia: "above", Hobart: "right", NorfolkIsland: "below", Sydney: "right", SydneyAirport: "below",
  Wollongong: "below", Penrith: "left", Richmond: "above", BadgerysCreek: "left", NorahHead: "right",
  Newcastle: "right", Williamtown: "right",
};
const labelPos = (side: Side, r: number) =>
  side === "left" ? { dx: -(r + 7), dy: 4, anchor: "end" as const }
    : side === "below" ? { dx: 0, dy: r + 15, anchor: "middle" as const }
      : side === "above" ? { dx: 0, dy: -(r + 7), anchor: "middle" as const }
        : { dx: r + 7, dy: 4, anchor: "start" as const };

export function AustraliaMap({
  geo,
  stations,
  values,
  selected,
  onSelect,
  scheme,
  showActual,
}: {
  geo: FeatureCollection;
  stations: MapStation[];
  values: Map<string, MapValue>;
  selected: string;
  onSelect: (id: string) => void;
  scheme: "light" | "dark";
  showActual: boolean;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [tip, setTip] = useState<TipState | null>(null);

  const layout = useMemo(() => {
    const mainland = geo.features.find((f) => f.properties?.name === "Australia") as Feature;
    const norfolk = geo.features.find((f) => f.properties?.name === "Norfolk Island") as Feature | undefined;
    const projection = geoConicConformal().parallels([-18, -36]).rotate([-134, 0]).fitExtent([[12, 12], [W - 96, H - 12]], mainland);
    const path = geoPath(projection);
    // Norfolk Island sits ~1,400 km east of the coast: draw it in an inset instead.
    const insetY = projection([153, -29.04])![1];
    const inset = { x: W - 74, y: insetY - 34, w: 64, h: 68 };
    const insetProjection = geoConicConformal().parallels([-18, -36]).rotate([-134, 0]).scale(projection.scale() * 6);
    const nf = insetProjection([167.95, -29.04])!;
    insetProjection.translate([insetProjection.translate()[0] + inset.x + inset.w / 2 - nf[0], insetProjection.translate()[1] + inset.y + inset.h / 2 - nf[1]]);
    const points = stations.map((s) => {
      if (s.id === "NorfolkIsland") return [inset.x + inset.w / 2, inset.y + inset.h / 2] as [number, number];
      return projection([s.lon, s.lat]) as [number, number];
    });
    return {
      land: path(mainland) ?? "",
      norfolkLand: norfolk ? geoPath(insetProjection)(norfolk) ?? "" : "",
      inset,
      points,
      delaunay: Delaunay.from(points),
    };
  }, [geo, stations]);

  const locate = (e: React.PointerEvent | React.MouseEvent) => {
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const y = ((e.clientY - rect.top) / rect.height) * H;
    const i = layout.delaunay.find(x, y);
    const [px, py] = layout.points[i];
    const scale = rect.width / W;
    return Math.hypot((px - x) * scale, (py - y) * scale) < 32 ? i : -1;
  };

  const onMove = (e: React.PointerEvent) => {
    const i = locate(e);
    if (i < 0) {
      setHover(null);
      setTip(null);
      return;
    }
    const s = stations[i];
    const v = values.get(s.id);
    setHover(s.id);
    setTip({
      x: e.clientX,
      y: e.clientY,
      content: (
        <>
          <div className="tip-value">{v?.prob == null ? "No data" : fmt.prob(v.prob)}</div>
          <div>{s.name}</div>
          {showActual && v?.actual != null && <div className="tip-label">{v.actual ? "It rained" : "It stayed dry"}</div>}
        </>
      ),
    });
  };

  const order = stations.map((_, i) => i).sort((a, b) => {
    const rank = (i: number) => (stations[i].id === selected ? 2 : stations[i].id === hover ? 1 : 0);
    return rank(a) - rank(b);
  });

  return (
    <div className="map">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Map of Australia with each weather station coloured by its forecast chance of rain"
        onPointerMove={onMove}
        onPointerLeave={() => {
          setHover(null);
          setTip(null);
        }}
        onClick={(e) => {
          const i = locate(e);
          if (i >= 0) onSelect(stations[i].id);
        }}
        style={{ cursor: hover ? "pointer" : "default", touchAction: "manipulation" }}
      >
        <path d={layout.land} className="map-land" />
        <g className="map-inset">
          <rect x={layout.inset.x} y={layout.inset.y} width={layout.inset.w} height={layout.inset.h} rx={6} />
          <path d={layout.norfolkLand} className="map-land" />
          <text x={layout.inset.x + layout.inset.w / 2} y={layout.inset.y + layout.inset.h + 15} textAnchor="middle" className="map-note">
            Norfolk Is.
          </text>
        </g>
        {order.map((i) => {
          const s = stations[i];
          const [x, y] = layout.points[i];
          const v = values.get(s.id);
          const isSel = s.id === selected;
          const r = isSel ? 9 : 7.5;
          return (
            <g key={s.id} transform={`translate(${x},${y})`}>
              {isSel && <circle r={r + 8} className="map-selected" />}
              {showActual && v?.actual === 1 && <circle r={r + 3.5} className="map-rained" />}
              <circle
                r={r}
                className={v?.prob == null ? "map-station no-data" : "map-station"}
                style={v?.prob == null ? undefined : { fill: probColor(v.prob, scheme) }}
              />
            </g>
          );
        })}
        {stations.map((s, i) => {
          if (!LABELLED.has(s.id) && s.id !== selected) return null;
          const [x, y] = layout.points[i];
          const isSel = s.id === selected;
          if (s.id === "NorfolkIsland") return null; // the inset already names it
          const { dx, dy, anchor } = labelPos(LABEL_SIDE[s.id] ?? "right", isSel ? 9 : 7.5);
          return (
            <text key={s.id} x={x + dx} y={y + dy} textAnchor={anchor} className={isSel ? "map-label is-selected" : "map-label"}>
              {s.name}
            </text>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}
