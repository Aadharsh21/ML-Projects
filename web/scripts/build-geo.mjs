// Extracts Australia from world-atlas into a small GeoJSON file for the rain map.
// Run with `node scripts/build-geo.mjs`; the output is committed.
import { readFileSync, writeFileSync } from "node:fs";
import { feature } from "topojson-client";

const world = JSON.parse(readFileSync(new URL("../node_modules/world-atlas/countries-50m.json", import.meta.url)));
const countries = feature(world, world.objects.countries);
// round coordinates to ~100 m to keep the file small
const round = (c) => (typeof c[0] === "number" ? c.map((v) => Math.round(v * 1000) / 1000) : c.map(round));
// keep the mainland, Tasmania and coastal islands; drop far-flung territories such as
// Macquarie and Lord Howe islands, which would otherwise shrink the map to fit them
const nearMainland = (polygon) => {
  const ring = polygon[0];
  const lon = ring.reduce((s, p) => s + p[0], 0) / ring.length;
  const lat = ring.reduce((s, p) => s + p[1], 0) / ring.length;
  return lon > 112 && lon < 154.5 && lat > -44.5 && lat < -9;
};
const pick = (name, filter) => {
  const f = countries.features.find((x) => x.properties.name === name);
  if (!f) throw new Error(`${name} not found`);
  let coords = f.geometry.type === "MultiPolygon" ? f.geometry.coordinates : [f.geometry.coordinates];
  if (filter) coords = coords.filter(filter);
  return { type: "Feature", properties: { name }, geometry: { type: "MultiPolygon", coordinates: round(coords) } };
};
const out = { type: "FeatureCollection", features: [pick("Australia", nearMainland), pick("Norfolk Island")] };
writeFileSync(new URL("../public/data/rain/australia.json", import.meta.url), JSON.stringify(out));
console.log("features:", out.features.map((f) => f.properties.name).join(", "));
