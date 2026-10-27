/**
 * Ready-to-draw SVG layout of Australia and the 49 rain stations: a Lambert conformal
 * conic projection (standard parallels 18°S and 36°S, centred on 134°E) fitted to the
 * mainland, with Norfolk Island drawn in an inset because it sits ~1,400 km offshore.
 */
const RAD = Math.PI / 180;
const tany = (phi) => Math.tan((Math.PI / 2 + phi) / 2);
/** Raw conformal conic (as d3-geo's geoConicConformal), y pointing up. */
function conicConformal(parallel0, parallel1, centerLon) {
    const p0 = parallel0 * RAD;
    const p1 = parallel1 * RAD;
    const n = Math.log(Math.cos(p0) / Math.cos(p1)) / Math.log(tany(p1) / tany(p0));
    const f = (Math.cos(p0) * Math.pow(tany(p0), n)) / n;
    return (lon, lat) => {
        const lambda = (lon - centerLon) * RAD;
        let phi = lat * RAD;
        if (f > 0)
            phi = Math.max(phi, -Math.PI / 2 + 1e-6);
        else
            phi = Math.min(phi, Math.PI / 2 - 1e-6);
        const r = f / Math.pow(tany(phi), n);
        return [r * Math.sin(n * lambda), f - r * Math.cos(n * lambda)];
    };
}
const LABELLED = ["Perth", "Darwin", "Brisbane", "Sydney", "Melbourne", "Adelaide", "Hobart", "AliceSprings", "Cairns", "Canberra"];
// names go to the right of the dot unless that would cover neighbours or the inset
const LABEL_SIDE = {
    Perth: "left", PerthAirport: "left", PearceRAAF: "left", Witchcliffe: "left", Albany: "below", Walpole: "below",
    Adelaide: "left", Nuriootpa: "above", Brisbane: "left", GoldCoast: "left", CoffsHarbour: "left",
    Canberra: "below", Tuggeranong: "below", MountGinini: "left", Melbourne: "below", MelbourneAirport: "left",
    Watsonia: "above", Hobart: "right", NorfolkIsland: "below", Sydney: "right", SydneyAirport: "below",
    Wollongong: "below", Penrith: "left", Richmond: "above", BadgerysCreek: "left", NorahHead: "right",
    Newcastle: "right", Williamtown: "right",
};
const NORFOLK = [167.95, -29.04];
function pathOf(polygons, point) {
    const parts = [];
    for (const polygon of polygons) {
        for (const ring of polygon) {
            ring.forEach(([lon, lat], i) => {
                const [x, y] = point(lon, lat);
                parts.push(`${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`);
            });
            parts.push("Z");
        }
    }
    return parts.join("");
}
/** Lay out the map for an SVG with the given viewBox size (default 760 x 600). */
export function rainMap(geo, stations, width = 760, height = 600) {
    const mainland = geo.features.find((f) => f.properties.name === "Australia");
    if (!mainland)
        throw new Error("map data has no Australia feature");
    const norfolk = geo.features.find((f) => f.properties.name === "Norfolk Island");
    const raw = conicConformal(-18, -36, 134);
    // fit the mainland into the extent, leaving room on the right for the inset
    const extent = { x0: 12, y0: 12, x1: width - 96, y1: height - 12 };
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (const polygon of mainland.geometry.coordinates)
        for (const ring of polygon)
            for (const [lon, lat] of ring) {
                const [x, y] = raw(lon, lat);
                bx0 = Math.min(bx0, x);
                bx1 = Math.max(bx1, x);
                by0 = Math.min(by0, -y);
                by1 = Math.max(by1, -y);
            }
    const k = Math.min((extent.x1 - extent.x0) / (bx1 - bx0), (extent.y1 - extent.y0) / (by1 - by0));
    const tx = extent.x0 + ((extent.x1 - extent.x0) - k * (bx1 - bx0)) / 2 - k * bx0;
    const ty = extent.y0 + ((extent.y1 - extent.y0) - k * (by1 - by0)) / 2 - k * by0;
    const project = (lon, lat) => {
        const [x, y] = raw(lon, lat);
        return [tx + k * x, ty - k * y];
    };
    // Norfolk Island at six times the scale, centred in a box level with the coast at 29°S
    const insetY = project(153, NORFOLK[1])[1];
    const box = { x: width - 74, y: insetY - 34, width: 64, height: 68 };
    const [nx, ny] = raw(NORFOLK[0], NORFOLK[1]);
    const projectInset = (lon, lat) => {
        const [x, y] = raw(lon, lat);
        return [box.x + box.width / 2 + 6 * k * (x - nx), box.y + box.height / 2 - 6 * k * (y - ny)];
    };
    const points = {};
    for (const s of stations) {
        points[s.id] = s.id === "NorfolkIsland" ? [box.x + box.width / 2, box.y + box.height / 2] : project(s.lon, s.lat);
    }
    return {
        width,
        height,
        land: pathOf(mainland.geometry.coordinates, project),
        inset: { ...box, land: norfolk ? pathOf(norfolk.geometry.coordinates, projectInset) : "", label: "Norfolk Is." },
        points,
        labelled: LABELLED.filter((id) => id in points),
        labelFor(id, radius) {
            const [x, y] = points[id] ?? [0, 0];
            switch (LABEL_SIDE[id] ?? "right") {
                case "left": return { x: x - (radius + 7), y: y + 4, anchor: "end" };
                case "below": return { x, y: y + radius + 15, anchor: "middle" };
                case "above": return { x, y: y - (radius + 7), anchor: "middle" };
                default: return { x: x + radius + 7, y: y + 4, anchor: "start" };
            }
        },
        stationAt(x, y, maxDistance = 24) {
            let best = null;
            let bestD = maxDistance;
            for (const [id, [px, py]] of Object.entries(points)) {
                const d = Math.hypot(px - x, py - y);
                if (d <= bestD) {
                    bestD = d;
                    best = id;
                }
            }
            return best;
        },
    };
}
