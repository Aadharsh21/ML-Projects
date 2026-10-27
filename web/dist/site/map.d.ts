/**
 * Ready-to-draw SVG layout of Australia and the 49 rain stations: a Lambert conformal
 * conic projection (standard parallels 18°S and 36°S, centred on 134°E) fitted to the
 * mainland, with Norfolk Island drawn in an inset because it sits ~1,400 km offshore.
 */
export interface GeoFeature {
    type: "Feature";
    properties: {
        name: string;
    };
    geometry: {
        type: "MultiPolygon";
        coordinates: number[][][][];
    };
}
export interface GeoCollection {
    type: "FeatureCollection";
    features: GeoFeature[];
}
export interface MapStation {
    id: string;
    name: string;
    lat: number;
    lon: number;
}
export type LabelAnchor = "start" | "middle" | "end";
export interface RainMapLayout {
    width: number;
    height: number;
    /** SVG path data for the mainland, Tasmania and coastal islands */
    land: string;
    /** box for Norfolk Island, with its own path and a caption to place under it */
    inset: {
        x: number;
        y: number;
        width: number;
        height: number;
        land: string;
        label: string;
    };
    /** station id to its position in the SVG's coordinate system */
    points: Record<string, [number, number]>;
    /** the cities to label even when not selected */
    labelled: string[];
    /** where to put a station's name next to a dot of the given radius */
    labelFor(id: string, radius: number): {
        x: number;
        y: number;
        anchor: LabelAnchor;
    };
    /** the station within maxDistance of a point in SVG coordinates, or null */
    stationAt(x: number, y: number, maxDistance?: number): string | null;
}
/** Lay out the map for an SVG with the given viewBox size (default 760 x 600). */
export declare function rainMap(geo: GeoCollection, stations: MapStation[], width?: number, height?: number): RainMapLayout;
