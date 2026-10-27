import type { FeatureCollection } from "geojson";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Segmented } from "../../components/controls";
import { LocateIcon } from "../../components/icons";
import { Fact, Loading, PageIntro, SiteFooter, SiteHeader } from "../../components/Layout";
import { probGradient } from "../../lib/colors";
import { loadAll } from "../../lib/data";
import { fmt, fmtDate } from "../../lib/format";
import { readHash, writeHash } from "../../lib/hash";
import { useColorScheme } from "../../lib/theme";
import { Forest, type ForestJSON } from "../../ml/lgbm";
import {
  RainDays,
  dateOf,
  groupShap,
  rainFeatures,
  type DaysJSON,
  type Observation,
  type RainMeta,
  type RainPreprocessing,
} from "../../ml/rain";
import { AustraliaMap, type MapValue } from "./AustraliaMap";
import { describe, sentence, waterfall } from "./explain";
import { Outcome, Waterfall } from "./ForecastPanel";
import { fetchLive, type LiveReading } from "./live";
import { NO_EDITS, Readings, applyEdits, hasEdits, type Edits } from "./Readings";
import { Results } from "./Results";
import { StationHistory } from "./StationHistory";
import { Timeline } from "./Timeline";

type RainModelJSON = ForestJSON & { preprocessing: RainPreprocessing };

interface Loaded {
  forest: Forest;
  prep: RainPreprocessing;
  days: RainDays;
  meta: RainMeta;
  geo: FeatureCollection;
  probs: Float32Array;
  rainShare: Float32Array;
}

type Mode = "past" | "live";
type LiveState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; readings: Map<string, LiveReading>; fetchedAt: Date };

const DEFAULT_STATION = "Sydney";
const DEFAULT_DATE = "2016-07-08";
const tick = () => new Promise((r) => setTimeout(r, 0));

function dayFromIso(start: string, iso: string): number {
  return Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86_400_000);
}

function haversine(a: [number, number], b: [number, number]) {
  const rad = Math.PI / 180;
  const dLat = (b[0] - a[0]) * rad;
  const dLon = (b[1] - a[1]) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

export function RainApp() {
  const scheme = useColorScheme();
  const [data, setData] = useState<Loaded | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("Loading the model and 25,710 days of readings");
  const [loadError, setLoadError] = useState<string | null>(null);

  const initial = useMemo(readHash, []);
  const [mode, setMode] = useState<Mode>(initial.mode === "live" ? "live" : "past");
  const [station, setStation] = useState(initial.station ?? DEFAULT_STATION);
  const [day, setDay] = useState<number | null>(null);
  const [edits, setEdits] = useState<Edits>(NO_EDITS);
  const [playing, setPlaying] = useState(false);
  const [live, setLive] = useState<LiveState>({ status: "idle" });
  const [locating, setLocating] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await loadAll<{ model: RainModelJSON; days: DaysJSON; meta: RainMeta; geo: FeatureCollection }>(
          { model: "rain/model.json", days: "rain/days.json", meta: "rain/meta.json", geo: "rain/australia.json" },
          (f) => setProgress(f * 0.75),
        );
        if (cancelled) return;
        setStage("Scoring every station-day in your browser");
        const forest = new Forest(d.model);
        const days = new RainDays(d.days);
        const probs = new Float32Array(days.length);
        const chunk = 2000;
        for (let s = 0; s < days.length; s += chunk) {
          for (let r = s; r < Math.min(days.length, s + chunk); r++) {
            probs[r] = forest.predictProba(rainFeatures(days.observation(r), d.model.preprocessing, forest.features, days.start));
          }
          setProgress(0.75 + (0.25 * s) / days.length);
          await tick();
          if (cancelled) return;
        }
        const rained = new Float32Array(days.lastDay + 1);
        const counts = new Float32Array(days.lastDay + 1);
        for (let r = 0; r < days.length; r++) {
          counts[days.day(r)]++;
          rained[days.day(r)] += days.rainTomorrow(r);
        }
        const rainShare = rained.map((v, i) => (counts[i] ? v / counts[i] : 0));
        setData({ forest, prep: d.model.preprocessing, days, meta: d.meta, geo: d.geo, probs, rainShare });
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // resolve the starting day once the data is in
  useEffect(() => {
    if (!data || day !== null) return;
    const wanted = initial.date ? dayFromIso(data.days.start, initial.date) : dayFromIso(data.days.start, DEFAULT_DATE);
    setDay(Math.min(data.days.lastDay, Math.max(0, Number.isFinite(wanted) ? wanted : 0)));
    if (!data.meta.stations.some((s) => s.id === station)) setStation(DEFAULT_STATION);
  }, [data, day, initial.date, station]);

  const loadLive = useCallback(async () => {
    if (!data) return;
    setLive({ status: "loading" });
    try {
      const readings = await fetchLive(data.meta.stations, data.days.start);
      setLive({ status: "ready", readings, fetchedAt: new Date() });
    } catch (e) {
      setLive({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, [data]);

  useEffect(() => {
    if (mode === "live" && live.status === "idle" && data) void loadLive();
  }, [mode, live.status, data, loadLive]);

  useEffect(() => {
    if (!data || day === null) return;
    writeHash({ mode: mode === "live" ? "live" : undefined, station, date: mode === "past" ? fmtDate.iso(dateOf(data.days.start, day)) : undefined });
  }, [data, mode, station, day]);

  const chooseStation = (id: string) => {
    setStation(id);
    setEdits(NO_EDITS);
  };
  const chooseDay = useCallback((d: number) => {
    setDay(d);
    setEdits(NO_EDITS);
  }, []);

  const locate = () => {
    if (!data || !navigator.geolocation) {
      setLocating("Location isn't available in this browser.");
      return;
    }
    setLocating("Finding your nearest station...");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const here: [number, number] = [pos.coords.latitude, pos.coords.longitude];
        let best = data.meta.stations[0];
        let bestKm = Infinity;
        for (const s of data.meta.stations) {
          const km = haversine(here, [s.lat, s.lon]);
          if (km < bestKm) {
            bestKm = km;
            best = s;
          }
        }
        chooseStation(best.id);
        setLocating(bestKm > 500 ? `Nearest station is ${best.name}, ${fmt.int(bestKm)} km away.` : `Nearest station: ${best.name} (${fmt.int(bestKm)} km).`);
      },
      () => setLocating("Couldn't get your location. Pick a station on the map instead."),
      { timeout: 10000 },
    );
  };

  const view = useMemo(() => {
    if (!data || day === null) return null;
    const { forest, prep, days, meta } = data;
    const stationMeta = meta.stations.find((s) => s.id === station) ?? meta.stations[0];
    let recorded: Observation | null = null;
    let liveReading: LiveReading | undefined;
    if (mode === "past") {
      const row = days.rowFor(station, day);
      recorded = row === undefined ? null : days.observation(row);
    } else if (live.status === "ready") {
      liveReading = live.readings.get(station);
      recorded = liveReading?.obs ?? null;
    }
    const mapValues = new Map<string, MapValue>();
    if (mode === "past") {
      for (const s of meta.stations) {
        const row = days.rowFor(s.id, day);
        mapValues.set(s.id, row === undefined ? { prob: null, actual: null } : { prob: data.probs[row], actual: days.rainTomorrow(row) });
      }
    } else if (live.status === "ready") {
      for (const s of meta.stations) {
        const r = live.readings.get(s.id);
        mapValues.set(s.id, {
          prob: r ? forest.predictProba(rainFeatures(r.obs, prep, forest.features, days.start)) : null,
          actual: null,
        });
      }
    }
    let scored = 0;
    let correct = 0;
    for (const v of mapValues.values()) {
      if (v.prob === null || v.actual === null) continue;
      scored++;
      if ((v.prob >= 0.5 ? 1 : 0) === v.actual) correct++;
    }
    const history = new Map<number, { prob: number; actual: 0 | 1 }>();
    for (const r of days.rowsFor(station)) history.set(days.day(r), { prob: data.probs[r], actual: days.rainTomorrow(r) });
    const stats = meta.per_station.find((s) => s.station === station);

    if (!recorded) return { stationMeta, mapValues, scored, correct, history, stats, forecast: null, liveReading };
    const obs = applyEdits(recorded, edits);
    const x = rainFeatures(obs, prep, forest.features, days.start);
    const prob = forest.predictProba(x);
    const recordedProb = hasEdits(edits) ? forest.predictProba(rainFeatures(recorded, prep, forest.features, days.start)) : prob;
    const phi = forest.shap(x);
    const wf = waterfall(phi[phi.length - 1], groupShap(phi, meta));
    const date = dateOf(days.start, obs.day);
    const describeFn = (label: string) => describe(label, obs, prep.compass, stationMeta.name, date);
    return {
      stationMeta,
      mapValues,
      scored,
      correct,
      history,
      stats,
      liveReading,
      forecast: { recorded, obs, prob, recordedProb, wf, describeFn, text: sentence(wf.steps, wf.end, describeFn), date },
    };
  }, [data, day, station, mode, live, edits]);

  const meta = data?.meta;
  const v2 = meta?.models.v2_lightgbm.test;
  const v1 = meta?.models.v1_logistic.test;

  return (
    <>
      <SiteHeader current="rain" />
      <main style={{ ["--accent" as string]: "var(--rain)" }}>
        <PageIntro
          title="Rain tomorrow"
          lede="Will it rain tomorrow? A model trained on ten years of daily weather at 49 Australian stations, graded on 2016 and 2017 days it never saw. Pick a station and a day, change the readings, or run it on today's weather."
          facts={
            v2 && v1 ? (
              <>
                <Fact value={fmt.pct1(v2.accuracy)} was={fmt.pct1(v1.accuracy)} label="right on unseen days" />
                <Fact value={v2.roc_auc.toFixed(3)} was={v1.roc_auc.toFixed(3)} label="ROC AUC" />
              </>
            ) : (
              <Fact value="86.1%" label="right on unseen days" />
            )
          }
          notebook="01-rain-tomorrow.ipynb"
          source="ml/rain.py"
        />

        {loadError ? (
          <div className="wrap">
            <p className="error">The model data didn't load ({loadError}). Refresh the page to try again.</p>
          </div>
        ) : !data || !view || day === null ? (
          <div className="wrap">
            <Loading label={stage} progress={progress} />
          </div>
        ) : (
          <>
            <section className="wrap instrument" aria-label="Forecast explorer">
              <div className="instrument-bar">
                <Segmented
                  label="Which readings"
                  value={mode}
                  onChange={(m) => {
                    setMode(m);
                    setEdits(NO_EDITS);
                    setPlaying(false);
                  }}
                  options={[
                    { value: "past", label: "Past days (2016-17)" },
                    { value: "live", label: "Today, live" },
                  ]}
                />
                <label className="station-pick">
                  <span className="visually-hidden">Station</span>
                  <select className="select" value={station} onChange={(e) => chooseStation(e.target.value)}>
                    {[...data.meta.stations]
                      .sort((a, b) => a.name.localeCompare(b.name))
                      .map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                  </select>
                </label>
                <button type="button" className="btn" onClick={locate}>
                  <LocateIcon />
                  Nearest to me
                </button>
                {locating && <span className="note">{locating}</span>}
              </div>

              <div className="instrument-grid">
                <div className="map-col">
                  <AustraliaMap
                    geo={data.geo}
                    stations={data.meta.stations}
                    values={view.mapValues}
                    selected={station}
                    onSelect={chooseStation}
                    scheme={scheme}
                    showActual={mode === "past"}
                  />
                  <div className="map-legend">
                    <div className="legend">
                      <span className="key">
                        <span className="gradient" style={{ background: probGradient(scheme) }} />
                        <span>Forecast chance of rain, 0 to 100%</span>
                      </span>
                      {mode === "past" && (
                        <span className="key">
                          <svg width="22" height="22" aria-hidden="true">
                            <circle cx="11" cy="11" r="9.5" fill="none" stroke="var(--ink)" strokeWidth="2" />
                            <circle cx="11" cy="11" r="6" fill="var(--rule-strong)" />
                          </svg>
                          It actually rained
                        </span>
                      )}
                    </div>
                    {mode === "past" && view.scored > 0 && (
                      <p className="map-score">
                        <strong>{view.correct}</strong> of {view.scored} stations forecast right on {fmtDate.day(dateOf(data.days.start, day))}
                      </p>
                    )}
                  </div>
                  {mode === "past" && (
                    <Timeline
                      start={data.days.start}
                      lastDay={data.days.lastDay}
                      day={day}
                      onDay={chooseDay}
                      playing={playing}
                      onPlaying={setPlaying}
                      rainShare={data.rainShare}
                    />
                  )}
                </div>

                <aside className="forecast-col" aria-live="polite">
                  <div className="forecast-head">
                    <h2>{view.stationMeta.name}</h2>
                    <p className="muted">
                      {mode === "past"
                        ? fmtDate.day(dateOf(data.days.start, day))
                        : view.liveReading
                          ? `Today, ${fmtDate.day(new Date(`${view.liveReading.date}T00:00:00Z`))}`
                          : "Today"}
                    </p>
                  </div>
                  {mode === "live" && live.status === "loading" && <p className="note">Fetching today's weather for all 49 stations...</p>}
                  {mode === "live" && live.status === "error" && (
                    <div className="error">
                      Couldn't reach the weather service ({live.message}).{" "}
                      <button type="button" className="btn" onClick={() => void loadLive()}>
                        Try again
                      </button>
                    </div>
                  )}
                  {view.forecast ? (
                    <>
                      <div className="readout">
                        <span className="readout-value">{fmt.prob(view.forecast.prob)}</span>
                        <span className="readout-label">chance of rain (over 1 mm) by 9am the next day</span>
                      </div>
                      {hasEdits(edits) && (
                        <p className="note">
                          With your changes. The recorded readings give {fmt.prob(view.forecast.recordedProb)}.
                        </p>
                      )}
                      {mode === "past" && <Outcome actual={view.forecast.recorded.rainTomorrow} prob={view.forecast.prob} />}
                      {mode === "live" && view.liveReading && (
                        <p className="note">
                          Readings from Open-Meteo's weather model for this location
                          {view.liveReading.modelRainMm !== null && <>, which itself expects {view.liveReading.modelRainMm} mm in the same window</>}
                          . Live forecasts are approximate: the model was trained on the stations' own instruments.
                        </p>
                      )}
                      <p className="because">{view.forecast.text}</p>
                      <Waterfall
                        start={view.forecast.wf.start}
                        steps={view.forecast.wf.steps}
                        end={view.forecast.wf.end}
                        describe={view.forecast.describeFn}
                        startLabel="Typical day"
                      />
                    </>
                  ) : (
                    mode === "past" && (
                      <p className="note">
                        {view.stationMeta.name} has no readings for this day. Pick another day or station.
                      </p>
                    )
                  )}
                </aside>
              </div>
            </section>

            {view.forecast && (
              <section className="section wrap" aria-labelledby="readings-title">
                <div className="section-head">
                  <h2 id="readings-title">What if the readings were different?</h2>
                  <p>
                    Drag any reading and the forecast above updates instantly. Grey ticks mark the recorded value.
                  </p>
                </div>
                <Readings
                  recorded={view.forecast.recorded}
                  edits={edits}
                  onEdits={setEdits}
                  compass={data.prep.compass}
                  forecastFields={view.liveReading?.forecastFields}
                />
              </section>
            )}

            <section className="section wrap" aria-labelledby="history-title">
              <div className="section-head">
                <h2 id="history-title">{view.stationMeta.name}, day by day</h2>
                {view.stats && (
                  <p>
                    Forecast right on {fmt.pct(view.stats.accuracy)} of {fmt.int(view.stats.days)} days in 2016-17. Rain fell
                    on {fmt.pct(view.stats.rain_rate)} of them.
                  </p>
                )}
              </div>
              <StationHistory
                start={data.days.start}
                lastDay={data.days.lastDay}
                values={view.history}
                selectedDay={mode === "past" ? day : -1}
                onDay={(d) => {
                  setMode("past");
                  chooseDay(d);
                  document.querySelector(".instrument")?.scrollIntoView({ behavior: "smooth", block: "start" });
                }}
                scheme={scheme}
              />
              <div className="legend" style={{ marginTop: 12 }}>
                <span className="key">
                  <span className="gradient" style={{ background: probGradient(scheme) }} />
                  Forecast chance of rain
                </span>
                <span className="key">
                  <svg width="12" height="12" aria-hidden="true">
                    <circle cx="6" cy="6" r="3" fill="var(--ink)" />
                  </svg>
                  Rained the next day
                </span>
                <span className="key muted">Click a day to open it</span>
              </div>
            </section>

            <Results meta={data.meta} />
          </>
        )}
      </main>
      <SiteFooter>
        <p>
          Data: {meta?.source ?? "Rain in Australia (Kaggle), Australian Bureau of Meteorology"}. Live readings:{" "}
          <a href="https://open-meteo.com/">Open-Meteo</a>.
        </p>
      </SiteFooter>
    </>
  );
}
