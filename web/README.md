# Browser inference package

Everything a website needs to run the four models client-side: the exported
models and data in [`public/data`](public/data), a dependency-free TypeScript
engine in [`src/ml`](src/ml) that reproduces the Python models exactly, and
page-level helpers in [`src/site`](src/site) that turn it into one loader per
project.

Nothing here needs a server. [`dist/`](dist) is the same code compiled to plain
ES modules and committed, so any page can import a tagged release straight from
jsDelivr. The data files are fetched from the same release, relative to the
module, so the engine and the models can never get out of step:

```ts
const ml = await import("https://cdn.jsdelivr.net/gh/Aadharsh21/ML-Projects@v2.0.0/web/dist/index.js");

const rain = await ml.loadRain({ onProgress: (fraction, stage) => console.log(stage, fraction) });
const f = rain.forecast("Sydney", rain.dayOf("2016-07-08"));
f.probability; // 0.648: chance of more than 1 mm of rain by 9am the next day
f.sentence; // "Rain looks likely mainly because of humidity at 3pm (80%) and sunshine (0.0 h)."
```

## Page helpers

Each loader downloads its project's files, builds the engine objects and returns
what a page needs. All of them accept `{ onProgress, dataUrl, fetcher }`.

| Loader | Returns |
|---|---|
| `loadRain()` | Scores all 25,710 test station-days on load. `forecast(station, day, edits?)` gives the chance of rain, what really happened, a SHAP waterfall and a one-sentence reason, with any readings edited; `dayView(day)` colours the map; `history(station)`, `rainShare`, `nearest(lat, lon)`, `fetchLive()` for today's Open-Meteo weather, and `map()` for a ready-to-draw SVG layout of Australia |
| `loadFraud()` | The 71,202-transaction test stream with both models' scores. `evaluate(policy, upTo?)` for any threshold or expected-loss rule and review cost, `indexAt(hour)` and `events()` to replay the 9.3 hours, `tradeoff()` across cut-offs, and `explain(index)` for SHAP reasons |
| `loadInsurance()` | `quote(profile)`: expected charges, the typical (median) charge and middle-80% range, and exact Shapley values from the average customer; `customers` for the scatter, `byAge()` |
| `loadIris()` | `run({ method, k, seed, scaled, restarts })` records k-means or Gaussian-mixture EM one frame per iteration; `score()`, `project()` into the PCA plane or any two measurements, `ellipse()` for mixture components |
| `selfTest()` | Reruns Python's reference rows in the browser and reports the largest difference for each of ten checks |

`RAIN_READINGS` and `RAIN_WINDS` describe the sliders for the "what if" readings,
`chanceColor()` and `chanceGradient()` give the blue ramp for chances of rain, and
`formatChance()`, `formatUsd()` and friends keep figures consistent.

## Engine

| File | What it does |
|---|---|
| [`lgbm.ts`](src/ml/lgbm.ts) | `Forest`: evaluates exported LightGBM models (`predictProba`, `predictRaw`) and computes SHAP values with a port of LightGBM's TreeSHAP (`shap`) |
| [`rain.ts`](src/ml/rain.ts) | `RainDays` decodes the 2016-2017 test days; `rainFeatures` builds the 32 model inputs from raw readings; `groupShap` sums SHAP values into readable groups |
| [`explain.ts`](src/ml/explain.ts) | Turns rain SHAP values into a probability waterfall and a one-sentence reason |
| [`live.ts`](src/ml/live.ts) | `fetchLive`: today's weather for every station from the Open-Meteo API (free, no key, CORS-enabled), mapped onto the model's inputs |
| [`fraud.ts`](src/ml/fraud.ts) | `Stream` replays the scored test transactions and computes confusion matrices, dollar outcomes and trade-off curves for any threshold or rule; `fraudFeatures` builds inputs from a raw transaction |
| [`insurance.ts`](src/ml/insurance.ts) | `InsuranceModel`: charges, the middle-80% range and exact Shapley values against the average customer |
| [`clustering.ts`](src/ml/clustering.ts) | k-means and Gaussian-mixture EM that step one iteration at a time, k-means++ seeding (plain or scikit-learn's greedy variant), silhouette, adjusted Rand index and optimal cluster-to-species matching |

## Using the engine directly

```ts
import { Forest, RainDays, rainFeatures, groupShap, waterfall } from "./src";

const model = await (await fetch(`${BASE}/rain/model.json`)).json();
const days = new RainDays(await (await fetch(`${BASE}/rain/days.json`)).json());
const meta = await (await fetch(`${BASE}/rain/meta.json`)).json();
const forest = new Forest(model);

const row = days.rowFor("Sydney", 189); // day 189 after 2016-01-01
const obs = days.observation(row!); // edit obs.numeric to ask "what if?"
const x = rainFeatures(obs, model.preprocessing, forest.features, days.start);
forest.predictProba(x); // chance of rain by 9am the next day
const phi = forest.shap(x); // last entry is the base value
waterfall(phi[phi.length - 1], groupShap(phi, meta)); // biggest reasons first
```

```ts
const stream = new Stream(await (await fetch(`${BASE}/fraud/stream.json`)).json());
const flags = stream.flagged("v2", { kind: "threshold", probability: 0.5 });
stream.outcome(flags, 5); // { tp: 70, fp: 8, fn: 24, blocked: 8193.39, net: 7803.39, ... }
```

```ts
const ins = new InsuranceModel(await (await fetch(`${BASE}/insurance/model.json`)).json());
const me = { age: 30, male: 1, bmi: 31, children: 0, smoker: 1, region: 2 };
ins.predict(me); // expected yearly charges in dollars
ins.range(me); // { lo, median, hi }: where 8 in 10 similar customers land
```

## Data files

| Path | Contents |
|---|---|
| `rain/model.json` | LightGBM trees plus feature-engineering constants (`preprocessing`) |
| `rain/days.json` | Raw readings for all 25,710 test station-days (columnar) |
| `rain/meta.json` | Stations, feature groups, metrics for every model, calibration, feature impact, per-station accuracy |
| `rain/australia.json` | Outline of Australia and Norfolk Island (GeoJSON, from Natural Earth via world-atlas) |
| `fraud/model.json` | LightGBM trees |
| `fraud/stream.json` | The 71,202 test transactions: time, amount, fraud label, v1 and v2 scores (log-odds) |
| `fraud/inspect.json` | Model inputs for every fraud and every high-scoring transaction, for explanations |
| `fraud/meta.json` | Metrics for every model, the cost model, precision-recall curves, feature impact |
| `insurance/model.json` | Coefficients, residuals for the prediction ranges |
| `insurance/data.json` | All 1,338 customers (also the Shapley baseline) |
| `insurance/meta.json` | Cross-validated comparison, range coverage, group averages |
| `iris/data.json` | The 150 flowers, species and PCA projection |
| `iris/meta.json` | k sweep (inertia, silhouette, ARI), method comparison, BIC |

Every file is written by `python -m ml all` from the repository root. The map
outline comes from `npm run geo`, and `dist/` from `npm run build`.

## Tests

```
npm install
npm test
```

The tests load fixtures exported by the Python side and check that the engine
matches LightGBM's probabilities to 1e-9 and its SHAP values to 1e-8, that rain
features match Python bit for bit, that EM matches scikit-learn's
`GaussianMixture` step for step, and that the fraud stream reproduces the
published confusion matrices and dollar figures. The helper tests check that
scoring every rain day reproduces the published accuracy, that the playground
finds the published clusterings, and that `dist/` exports the same thing as the
source. CI rebuilds `dist/` and fails if it differs from what is committed.
