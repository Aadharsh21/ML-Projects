# Browser inference package

Everything a website needs to run the four models client-side: the exported
models and data in [`public/data`](public/data), and a dependency-free TypeScript
engine in [`src/ml`](src/ml) that reproduces the Python models exactly.

Nothing here needs a server. The files are plain JSON, so a site can bundle them
or fetch them straight from GitHub, for example:

```
https://raw.githubusercontent.com/Aadharsh21/ML-Projects/main/web/public/data/rain/model.json
```

## Engine

| File | What it does |
|---|---|
| [`lgbm.ts`](src/ml/lgbm.ts) | `Forest`: evaluates exported LightGBM models (`predictProba`, `predictRaw`) and computes SHAP values with a port of LightGBM's TreeSHAP (`shap`) |
| [`rain.ts`](src/ml/rain.ts) | `RainDays` decodes the 2016-2017 test days; `rainFeatures` builds the 32 model inputs from raw readings; `groupShap` sums SHAP values into readable groups |
| [`explain.ts`](src/ml/explain.ts) | Turns rain SHAP values into a probability waterfall and a one-sentence reason |
| [`live.ts`](src/ml/live.ts) | `fetchLive`: today's weather for every station from the Open-Meteo API (free, no key, CORS-enabled), mapped onto the model's inputs |
| [`fraud.ts`](src/ml/fraud.ts) | `Stream` replays the scored test transactions and computes confusion matrices, dollar outcomes and trade-off curves for any threshold or rule; `fraudFeatures` builds inputs from a raw transaction |
| [`insurance.ts`](src/ml/insurance.ts) | `InsuranceModel`: charges, the middle-80% range and exact Shapley values against the average customer |
| [`clustering.ts`](src/ml/clustering.ts) | k-means and Gaussian-mixture EM that step one iteration at a time, k-means++ seeding, silhouette, adjusted Rand index and optimal cluster-to-species matching |

## Using it

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
outline comes from `npm run geo`.

## Tests

```
npm install
npm test
```

The tests load fixtures exported by the Python side and check that the engine
matches LightGBM's probabilities to 1e-9 and its SHAP values to 1e-8, that rain
features match Python bit for bit, that EM matches scikit-learn's
`GaussianMixture` step for step, and that the fraud stream reproduces the
published confusion matrices and dollar figures.
