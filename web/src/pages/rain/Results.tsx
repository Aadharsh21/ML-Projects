import { scaleLinear } from "d3-scale";
import { DataTable, HBars } from "../../components/charts";
import { fmt } from "../../lib/format";
import type { RainMeta } from "../../ml/rain";

const ORDER = ["majority", "persistence", "v1_logistic", "v2_logistic", "v2_lightgbm"];

function Calibration({ meta }: { meta: RainMeta }) {
  const size = 300;
  const m = { l: 44, r: 12, t: 12, b: 36 };
  const x = scaleLinear().domain([0, 1]).range([m.l, size - m.r]);
  const y = scaleLinear().domain([0, 1]).range([size - m.b, m.t]);
  const pts = meta.calibration;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="calibration" role="img" aria-label="Calibration: forecast probability against how often it actually rained">
      <g className="grid">
        {ticks.map((t) => (
          <line key={`y${t}`} x1={m.l} x2={size - m.r} y1={y(t)} y2={y(t)} />
        ))}
      </g>
      <g className="axis">
        {ticks.map((t) => (
          <text key={`yl${t}`} x={m.l - 8} y={y(t) + 4} textAnchor="end">
            {fmt.pct(t)}
          </text>
        ))}
        {ticks.map((t) => (
          <text key={`xl${t}`} x={x(t)} y={size - m.b + 18} textAnchor="middle">
            {fmt.pct(t)}
          </text>
        ))}
        <text x={(m.l + size - m.r) / 2} y={size - 2} textAnchor="middle">
          Forecast chance of rain
        </text>
      </g>
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="cal-diagonal" />
      <polyline points={pts.map((p) => `${x(p.mean_predicted)},${y(p.observed)}`).join(" ")} className="cal-line" />
      {pts.map((p) => (
        <circle key={p.lo} cx={x(p.mean_predicted)} cy={y(p.observed)} r={4.5} className="cal-dot">
          <title>{`Forecasts of ${fmt.pct(p.lo)}-${fmt.pct(p.hi)} (${fmt.int(p.count)} days): rained on ${fmt.pct1(p.observed)}`}</title>
        </circle>
      ))}
    </svg>
  );
}

export function Results({ meta }: { meta: RainMeta }) {
  const rows = ORDER.map((k) => ({ key: k, ...meta.models[k] }));
  const test = meta.splits.test;
  return (
    <section className="section wrap" aria-labelledby="rain-how">
      <div className="section-head">
        <h2 id="rain-how">How the model works</h2>
      </div>
      <div className="two-col">
        <div className="prose">
          <p>
            Each row of the data is one day at one station: the morning and afternoon readings, plus whether more than
            1&nbsp;mm of rain fell by 9am the next day. The model is a LightGBM ensemble of {meta.best_iteration} small
            decision trees. It learns from 2007 to 2014, tunes itself on 2015, and is graded on {fmt.int(test.rows)} days
            from 2016 and 2017 that it never saw, the same split the first version used.
          </p>
          <p>
            <strong>What changed from v1.</strong> The first version scaled and imputed its inputs using every row,
            including the test years, so a little of the answer leaked into training. v2 fits all preprocessing on
            training days only. It adds features a forecaster would look at: how pressure, humidity, cloud and
            temperature moved between 9am and 3pm, the time of year, where the station is, and wind directions as
            compass vectors so that north-north-west sits next to north. Trees replace the linear model and route
            missing readings on their own instead of filling them with averages.
          </p>
          <p>
            A logistic regression given the same new features stays at {fmt.pct1(meta.models.v2_logistic.test.accuracy)},
            so the gain comes from letting the model combine readings, like high humidity <em>and</em> falling pressure.
            When it says 70%, it rains on about 70% of those days (see the chart), which is what makes the percentages worth showing.
          </p>
        </div>
        <figure>
          <div className="fig-title">Forecasts match reality</div>
          <Calibration meta={meta} />
          <figcaption>
            Each dot groups test days by forecast. The closer to the diagonal, the more the percentages can be taken at
            face value.
          </figcaption>
        </figure>
      </div>

      <h3 className="sub" style={{ marginTop: 40 }}>Graded on 2016-2017</h3>
      <DataTable
        rows={rows}
        highlight={(r) => r.key === "v2_lightgbm"}
        caption="Model comparison on the 2016-2017 test days"
        columns={[
          { key: "model", label: "Model", value: (r) => r.label },
          { key: "acc", label: "Accuracy", num: true, value: (r) => fmt.pct1(r.test.accuracy) },
          { key: "auc", label: "ROC AUC", num: true, value: (r) => r.test.roc_auc.toFixed(3) },
          { key: "ap", label: "Avg. precision", num: true, value: (r) => r.test.avg_precision.toFixed(3) },
          { key: "brier", label: "Brier score (lower is better)", num: true, value: (r) => r.test.brier.toFixed(3) },
        ]}
      />
      <p className="note" style={{ marginTop: 10 }}>
        95% bootstrap interval for v2: accuracy {fmt.pct1(meta.ci95.accuracy[0])} to {fmt.pct1(meta.ci95.accuracy[1])}, ROC AUC{" "}
        {meta.ci95.roc_auc[0].toFixed(3)} to {meta.ci95.roc_auc[1].toFixed(3)}. It rained on {fmt.pct1(test.rain_rate)} of
        test days, so always saying "no rain" already scores {fmt.pct1(1 - test.rain_rate)}.
      </p>

      <h3 className="sub" style={{ marginTop: 40 }}>What the model listens to</h3>
      <div style={{ maxWidth: 640 }}>
        <HBars
          label="Average impact of each reading on the forecast"
          rows={meta.importance.slice(0, 12).map((r) => ({ label: r.group, value: r.mean_abs_shap }))}
          format={(v) => v.toFixed(2)}
          color="var(--rain)"
        />
        <p className="note" style={{ marginTop: 8 }}>
          Average size of each reading's SHAP value on test days, in log-odds. Afternoon humidity dominates. Of the new
          features, station location and the 9am to 3pm pressure change matter most.
        </p>
      </div>
    </section>
  );
}
