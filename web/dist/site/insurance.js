/**
 * The insurance page's calculator: expected yearly charges for any customer, the range
 * where 8 in 10 similar customers land, and exact Shapley values that split the
 * difference from the average customer across the six inputs.
 */
import { INPUT_KEYS, InsuranceModel, profileFromRow } from "../ml/insurance.js";
import { loadFiles } from "./load.js";
export const INSURANCE_LABELS = {
    age: "Age",
    male: "Sex",
    bmi: "BMI",
    children: "Children",
    smoker: "Smoker",
    region: "Region",
};
const GROUP_LABELS = {
    non_smoker: "Non-smokers",
    smoker_bmi_under_30: "Smokers, BMI under 30",
    smoker_bmi_30_plus: "Smokers, BMI 30 or more",
};
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
export async function loadInsurance(options = {}) {
    const files = await loadFiles({ model: "insurance/model.json", data: "insurance/data.json", meta: "insurance/meta.json" }, options, 1, "Downloading the model and 1,338 customers");
    const model = new InsuranceModel(files.model);
    const obese = files.model.obese_bmi;
    const groupOf = (p) => (!p.smoker ? "non_smoker" : p.bmi >= obese ? "smoker_bmi_30_plus" : "smoker_bmi_under_30");
    const charges = files.data.columns.indexOf("charges");
    const customers = files.data.rows.map((row) => {
        const profile = profileFromRow(row);
        return { profile, charges: row[charges], group: groupOf(profile) };
    });
    const background = customers.map((c) => c.profile);
    const regions = files.model.regions;
    const text = (key, p) => {
        switch (key) {
            case "age": return String(p.age);
            case "male": return p.male ? "male" : "female";
            case "bmi": return p.bmi.toFixed(1);
            case "children": return String(p.children);
            case "smoker": return p.smoker ? "yes" : "no";
            case "region": return capitalize(regions[p.region] ?? String(p.region));
        }
    };
    return {
        meta: files.meta,
        model,
        regions,
        customers,
        groupLabels: GROUP_LABELS,
        groupOf,
        quote(profile) {
            const { phi, base } = model.shapley(profile, background);
            const contributions = INPUT_KEYS.map((key, i) => ({ key, label: INSURANCE_LABELS[key], text: text(key, profile), value: phi[i] })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
            return { profile, group: groupOf(profile), charges: model.predict(profile), range: model.range(profile), base, contributions };
        },
        byAge(profile, from = 18, to = 64) {
            const out = [];
            for (let age = from; age <= to; age++) {
                const p = { ...profile, age };
                out.push({ age, charges: model.predict(p), ...model.range(p) });
            }
            return out;
        },
    };
}
