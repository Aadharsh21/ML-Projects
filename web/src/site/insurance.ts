/**
 * The insurance page's calculator: expected yearly charges for any customer, the range
 * where 8 in 10 similar customers land, and exact Shapley values that split the
 * difference from the average customer across the six inputs.
 */
import { INPUT_KEYS, InsuranceModel, profileFromRow, type InputKey, type InsuranceModelJSON, type Profile } from "../ml/insurance.js";
import { loadFiles, type LoadOptions } from "./load.js";

export type InsuranceGroup = "non_smoker" | "smoker_bmi_under_30" | "smoker_bmi_30_plus";

interface CvScore {
  mean: number;
  std: number;
}

export interface InsuranceMeta {
  rows: number;
  mean_charges: number;
  group_means: Record<InsuranceGroup, number>;
  /** 5x5 repeated cross-validation, in dollars for mae and rmse */
  models: Record<string, { label: string; cv: { r2: CvScore; mae: CvScore; rmse: CvScore } }>;
  /** how often the middle-80% range contained the real charge on held-out customers */
  interval: { target: number; coverage: number; mean_width: number; repeats: number };
  /** the v2 model's coefficients, in dollars */
  coefficients: Record<string, number>;
  source: string;
}

export interface InsuranceCustomer {
  profile: Profile;
  charges: number;
  group: InsuranceGroup;
}

export interface InsuranceContribution {
  key: InputKey;
  label: string;
  /** this profile's value, written out (e.g. "yes", "31.0") */
  text: string;
  /** dollars this input adds to (or takes off) the average customer's charges */
  value: number;
}

export interface InsuranceQuote {
  profile: Profile;
  group: InsuranceGroup;
  /** the model's expected yearly charges */
  charges: number;
  /** where 8 in 10 similar customers land, and the typical (median) charge */
  range: { lo: number; median: number; hi: number };
  /** the average customer's predicted charges: where the contributions start */
  base: number;
  /** largest effect first; they add up to charges - base */
  contributions: InsuranceContribution[];
}

export interface InsuranceCalculator {
  meta: InsuranceMeta;
  model: InsuranceModel;
  /** northeast, northwest, southeast, southwest (Profile.region indexes this) */
  regions: string[];
  customers: InsuranceCustomer[];
  /** labels for the three bands of customers */
  groupLabels: Record<InsuranceGroup, string>;
  groupOf(profile: Profile): InsuranceGroup;
  quote(profile: Profile): InsuranceQuote;
  /** expected charges and range across ages for an otherwise fixed profile */
  byAge(profile: Profile, from?: number, to?: number): { age: number; charges: number; lo: number; median: number; hi: number }[];
}

export const INSURANCE_LABELS: Record<InputKey, string> = {
  age: "Age",
  male: "Sex",
  bmi: "BMI",
  children: "Children",
  smoker: "Smoker",
  region: "Region",
};

const GROUP_LABELS: Record<InsuranceGroup, string> = {
  non_smoker: "Non-smokers",
  smoker_bmi_under_30: "Smokers, BMI under 30",
  smoker_bmi_30_plus: "Smokers, BMI 30 or more",
};

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export async function loadInsurance(options: LoadOptions = {}): Promise<InsuranceCalculator> {
  const files = await loadFiles<{ model: InsuranceModelJSON; data: { columns: string[]; rows: number[][] }; meta: InsuranceMeta }>(
    { model: "insurance/model.json", data: "insurance/data.json", meta: "insurance/meta.json" },
    options,
    1,
    "Downloading the model and 1,338 customers",
  );
  const model = new InsuranceModel(files.model);
  const obese = files.model.obese_bmi;
  const groupOf = (p: Profile): InsuranceGroup => (!p.smoker ? "non_smoker" : p.bmi >= obese ? "smoker_bmi_30_plus" : "smoker_bmi_under_30");
  const charges = files.data.columns.indexOf("charges");
  const customers = files.data.rows.map((row) => {
    const profile = profileFromRow(row);
    return { profile, charges: row[charges], group: groupOf(profile) };
  });
  const background = customers.map((c) => c.profile);
  const regions = files.model.regions;

  const text = (key: InputKey, p: Profile): string => {
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
      const contributions = INPUT_KEYS.map((key, i) => ({ key, label: INSURANCE_LABELS[key], text: text(key, profile), value: phi[i] })).sort(
        (a, b) => Math.abs(b.value) - Math.abs(a.value),
      );
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
