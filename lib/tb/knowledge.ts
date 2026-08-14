export const primaryClassValues = ["TB", "PNEUMONIA", "NORMAL"] as const;

export const subtypePredictionValues = [
  "DRUG_SENSITIVE",
  "DRUG_RESISTANT",
  "MDR_TB",
  "PRE_XDR_TB",
  "XDR_TB",
  "NOT_APPLICABLE",
] as const;

export const patientTypeValues = [
  "NEW",
  "RELAPSE",
  "TREATMENT_AFTER_FAILURE",
  "LOSS_TO_FOLLOW_UP",
  "TRANSFER_IN",
] as const;

export const comorbidityValues = [
  "HIV_POSITIVE",
  "HIV_NEGATIVE",
  "DIABETES_POSITIVE",
  "DIABETES_NEGATIVE",
  "RENAL_IMPAIRMENT",
  "HEPATIC_IMPAIRMENT",
] as const;

export type PrimaryClass = (typeof primaryClassValues)[number];
export type SubtypePrediction = (typeof subtypePredictionValues)[number];
export type PatientType = (typeof patientTypeValues)[number];
export type Comorbidity = (typeof comorbidityValues)[number];

export type PredictionData = {
  primary_class: PrimaryClass;
  subtype_prediction: SubtypePrediction;
  confidence_score: number;
};

export type PatientMetadata = {
  patient_type: PatientType;
  age: number;
  weight_kg: number;
  pregnancy_status: boolean;
  comorbidities: Comorbidity[];
};

export type TbKnowledgeChunk = {
  id: string;
  source: string;
  title: string;
  content: string;
  keywords: string[];
};

export const tbKnowledgeSource = "provided-tb-knowledge-source-v3";

export const tbKnowledgeChunks: TbKnowledgeChunk[] = [
  {
    content:
      "Chest X-ray is a triage and supportive tool only; it does not confirm TB or determine drug resistance. For presumptive TB, collect at least two sputum specimens. Test one with a WHO-recommended rapid molecular test such as GeneXpert, reserving the other for smear microscopy, culture, or follow-on testing.",
    id: "tb-ds-confirmation",
    keywords: ["x-ray", "specimen", "sputum", "genexpert", "smear", "culture"],
    source: tbKnowledgeSource,
    title: "Initial TB evaluation and specimen collection",
  },
  {
    content:
      "Drug-sensitive TB requires MTB complex DNA detection, rifampicin susceptibility, and confirmed isoniazid susceptibility through first-line LPA (katG/inhA) or phenotypic first-line DST. A rifampicin-susceptible result alone is provisional: it does not rule out isoniazid mono-resistant TB or confirm susceptibility to the full first-line drug set.",
    id: "tb-ds-full-confirmation",
    keywords: ["ds", "isoniazid", "lpa", "katg", "rifampicin"],
    source: tbKnowledgeSource,
    title: "Drug-Sensitive TB full diagnostic confirmation",
  },
  {
    content:
      "Rifampicin resistance detected through an rpoB mutation is operationally classified as MDR/RR-TB. First-line LPA can assess katG/inhA mutations, and second-line LPA should assess gyrA/gyrB fluoroquinolone resistance.",
    id: "tb-mdr-rr-confirmation",
    keywords: ["mdr", "rr", "rpoB", "lpa", "rifampicin"],
    source: tbKnowledgeSource,
    title: "MDR/RR-TB diagnostic confirmation",
  },
  {
    content:
      "Pre-XDR-TB is classified when rifampicin resistance and fluoroquinolone resistance are detected. Second-line LPA (such as MTBDRsl) detects gyrA/gyrB mutations and distinguishes MDR/RR-TB from Pre-XDR-TB.",
    id: "tb-pre-xdr-confirmation",
    keywords: ["pre-xdr", "fluoroquinolone", "gyrA", "gyrB", "lpa"],
    source: tbKnowledgeSource,
    title: "Pre-XDR-TB diagnostic confirmation",
  },
  {
    content:
      "XDR-TB requires rifampicin resistance plus fluoroquinolone resistance plus resistance to at least one Group A drug. Comprehensive phenotypic Culture and Drug Susceptibility Testing is required to assess and confirm Group A drug resistance.",
    id: "tb-xdr-confirmation",
    keywords: ["xdr", "group a", "dst", "culture", "resistance"],
    source: tbKnowledgeSource,
    title: "XDR-TB diagnostic confirmation",
  },
];

export function searchLocalTbKnowledge(query: string, topK = 4) {
  const tokens = query
    .toLowerCase()
    .split(/[^a-z0-9-]+/)
    .filter(Boolean);

  return tbKnowledgeChunks
    .map((chunk) => {
      const haystack = `${chunk.title} ${chunk.content} ${chunk.keywords.join(
        " "
      )}`.toLowerCase();
      const score = tokens.reduce(
        (total, token) => total + (haystack.includes(token) ? 1 : 0),
        0
      );

      return {
        content: chunk.content,
        id: chunk.id,
        metadata: { title: chunk.title },
        score,
        source: chunk.source,
      };
    })
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
