import type {
  PatientMetadata,
  PredictionData,
  SubtypePrediction,
} from "./knowledge";

export type TbDecision = {
  classification: {
    primaryClass: PredictionData["primary_class"];
    subtype: SubtypePrediction;
    tbType:
      | "DRUG_SENSITIVE_TB"
      | "MDR_RR_TB"
      | "PRE_XDR_TB"
      | "XDR_TB"
      | "NOT_TB";
    confidenceScore: number;
  };
  regimen: {
    name: string;
    duration: string;
    medicines: string[];
    phaseSummary?: string[];
  } | null;
  routing: string[];
  safetyFlags: string[];
  doseContext: {
    ageGroup: "PEDIATRIC" | "ADULT";
    weightBand: "<30kg" | "30-37kg" | "38-54kg" | "55-70kg" | ">70kg";
  };
  evidence: { source: string; chunkId: string; note: string }[];
  disclaimer: string;
};

const referralPatientTypes = new Set([
  "RELAPSE",
  "TREATMENT_AFTER_FAILURE",
  "LOSS_TO_FOLLOW_UP",
]);

function getWeightBand(
  weightKg: number
): TbDecision["doseContext"]["weightBand"] {
  if (weightKg < 30) {
    return "<30kg";
  }
  if (weightKg <= 37) {
    return "30-37kg";
  }
  if (weightKg <= 54) {
    return "38-54kg";
  }
  if (weightKg <= 70) {
    return "55-70kg";
  }
  return ">70kg";
}

function getTbType(
  subtype: SubtypePrediction
): TbDecision["classification"]["tbType"] {
  switch (subtype) {
    case "DRUG_RESISTANT":
    case "MDR_TB":
      return "MDR_RR_TB";
    case "PRE_XDR_TB":
      return "PRE_XDR_TB";
    case "XDR_TB":
      return "XDR_TB";
    case "DRUG_SENSITIVE":
      return "DRUG_SENSITIVE_TB";
    default:
      return "NOT_TB";
  }
}

export function buildTbDecision({
  patientMetadata,
  predictionData,
}: {
  patientMetadata: PatientMetadata;
  predictionData: PredictionData;
}): TbDecision {
  const ageGroup = patientMetadata.age < 15 ? "PEDIATRIC" : "ADULT";
  const weightBand = getWeightBand(patientMetadata.weight_kg);

  if (predictionData.primary_class !== "TB") {
    return {
      classification: {
        confidenceScore: predictionData.confidence_score,
        primaryClass: predictionData.primary_class,
        subtype: "NOT_APPLICABLE",
        tbType: "NOT_TB",
      },
      disclaimer:
        "This decision support output is not a diagnosis and must be reviewed by a qualified clinician.",
      doseContext: { ageGroup, weightBand },
      evidence: [],
      regimen: null,
      routing: [
        "TB-specific regimen routing is not applicable unless the model primary class is TB.",
      ],
      safetyFlags: [],
    };
  }

  const tbType = getTbType(predictionData.subtype_prediction);
  const routing: string[] = [];
  const safetyFlags: string[] = [];

  if (referralPatientTypes.has(patientMetadata.patient_type)) {
    routing.push(
      "Previously treated or interrupted-treatment patient: register under the current NTP category and refer for GeneXpert / DR-TB testing rather than using the removed Category 2 retreatment regimen."
    );
  }

  if (ageGroup === "PEDIATRIC") {
    routing.push("Use pediatric TB dosing workflow and child weight bands.");
  }

  if (patientMetadata.pregnancy_status) {
    safetyFlags.push(
      "Pregnancy: avoid teratogenic injectable aminoglycosides such as kanamycin or amikacin and request specialist review for DR-TB medicines."
    );
  }

  if (patientMetadata.comorbidities.includes("HIV_POSITIVE")) {
    safetyFlags.push(
      "HIV positive: review ART timing and drug interactions, especially rifampicin interactions."
    );
  }

  if (patientMetadata.comorbidities.includes("DIABETES_POSITIVE")) {
    safetyFlags.push(
      "Diabetes: add glucose control and treatment response monitoring."
    );
  }

  if (patientMetadata.comorbidities.includes("RENAL_IMPAIRMENT")) {
    safetyFlags.push(
      "Renal impairment: require dose-spacing and renal monitoring review."
    );
  }

  if (patientMetadata.comorbidities.includes("HEPATIC_IMPAIRMENT")) {
    safetyFlags.push(
      "Hepatic impairment: review hepatotoxic first-line drugs and consider specialist-guided alternatives."
    );
  }

  switch (tbType) {
    case "DRUG_SENSITIVE_TB":
      return {
        classification: {
          confidenceScore: predictionData.confidence_score,
          primaryClass: "TB",
          subtype: predictionData.subtype_prediction,
          tbType,
        },
        disclaimer:
          "This decision support output is not a diagnosis or prescription and must be reviewed against current NTP/NNF guidance by a qualified clinician.",
        doseContext: { ageGroup, weightBand },
        evidence: [
          {
            chunkId: "tb-ds-confirmation",
            note: "DS-TB confirmation uses rifampicin-susceptible GeneXpert context and clinical history.",
            source: "provided-tb-knowledge-source",
          },
          {
            chunkId: "tb-ds-regimen",
            note: "Standard DS-TB regimen is 2HRZE/4HR.",
            source: "provided-tb-knowledge-source",
          },
        ],
        regimen: {
          duration: "6 months",
          medicines: ["Isoniazid", "Rifampicin", "Pyrazinamide", "Ethambutol"],
          name: "2HRZE / 4HR",
          phaseSummary: [
            "Intensive phase: 2 months HRZE",
            "Continuation phase: 4 months HR",
          ],
        },
        routing,
        safetyFlags,
      };
    case "MDR_RR_TB":
      return {
        classification: {
          confidenceScore: predictionData.confidence_score,
          primaryClass: "TB",
          subtype: predictionData.subtype_prediction,
          tbType,
        },
        disclaimer:
          "This decision support output is not a diagnosis or prescription and must be reviewed against current NTP/NNF guidance by a qualified clinician.",
        doseContext: { ageGroup, weightBand },
        evidence: [
          {
            chunkId: "tb-mdr-rr-regimen",
            note: "Rifampicin-resistant TB is operationally managed as MDR/RR-TB with all-oral BPaLM when eligible.",
            source: "provided-tb-knowledge-source",
          },
        ],
        regimen: {
          duration: "6 months",
          medicines: ["Bedaquiline", "Pretomanid", "Linezolid", "Moxifloxacin"],
          name: "BPaLM",
        },
        routing,
        safetyFlags,
      };
    case "PRE_XDR_TB":
      return {
        classification: {
          confidenceScore: predictionData.confidence_score,
          primaryClass: "TB",
          subtype: predictionData.subtype_prediction,
          tbType,
        },
        disclaimer:
          "This decision support output is not a diagnosis or prescription and must be reviewed against current NTP/NNF guidance by a qualified clinician.",
        doseContext: { ageGroup, weightBand },
        evidence: [
          {
            chunkId: "tb-pre-xdr-regimen",
            note: "Fluoroquinolone resistance removes moxifloxacin from BPaLM, producing BPaL.",
            source: "provided-tb-knowledge-source",
          },
        ],
        regimen: {
          duration: "6-9 months",
          medicines: ["Bedaquiline", "Pretomanid", "Linezolid"],
          name: "BPaL",
          phaseSummary: [
            "Consider extension to 9 months if culture remains positive at month 4.",
          ],
        },
        routing,
        safetyFlags,
      };
    case "XDR_TB":
      return {
        classification: {
          confidenceScore: predictionData.confidence_score,
          primaryClass: "TB",
          subtype: predictionData.subtype_prediction,
          tbType,
        },
        disclaimer:
          "This decision support output is not a diagnosis or prescription and must be reviewed against current NTP/NNF guidance by a qualified clinician.",
        doseContext: { ageGroup, weightBand },
        evidence: [
          {
            chunkId: "tb-xdr-regimen",
            note: "XDR-TB requires expert individualized regimen design based on culture and DST.",
            source: "provided-tb-knowledge-source",
          },
        ],
        regimen: {
          duration: "18-24 months",
          medicines: ["Individualized effective Group B/C agents per DST"],
          name: "Expert individualized longer regimen",
        },
        routing: [
          ...routing,
          "Route to DR-TB expert consilium with culture and comprehensive DST before regimen finalization.",
        ],
        safetyFlags,
      };
    default:
      return {
        classification: {
          confidenceScore: predictionData.confidence_score,
          primaryClass: "TB",
          subtype: predictionData.subtype_prediction,
          tbType: "NOT_TB",
        },
        disclaimer:
          "This decision support output is not a diagnosis and must be reviewed by a qualified clinician.",
        doseContext: { ageGroup, weightBand },
        evidence: [],
        regimen: null,
        routing: ["Subtype is not applicable to TB routing."],
        safetyFlags,
      };
  }
}

export function normalizePrimaryClass(
  prediction: string
): PredictionData["primary_class"] {
  const normalized = prediction.trim().toUpperCase();

  if (normalized === "TUBERCULOSIS" || normalized === "TB") {
    return "TB";
  }
  if (normalized === "PNEUMONIA") {
    return "PNEUMONIA";
  }
  return "NORMAL";
}

export function normalizeSubtype({
  primaryClass,
  subtype,
}: {
  primaryClass: PredictionData["primary_class"];
  subtype?: string | null;
}): SubtypePrediction {
  if (primaryClass !== "TB") {
    return "NOT_APPLICABLE";
  }

  switch (subtype) {
    case "DRUG_SENSITIVE":
    case "DRUG_RESISTANT":
    case "MDR_TB":
    case "PRE_XDR_TB":
    case "XDR_TB":
      return subtype;
    default:
      return "DRUG_SENSITIVE";
  }
}
