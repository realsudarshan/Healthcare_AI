export type LabResult =
  | "NOT_DONE"
  | "DETECTED"
  | "NOT_DETECTED"
  | "RESISTANT"
  | "SUSCEPTIBLE"
  | "INDETERMINATE";

export type TbLabInput = {
  mtbDetected: LabResult;
  rifampicinResistance: LabResult;
  isoniazidResistance: LabResult;
  fluoroquinoloneResistance: LabResult;
  groupAResistance: LabResult;
};

export function classifyTbFromLabs(input: TbLabInput) {
  const disclaimer =
    "Decision support only. Confirm all results with the laboratory and a qualified clinician; this does not provide treatment or medication advice.";
  if (input.mtbDetected !== "DETECTED") {
    return {
      classification: "UNCONFIRMED",
      disclaimer,
      label: "TB not confirmed by GeneXpert",
      nextTests: [
        "GeneXpert MTB/RIF Assay",
        "AFB smear microscopy",
        "Culture if clinical suspicion persists",
      ],
      rationale:
        input.mtbDetected === "NOT_DETECTED"
          ? "MTB complex DNA was not detected. Consider an alternative diagnosis or repeat testing/culture if clinical suspicion remains."
          : "GeneXpert MTB/RIF is needed first to confirm MTB complex DNA.",
    };
  }
  if (input.rifampicinResistance === "SUSCEPTIBLE") {
    if (input.isoniazidResistance === "RESISTANT") {
      return {
        classification: "HR_TB",
        disclaimer,
        label: "Isoniazid-resistant, rifampicin-susceptible TB",
        nextTests: [
          "Phenotypic first-line Culture & DST (H, R, Z, E)",
          "Clinician review using the applicable Hr-TB guideline",
        ],
        rationale:
          "MTB is detected and rifampicin is susceptible, but first-line LPA or DST indicates isoniazid resistance. This is not DS-TB.",
      };
    }
    if (input.isoniazidResistance !== "SUSCEPTIBLE") {
      return {
        classification: "RIFAMPICIN_SUSCEPTIBLE_PENDING_DST",
        disclaimer,
        label: "Rifampicin-susceptible TB; full first-line DST pending",
        nextTests: [
          "First-line LPA (MTBDRplus) for katG/inhA",
          "Phenotypic first-line Culture & DST (H, R, Z, E)",
        ],
        rationale:
          "Rifampicin susceptibility alone does not confirm DS-TB. Isoniazid susceptibility must be confirmed using first-line LPA (katG/inhA) or phenotypic DST.",
      };
    }
    return {
      classification: "DS_TB",
      disclaimer,
      label: "Drug-Sensitive Tuberculosis",
      nextTests: [
        "AFB smear microscopy",
        "Culture or full first-line DST when indicated",
      ],
      rationale:
        "MTB is detected and both rifampicin and isoniazid susceptibility are confirmed. Full first-line DST can further document susceptibility to pyrazinamide and ethambutol when indicated.",
    };
  }
  if (input.rifampicinResistance !== "RESISTANT") {
    return {
      classification: "TB_CONFIRMED_RESISTANCE_PENDING",
      disclaimer,
      label: "TB confirmed; rifampicin result pending",
      nextTests: ["Repeat GeneXpert rpoB result", "First-line LPA (MTBDRplus)"],
      rationale:
        "MTB is detected but rifampicin resistance is not available or is indeterminate.",
    };
  }
  if (input.fluoroquinoloneResistance === "SUSCEPTIBLE") {
    return {
      classification: "MDR_RR_TB",
      disclaimer,
      label: "MDR/RR-TB",
      nextTests: ["First-line LPA (MTBDRplus)", "Culture and DST as indicated"],
      rationale:
        "Rifampicin resistance is detected and no fluoroquinolone resistance is detected.",
    };
  }
  if (input.fluoroquinoloneResistance !== "RESISTANT") {
    return {
      classification: "MDR_RR_TB_FURTHER_TESTING",
      disclaimer,
      label: "MDR/RR-TB; escalation testing pending",
      nextTests: [
        "Second-line LPA (MTBDRsl)",
        "Culture & Drug Susceptibility Testing",
      ],
      rationale:
        "Rifampicin resistance is detected. Second-line LPA is required to assess fluoroquinolone resistance.",
    };
  }
  if (input.groupAResistance === "RESISTANT") {
    return {
      classification: "XDR_TB",
      disclaimer,
      label: "Extensively Drug-Resistant TB",
      nextTests: [
        "Comprehensive phenotypic Culture & DST",
        "Specialist laboratory and clinician review",
      ],
      rationale:
        "Rifampicin resistance, fluoroquinolone resistance, and Group A drug resistance are detected.",
    };
  }
  if (input.groupAResistance === "SUSCEPTIBLE") {
    return {
      classification: "PRE_XDR_TB",
      disclaimer,
      label: "Pre-Extensively Drug-Resistant TB",
      nextTests: ["Comprehensive phenotypic Culture & DST"],
      rationale:
        "Rifampicin and fluoroquinolone resistance are detected, with no Group A drug resistance detected.",
    };
  }
  return {
    classification: "PRE_XDR_OR_XDR_PENDING",
    disclaimer,
    label: "Pre-XDR-TB; XDR assessment pending",
    nextTests: ["Comprehensive phenotypic Culture & DST"],
    rationale:
      "Rifampicin and fluoroquinolone resistance are detected. Group A drug susceptibility is needed to distinguish Pre-XDR-TB from XDR-TB.",
  };
}
