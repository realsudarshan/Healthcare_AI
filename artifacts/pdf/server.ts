import { smoothStream, streamText } from "ai";
import { updateDocumentPrompt } from "@/lib/ai/prompts";
import { getLanguageModel } from "@/lib/ai/providers";
import { createDocumentHandler } from "@/lib/artifacts/server";

const PDF_SYSTEM_PROMPT = `You are a clinical report generator for a TB/Pneumonia healthcare AI system.
The user prompt contains ALL patient data as a JSON block. Extract every field and populate the report fully.
If a field is present in the data, it MUST appear in the report — never write "NOT PROVIDED" for a field that exists in the data.
Only write "Not recorded" for fields genuinely absent from the data.

Generate a complete, well-structured patient clinical report as clean HTML.

STRICT RULES:
- Output ONLY valid HTML starting with <div class="report"> — no markdown, no code fences, no preamble.
- Use only inline styles and these safe HTML elements: div, h1, h2, h3, p, table, thead, tbody, tr, th, td, ul, li, span, hr, strong, em.
- ALL inline styles must be on the element itself (style="...").
- Structure the report with ALL of these sections:

  1. REPORT HEADER
     - Centered title "Clinical Assessment Report" in large bold text
     - Patient Name, Patient ID, Date of Report, Record ID
     - Red bold banner: "⚠ DECISION SUPPORT ONLY — NOT A FINAL DIAGNOSIS. Confirm with a qualified clinician."

  2. X-RAY FINDINGS
     - AI Prediction (e.g. Tuberculosis / Pneumonia / Normal)
     - Confidence Score (as percentage)
     - All class probabilities as a small table (Normal %, Pneumonia %, TB %)

  3. TB CLASSIFICATION
     - TB Category (Drug-Sensitive / MDR/RR-TB / Pre-XDR / XDR / Not confirmed)
     - Lab classification label and rationale from tbDecision

  4. LABORATORY RESULTS — show as a 3-column table (Test | Result | Interpretation)
     Rows: GeneXpert MTB, Rifampicin resistance, Isoniazid LPA, Fluoroquinolone LPA, Culture/DST Group A
     Use the actual values from tbDecision.labResults if present.

  5. DRUG RECOMMENDATIONS
     - List medicines with doses cited from NNF 2018 based on the TB category
     - Drug-Sensitive TB: Isoniazid 300mg/day, Rifampicin 600mg/day (>50kg), Pyrazinamide, Ethambutol
     - MDR/RR-TB: Bedaquiline + second-line regimen per NNF 2018
     - Include source citation [NNF2018#page=XX]

  6. PATIENT CLINICAL DETAILS
     - Age, Weight, NTP Patient Type
     - HIV status, Diabetes status
     - Pregnancy, Renal impairment, Hepatic impairment flags

  7. COMORBIDITY CAUTIONS
     - List specific cautions based on the patient's comorbidities

  8. MISSING / PENDING TESTS
     - List any diagnostic tests not yet completed

  9. SAFETY NOTE
     - Bold paragraph: "This report is AI-generated decision support only. It must be reviewed and confirmed by a qualified clinician before any treatment decision. Drug doses and regimens must be verified against current local guidelines."

Styling rules:
- White background, black text, max-width 800px, font: system-ui sans-serif, font-size 13px, line-height 1.6
- Section headings: font-size 14px, font-weight 600, border-bottom: 2px solid #333, padding-bottom 4px, margin-top 20px
- Header title: text-align center, font-size 20px, font-weight 700, border-bottom 2px solid #333
- Tables: border-collapse collapse, width 100%, margin 8px 0
- th: background #f0f4f8, padding 7px 10px, border 1px solid #ccc, font-weight 600, text-align left
- td: padding 6px 10px, border 1px solid #ccc
- Red banner: color #c0392b, font-weight 700, font-size 13px, background #fff3f3, padding 8px, border-left 4px solid #c0392b
- Confidence: show as bold colored text — green if >80%, amber if 60-80%, red if <60%`;

export const pdfDocumentHandler = createDocumentHandler<"pdf">({
  kind: "pdf",
  onCreateDocument: async ({ title, dataStream, modelId }) => {
    let draftContent = "";

    const { stream } = streamText({
      experimental_transform: smoothStream({ chunking: "word" }),
      instructions: PDF_SYSTEM_PROMPT,
      model: getLanguageModel(modelId),
      prompt: title,
    });

    for await (const delta of stream) {
      if (delta.type === "text-delta") {
        draftContent += delta.text;
        dataStream.write({
          data: draftContent,
          transient: true,
          type: "data-pdfDelta",
        });
      }
    }

    // Strip any accidental markdown fences
    draftContent = draftContent
      .replace(/^```html?\n?/i, "")
      .replace(/\n?```\s*$/i, "")
      .trim();

    return draftContent;
  },
  onUpdateDocument: async ({ document, description, dataStream, modelId }) => {
    let draftContent = "";

    const { stream } = streamText({
      experimental_transform: smoothStream({ chunking: "word" }),
      instructions: `${updateDocumentPrompt(document.content, "text")}\n\n${PDF_SYSTEM_PROMPT}\n\nUpdate the existing HTML report based on the instruction. Output ONLY the updated HTML starting with <div class="report">.`,
      model: getLanguageModel(modelId),
      prompt: description,
    });

    for await (const delta of stream) {
      if (delta.type === "text-delta") {
        draftContent += delta.text;
        dataStream.write({
          data: draftContent,
          transient: true,
          type: "data-pdfDelta",
        });
      }
    }

    draftContent = draftContent
      .replace(/^```html?\n?/i, "")
      .replace(/\n?```\s*$/i, "")
      .trim();

    return draftContent;
  },
});
