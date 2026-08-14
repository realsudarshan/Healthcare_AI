"use client";

import NextImage from "next/image";
import type React from "react";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface XrayFormProps {
  toolCallId: string;
}

type LabClassification = {
  classification: string;
  label: string;
  rationale: string;
  nextTests: string[];
  disclaimer: string;
};

type RagDiagnosis = {
  assessment?: {
    diagnosis_summary?: string;
    medicine_support?: {
      cautions?: string[];
      disclaimer?: string;
      recommendations?: {
        condition: string;
        medicine: string;
        source: string;
        suggestion: string;
      }[];
      title?: string;
    };
    most_likely?: string;
    tb_category?: string;
    supporting_evidence?: string[];
    missing_or_uncertain?: string[];
    safety_note?: string;
  };
  chunks?: {
    id: string;
    source: string;
    score: number;
    content: string;
    metadata?: Record<string, unknown>;
  }[];
  error?: string;
  in_scope?: boolean;
  message?: string;
  source_references?: { id: string; source: string; label: string }[];
};

const fieldClass =
  "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm";

const selectClass =
  "flex h-9 w-full rounded-md border border-input bg-background text-foreground px-3 py-1 text-sm shadow-sm";

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

export function XrayForm({ toolCallId }: XrayFormProps) {
  const [patientName, setPatientName] = useState("");
  const [patientId, setPatientId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [labConsent, setLabConsent] = useState<
    "pending" | "accepted" | "declined"
  >("pending");
  const [clinicalConsent, setClinicalConsent] = useState<
    "pending" | "accepted" | "declined"
  >("pending");
  const [isClassifying, setIsClassifying] = useState(false);
  const [labResult, setLabResult] = useState<LabClassification | null>(null);
  const [labValues, setLabValues] = useState<Record<string, string> | null>(
    null
  );
  const [isSavingContext, setIsSavingContext] = useState(false);
  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [contextSaved, setContextSaved] = useState(false);
  const [ragDiagnosis, setRagDiagnosis] = useState<RagDiagnosis | null>(null);
  const [narrative, setNarrative] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Analyses pixel data to determine whether an image looks like a chest X-ray.
   * Chest X-rays have three reliable properties measurable without any model:
   *   1. Near-grayscale — each pixel's R, G, B channels are almost equal.
   *      A colour photo has large channel spread; an X-ray does not.
   *   2. Dark overall — X-rays are mostly black with bright highlights.
   *      Mean luminance is typically below 160/255.
   *   3. Aspect ratio close to square — chest PA/AP views are 0.7–1.4 w/h.
   *
   * Returns a rejection reason string, or null if the image passes.
   */
  const checkIsXray = useCallback(
    (xrayFile: File): Promise<string | null> =>
      new Promise((resolve) => {
        const url = URL.createObjectURL(xrayFile);
        const img = new Image();
        img.onload = () => {
          URL.revokeObjectURL(url);

          // 1. Aspect ratio check
          const ratio = img.naturalWidth / img.naturalHeight;
          if (ratio < 0.6 || ratio > 1.8) {
            resolve(
              "Image proportions don't match a chest X-ray (expected roughly square)."
            );
            return;
          }

          // Sample a 64×64 thumbnail — fast and representative
          const SIZE = 64;
          const canvas = document.createElement("canvas");
          canvas.width = SIZE;
          canvas.height = SIZE;
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            resolve(null);
            return;
          }
          ctx.drawImage(img, 0, 0, SIZE, SIZE);
          const { data } = ctx.getImageData(0, 0, SIZE, SIZE); // RGBA flat array

          let totalLuminance = 0;
          let totalColorSpread = 0;
          const pixelCount = SIZE * SIZE;

          for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            // Perceived luminance (BT.601)
            totalLuminance += 0.299 * r + 0.587 * g + 0.114 * b;
            // Color spread: how different are the three channels?
            const mean = (r + g + b) / 3;
            totalColorSpread +=
              Math.abs(r - mean) + Math.abs(g - mean) + Math.abs(b - mean);
          }

          const avgLuminance = totalLuminance / pixelCount;
          // Average per-channel deviation from gray (0 = perfectly gray, 85 = max)
          const avgColorSpread = totalColorSpread / pixelCount / 3;

          // 2. Color check — X-rays are grayscale; spread > 15 indicates a colour photo
          if (avgColorSpread > 15) {
            resolve(
              "Image appears to be a colour photo. Please upload a chest X-ray radiograph."
            );
            return;
          }

          // 3. Brightness check — X-rays are predominantly dark (mean < 170)
          if (avgLuminance > 170) {
            resolve("Image appears too bright to be a chest X-ray radiograph.");
            return;
          }

          resolve(null); // passes all checks
        };
        img.onerror = () => {
          URL.revokeObjectURL(url);
          resolve("Could not read image data.");
        };
        img.src = url;
      }),
    []
  );

  const handleFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const selected = event.target.files?.[0] ?? null;
      if (!selected) {
        setFile(null);
        setPreviewUrl(null);
        return;
      }
      if (!ALLOWED_TYPES.includes(selected.type)) {
        toast.error("Only JPEG or PNG chest X-ray images are accepted.");
        event.target.value = "";
        return;
      }
      const rejection = await checkIsXray(selected);
      if (rejection) {
        toast.error(rejection);
        event.target.value = "";
        return;
      }
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
    },
    [checkIsXray]
  );

  const handleSubmit = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault();
      if (!patientName || !patientId || !file) {
        toast.error("Enter the patient details and select an image.");
        return;
      }
      setIsSubmitting(true);
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("patientName", patientName);
        formData.append("patientId", patientId);
        const response = await fetch("/api/predict-xray", {
          body: formData,
          method: "POST",
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        setResult(await response.json());
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Upload failed.");
      } finally {
        setIsSubmitting(false);
      }
    },
    [file, patientName, patientId]
  );

  const classifyLabs = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setIsClassifying(true);
      try {
        const formData = new FormData(event.currentTarget);
        const response = await fetch("/api/tb-classify", {
          body: JSON.stringify(Object.fromEntries(formData)),
          headers: { "Content-Type": "application/json" },
          method: "POST",
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const classification = await response.json();
        setLabValues(Object.fromEntries(formData) as Record<string, string>);
        setLabResult(classification);
      } catch (error) {
        toast.error(
          error instanceof Error
            ? error.message
            : "Could not classify the lab results."
        );
      } finally {
        setIsClassifying(false);
      }
    },
    []
  );

  const saveClinicalContext = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!labResult || !labValues || !result.recordId) {
        return;
      }
      const formData = new FormData(event.currentTarget);
      const comorbidities = [
        formData.get("hivStatus"),
        formData.get("diabetesStatus"),
      ];
      if (formData.get("renalImpairment") === "on") {
        comorbidities.push("RENAL_IMPAIRMENT");
      }
      if (formData.get("hepaticImpairment") === "on") {
        comorbidities.push("HEPATIC_IMPAIRMENT");
      }
      const patientMetadata = {
        age: Number(formData.get("age")),
        comorbidities,
        patient_type: formData.get("patientType"),
        pregnancy_status: formData.get("pregnancyStatus") === "on",
        weight_kg: Number(formData.get("weightKg")),
      };
      const clinicalNotes = String(formData.get("clinicalNotes") ?? "").trim();
      setIsSavingContext(true);
      setRagDiagnosis(null);
      setNarrative("");
      try {
        const response = await fetch("/api/tb-clinical-context", {
          body: JSON.stringify({
            patientMetadata,
            recordId: result.recordId,
            tbDecision: {
              classification: labResult.classification,
              label: labResult.label,
              labResults: labValues,
              rationale: labResult.rationale,
            },
          }),
          headers: { "Content-Type": "application/json" },
          method: "POST",
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        setContextSaved(true);
        toast.success("Patient clinical details saved.");
        setIsDiagnosing(true);
        try {
          const patientPayload = {
            patient: {
              comorbidities: {
                diabetes_status: formData.get("diabetesStatus"),
                hepatic_impairment: formData.get("hepaticImpairment") === "on",
                hiv_status: formData.get("hivStatus"),
                renal_impairment: formData.get("renalImpairment") === "on",
              },
              demographics: {
                age: patientMetadata.age,
                ntp_patient_type: patientMetadata.patient_type,
                patient_id: patientId,
                patient_name: patientName,
                pregnancy_status: patientMetadata.pregnancy_status,
                weight_kg: patientMetadata.weight_kg,
              },
              lab_tests: labValues,
              model_predictions: {
                confidence: result.confidence,
                prediction: result.prediction,
                probabilities: result.all_probs,
              },
              notes: clinicalNotes,
              tb_tests: {
                classification: labResult.classification,
                label: labResult.label,
                rationale: labResult.rationale,
                recommended_next_tests: labResult.nextTests,
              },
              xray_findings: {
                ai_finding: result.prediction,
                confidence: result.confidence,
              },
            },
            stream: true,
          };

          // Also fetch structured JSON for the detail cards (non-streaming)
          const structuredRes = await fetch("/api/rag-diagnose", {
            body: JSON.stringify({ ...patientPayload, stream: false }),
            headers: { "Content-Type": "application/json" },
            method: "POST",
          });
          if (structuredRes.ok) {
            const diagnosis = await structuredRes.json();
            setRagDiagnosis(diagnosis);
          }

          // Stream the LLM narrative word-by-word
          setIsStreaming(true);
          const streamRes = await fetch("/api/rag-diagnose", {
            body: JSON.stringify(patientPayload),
            headers: { "Content-Type": "application/json" },
            method: "POST",
          });
          if (!streamRes.ok || !streamRes.body) {
            throw new Error("Narrative stream failed.");
          }
          const reader = streamRes.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          let streamDone = false;
          while (!streamDone) {
            // biome-ignore lint/performance/noAwaitInLoops: stream must be read chunk-by-chunk sequentially
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";
            for (const line of lines) {
              if (!line.startsWith("data:")) {
                continue;
              }
              const raw = line.slice(5).trim();
              if (raw === "[DONE]") {
                streamDone = true;
                break;
              }
              try {
                const parsed = JSON.parse(raw) as {
                  token?: string;
                  error?: string;
                };
                if (parsed.error) {
                  throw new Error(parsed.error);
                }
                if (parsed.token) {
                  setNarrative((prev) => prev + parsed.token);
                }
              } catch (parseErr) {
                if (
                  parseErr instanceof Error &&
                  parseErr.message !== "Unexpected end of JSON input"
                ) {
                  throw parseErr;
                }
              }
            }
          }
          toast.success("Clinical narrative generated.");
        } catch (diagnosisError) {
          toast.error(
            diagnosisError instanceof Error
              ? diagnosisError.message
              : "Clinical details were saved, but narrative generation failed."
          );
        } finally {
          setIsStreaming(false);
        }
      } catch (error) {
        toast.error(
          error instanceof Error
            ? error.message
            : "Could not save patient details."
        );
      } finally {
        setIsSavingContext(false);
        setIsDiagnosing(false);
      }
    },
    [labResult, labValues, result, patientId, patientName]
  );

  const handleAcceptLab = useCallback(() => setLabConsent("accepted"), []);
  const handleDeclineLab = useCallback(() => setLabConsent("declined"), []);
  const handleAcceptClinical = useCallback(
    () => setClinicalConsent("accepted"),
    []
  );
  const handleDeclineClinical = useCallback(
    () => setClinicalConsent("declined"),
    []
  );
  const handlePatientNameChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setPatientName(e.target.value),
    []
  );
  const handlePatientIdChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setPatientId(e.target.value),
    []
  );

  if (result) {
    const prediction = String(result.prediction ?? "");
    const isNormal = prediction.toLowerCase() === "normal";
    const isTbFinding = ["tb", "tuberculosis"].includes(
      prediction.toLowerCase()
    );
    return (
      <div className="w-full space-y-4 rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-xl font-semibold">X-Ray Result</h3>
        <div className="grid gap-6 md:grid-cols-2">
          {previewUrl ? (
            <NextImage
              alt="Patient X-Ray"
              className="w-full rounded-lg border object-cover"
              height={400}
              src={previewUrl}
              unoptimized
              width={400}
            />
          ) : null}
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-2 border-b pb-4 text-sm">
              <span className="text-muted-foreground">Patient</span>
              <span className="font-medium">{patientName}</span>
              <span className="text-muted-foreground">Patient ID</span>
              <span className="font-medium">{patientId}</span>
            </div>
            <div
              className={cn(
                "rounded-lg border p-4",
                isNormal
                  ? "border-emerald-500/20 bg-emerald-500/10"
                  : "border-destructive/20 bg-destructive/10"
              )}
            >
              <div className="text-sm text-muted-foreground">
                AI X-Ray Finding
              </div>
              <div
                className={cn(
                  "mt-1 text-2xl font-bold",
                  isNormal ? "text-emerald-600" : "text-destructive"
                )}
              >
                {prediction}
              </div>
              <div className="mt-2 text-sm">
                {(Number(result.confidence) * 100).toFixed(1)}% confidence
              </div>
            </div>
            {isTbFinding ? (
              <TbLabConsent
                consent={labConsent}
                onAcceptLab={handleAcceptLab}
                onDeclineLab={handleDeclineLab}
              />
            ) : (
              <p className="text-sm text-muted-foreground">
                This X-ray finding does not confirm or rule out tuberculosis.
                Consider laboratory testing if clinical suspicion remains.
              </p>
            )}
          </div>
        </div>
        {labConsent === "accepted" ? (
          <TbLabForm
            isSubmitting={isClassifying}
            onSubmit={classifyLabs}
            result={labResult}
          />
        ) : null}
        {labResult && labValues ? (
          <ClinicalDetailsConsent
            consent={clinicalConsent}
            label={labResult.label}
            onAccept={handleAcceptClinical}
            onDecline={handleDeclineClinical}
          />
        ) : null}
        {clinicalConsent === "accepted" ? (
          <PatientMetadataForm
            isSaved={contextSaved && Boolean(ragDiagnosis)}
            isSaving={isSavingContext || isDiagnosing}
            onSubmit={saveClinicalContext}
          />
        ) : null}
        {isDiagnosing && !narrative ? (
          <p className="text-sm text-muted-foreground">
            Generating guideline-grounded RAG diagnosis...
          </p>
        ) : null}
        {ragDiagnosis ? <RagDiagnosisCard diagnosis={ragDiagnosis} /> : null}
        {narrative || isStreaming ? (
          <NarrativeCard isStreaming={isStreaming} narrative={narrative} />
        ) : null}
      </div>
    );
  }

  return (
    <form
      className="w-full space-y-5 rounded-lg border bg-card p-6 shadow-sm"
      onSubmit={handleSubmit}
    >
      <div>
        <h3 className="text-lg font-semibold">
          Upload Chest X-Ray for Analysis
        </h3>
        <p className="text-sm text-muted-foreground">
          The first step screens the radiograph only. Diagnostic tests are
          collected afterwards when needed.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <label className="space-y-2 text-sm font-medium">
          Patient Name
          <input
            className={fieldClass}
            disabled={isSubmitting}
            onChange={handlePatientNameChange}
            required
            value={patientName}
          />
        </label>
        <label className="space-y-2 text-sm font-medium">
          Patient ID
          <input
            className={fieldClass}
            disabled={isSubmitting}
            onChange={handlePatientIdChange}
            required
            value={patientId}
          />
        </label>
      </div>
      <label className="block space-y-2 text-sm font-medium">
        Chest X-Ray Image
        <input
          accept="image/jpeg,image/jpg,image/png"
          className={fieldClass}
          disabled={isSubmitting}
          id={`file-upload-${toolCallId}`}
          onChange={handleFileChange}
          ref={fileInputRef}
          required
          type="file"
        />
      </label>
      <button
        className="h-9 w-full rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground disabled:opacity-50"
        disabled={isSubmitting}
        type="submit"
      >
        {isSubmitting ? "Analyzing..." : "Analyze Radiograph"}
      </button>
    </form>
  );
}

function NarrativeCard({
  narrative,
  isStreaming,
}: {
  narrative: string;
  isStreaming: boolean;
}) {
  return (
    <div className="space-y-3 rounded-lg border bg-background p-4 text-sm">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold">AI Clinical Narrative</h4>
        {isStreaming ? (
          <span className="text-xs text-muted-foreground animate-pulse">
            Generating...
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            Guideline-grounded · Decision support only
          </span>
        )}
      </div>
      <div className="whitespace-pre-wrap leading-relaxed text-foreground/90">
        {narrative}
        {isStreaming ? (
          <span className="ml-0.5 inline-block w-0.5 h-4 bg-foreground/60 animate-pulse align-middle" />
        ) : null}
      </div>
    </div>
  );
}

function TbLabConsent({
  consent,
  onAcceptLab,
  onDeclineLab,
}: {
  consent: "pending" | "accepted" | "declined";
  onAcceptLab: () => void;
  onDeclineLab: () => void;
}) {
  if (consent === "accepted") {
    return (
      <p className="text-sm text-muted-foreground">
        Laboratory diagnostic follow-up has been requested below.
      </p>
    );
  }
  if (consent === "declined") {
    return (
      <p className="text-sm text-muted-foreground">
        Laboratory diagnostic follow-up was not requested.
      </p>
    );
  }
  return (
    <div className="space-y-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm">
      <p className="font-medium">
        There is a possibility of tuberculosis on this X-ray. Please perform
        laboratory tests to confirm it and assess resistance. Do you want to
        enter the lab results now?
      </p>
      <div className="flex gap-2">
        <button
          className="h-9 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground"
          onClick={onAcceptLab}
          type="button"
        >
          Yes
        </button>
        <button
          className="h-9 rounded-md border px-4 text-sm font-medium"
          onClick={onDeclineLab}
          type="button"
        >
          Not now
        </button>
      </div>
    </div>
  );
}

function ClinicalDetailsConsent({
  consent,
  label,
  onAccept,
  onDecline,
}: {
  consent: "pending" | "accepted" | "declined";
  label: string;
  onAccept: () => void;
  onDecline: () => void;
}) {
  if (consent === "accepted") {
    return (
      <p className="text-sm text-muted-foreground">
        Enter the patient clinical details below.
      </p>
    );
  }
  if (consent === "declined") {
    return (
      <p className="text-sm text-muted-foreground">
        Patient clinical details were not requested.
      </p>
    );
  }
  return (
    <div className="space-y-3 rounded-lg border bg-muted/30 p-4 text-sm">
      <p className="font-medium">
        The laboratory results indicate: {label}. Do you want to enter patient
        details for the next clinical assessment step?
      </p>
      <div className="flex gap-2">
        <button
          className="h-9 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground"
          onClick={onAccept}
          type="button"
        >
          Yes
        </button>
        <button
          className="h-9 rounded-md border px-4 text-sm font-medium"
          onClick={onDecline}
          type="button"
        >
          Not now
        </button>
      </div>
    </div>
  );
}

function PatientMetadataForm({
  isSaving,
  isSaved,
  onSubmit,
}: {
  isSaving: boolean;
  isSaved: boolean;
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <form
      className="space-y-4 rounded-lg border bg-muted/20 p-4"
      onSubmit={onSubmit}
    >
      <div>
        <h4 className="font-semibold">Doctor-entered patient details</h4>
        <p className="text-sm text-muted-foreground">
          Collect this after TB classification. These strict values are saved
          with the X-ray record for later clinical review.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <label className="space-y-2 text-sm font-medium">
          NTP patient type
          <select className={selectClass} defaultValue="NEW" name="patientType">
            <option value="NEW">New</option>
            <option value="RELAPSE">Relapse</option>
            <option value="TREATMENT_AFTER_FAILURE">
              Treatment after failure
            </option>
            <option value="LOSS_TO_FOLLOW_UP">Loss to follow-up</option>
            <option value="TRANSFER_IN">Transfer in</option>
          </select>
        </label>
        <label className="space-y-2 text-sm font-medium">
          Age
          <input
            className={fieldClass}
            max="120"
            min="1"
            name="age"
            required
            type="number"
          />
        </label>
        <label className="space-y-2 text-sm font-medium">
          Weight (kg)
          <input
            className={fieldClass}
            min="1"
            name="weightKg"
            required
            step="0.1"
            type="number"
          />
        </label>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-2 text-sm font-medium">
          HIV status
          <select
            className={selectClass}
            defaultValue="HIV_NEGATIVE"
            name="hivStatus"
          >
            <option value="HIV_NEGATIVE">HIV negative</option>
            <option value="HIV_POSITIVE">HIV positive</option>
          </select>
        </label>
        <label className="space-y-2 text-sm font-medium">
          Diabetes status
          <select
            className={selectClass}
            defaultValue="DIABETES_NEGATIVE"
            name="diabetesStatus"
          >
            <option value="DIABETES_NEGATIVE">Diabetes negative</option>
            <option value="DIABETES_POSITIVE">Diabetes positive</option>
          </select>
        </label>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        <label className="flex items-center gap-2 text-sm">
          <input name="pregnancyStatus" type="checkbox" />
          Pregnant
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input name="renalImpairment" type="checkbox" />
          Renal impairment
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input name="hepaticImpairment" type="checkbox" />
          Hepatic impairment
        </label>
      </div>
      <label className="block space-y-2 text-sm font-medium">
        Additional clinical notes
        <textarea
          className="min-h-24 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm"
          name="clinicalNotes"
          placeholder="Symptoms, duration, sputum result notes, CXR description, exposure history, oxygen status, or other relevant findings."
        />
      </label>
      <button
        className="h-9 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground disabled:opacity-50"
        disabled={isSaving || isSaved}
        type="submit"
      >
        {isSaved
          ? "RAG diagnosis generated"
          : isSaving
            ? "Saving and diagnosing..."
            : "Save details and diagnose with RAG"}
      </button>
    </form>
  );
}

function RagDiagnosisCard({ diagnosis }: { diagnosis: RagDiagnosis }) {
  if (diagnosis.in_scope === false) {
    return (
      <div className="rounded-lg border bg-muted/20 p-4 text-sm">
        {diagnosis.message ??
          "This query is outside the TB/pneumonia RAG scope."}
      </div>
    );
  }

  const { assessment } = diagnosis;
  const sources =
    diagnosis.source_references ??
    diagnosis.chunks?.map((chunk) => ({
      id: chunk.id,
      label: chunk.source,
      source: chunk.source,
    })) ??
    [];

  return (
    <div className="space-y-4 rounded-lg border bg-background p-4 text-sm">
      <div>
        <h4 className="font-semibold">Guideline-grounded RAG diagnosis</h4>
        <p className="text-muted-foreground">
          Uses collected TB details with NNF 2018 medicine references. Review
          against the patient and original reports.
        </p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-md border bg-muted/20 p-3">
          <div className="text-muted-foreground">Diagnosis summary</div>
          <div className="mt-1 font-medium">
            {assessment?.diagnosis_summary ??
              assessment?.most_likely ??
              "Insufficient evidence"}
          </div>
        </div>
        <div className="rounded-md border bg-muted/20 p-3">
          <div className="text-muted-foreground">TB category</div>
          <div className="mt-1 font-medium">
            {assessment?.tb_category ?? "Not determined"}
          </div>
        </div>
      </div>
      {assessment?.supporting_evidence?.length ? (
        <div>
          <div className="font-medium">Supporting evidence</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-muted-foreground">
            {assessment.supporting_evidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {assessment?.missing_or_uncertain?.length ? (
        <div>
          <div className="font-medium">Missing or uncertain tests</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-muted-foreground">
            {assessment.missing_or_uncertain.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {assessment?.medicine_support?.recommendations?.length ? (
        <div className="space-y-2">
          <div className="font-medium">
            {assessment.medicine_support.title ?? "Medicine suggestions"}
          </div>
          <div className="space-y-2">
            {assessment.medicine_support.recommendations.map((item) => (
              <div
                className="rounded-md border bg-muted/20 p-3"
                key={`${item.medicine}-${item.source}`}
              >
                <div className="font-medium">{item.medicine}</div>
                <p className="mt-1 text-muted-foreground">{item.suggestion}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {item.source}
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {assessment?.medicine_support?.cautions?.length ? (
        <div>
          <div className="font-medium">Medicine cautions</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-muted-foreground">
            {assessment.medicine_support.cautions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {sources.length ? (
        <div>
          <div className="font-medium">Retrieved guideline sources</div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-muted-foreground">
            {sources.slice(0, 4).map((source) => (
              <li key={source.id}>{source.label}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {assessment?.medicine_support?.disclaimer ? (
        <p className="text-xs text-muted-foreground">
          {assessment.medicine_support.disclaimer}
        </p>
      ) : null}
      {assessment?.safety_note ? (
        <p className="text-xs text-muted-foreground">
          {assessment.safety_note}
        </p>
      ) : null}
    </div>
  );
}

function TbLabForm({
  isSubmitting,
  onSubmit,
  result,
}: {
  isSubmitting: boolean;
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  result: LabClassification | null;
}) {
  return (
    <div className="space-y-4 rounded-lg border bg-muted/20 p-4">
      <div>
        <h4 className="font-semibold">TB diagnostic follow-up</h4>
        <p className="text-sm text-muted-foreground">
          Enter available laboratory results. Leave tests not yet performed as
          “Not done”.
        </p>
      </div>
      <form className="grid gap-4 md:grid-cols-2" onSubmit={onSubmit}>
        <label className="space-y-2 text-sm font-medium">
          GeneXpert MTB complex DNA
          <select
            className={selectClass}
            defaultValue="NOT_DONE"
            name="mtbDetected"
          >
            <option value="NOT_DONE">Not done</option>
            <option value="DETECTED">Detected</option>
            <option value="NOT_DETECTED">Not detected</option>
          </select>
        </label>
        <label className="space-y-2 text-sm font-medium">
          GeneXpert rpoB / rifampicin result
          <select
            className={selectClass}
            defaultValue="NOT_DONE"
            name="rifampicinResistance"
          >
            <option value="NOT_DONE">Not done</option>
            <option value="RESISTANT">Resistant</option>
            <option value="SUSCEPTIBLE">Susceptible</option>
            <option value="INDETERMINATE">Indeterminate</option>
          </select>
        </label>
        <label className="space-y-2 text-sm font-medium">
          First-line LPA / isoniazid result
          <select
            className={selectClass}
            defaultValue="NOT_DONE"
            name="isoniazidResistance"
          >
            <option value="NOT_DONE">Not done</option>
            <option value="RESISTANT">Resistance detected</option>
            <option value="SUSCEPTIBLE">Susceptible</option>
            <option value="INDETERMINATE">Indeterminate</option>
          </select>
        </label>
        <label className="space-y-2 text-sm font-medium">
          Second-line LPA (gyrA/gyrB)
          <select
            className={selectClass}
            defaultValue="NOT_DONE"
            name="fluoroquinoloneResistance"
          >
            <option value="NOT_DONE">Not done</option>
            <option value="RESISTANT">Resistance detected</option>
            <option value="SUSCEPTIBLE">No resistance detected</option>
            <option value="INDETERMINATE">Indeterminate</option>
          </select>
        </label>
        <label className="space-y-2 text-sm font-medium">
          Culture & DST Group A result
          <select
            className={selectClass}
            defaultValue="NOT_DONE"
            name="groupAResistance"
          >
            <option value="NOT_DONE">Not done</option>
            <option value="RESISTANT">Resistance detected</option>
            <option value="SUSCEPTIBLE">No resistance detected</option>
            <option value="INDETERMINATE">Indeterminate</option>
          </select>
        </label>
        <button
          className="h-9 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground disabled:opacity-50 md:col-span-2"
          disabled={isSubmitting}
          type="submit"
        >
          {isSubmitting ? "Classifying..." : "Classify TB from lab results"}
        </button>
      </form>
      {result ? (
        <div className="space-y-2 rounded-md border bg-background p-4 text-sm">
          <div className="font-semibold">{result.label}</div>
          <p>{result.rationale}</p>
          <div>
            <div className="font-medium">Recommended next diagnostic tests</div>
            <ul className="mt-1 list-disc pl-5 text-muted-foreground">
              {result.nextTests.map((test) => (
                <li key={test}>{test}</li>
              ))}
            </ul>
          </div>
          <p className="text-xs text-muted-foreground">{result.disclaimer}</p>
        </div>
      ) : null}
    </div>
  );
}
