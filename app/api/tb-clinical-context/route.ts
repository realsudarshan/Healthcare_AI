import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/app/(auth)/auth";
import { savePatientXrayClinicalContext } from "@/lib/db/queries";
import { comorbidityValues, patientTypeValues } from "@/lib/tb/knowledge";

const schema = z.object({
  patientMetadata: z.object({
    age: z.coerce.number().int().min(1).max(120),
    comorbidities: z.array(z.enum(comorbidityValues)),
    patient_type: z.enum(patientTypeValues),
    pregnancy_status: z.boolean(),
    weight_kg: z.coerce.number().positive().max(300),
  }),
  recordId: z.string().uuid(),
  tbDecision: z.object({
    classification: z.string(),
    label: z.string(),
    labResults: z.record(z.string(), z.string()),
    rationale: z.string(),
  }),
});

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  try {
    const payload = schema.parse(await request.json());
    const requiresDrTbReferral = [
      "RELAPSE",
      "TREATMENT_AFTER_FAILURE",
    ].includes(payload.patientMetadata.patient_type);
    const record = await savePatientXrayClinicalContext({
      ...payload,
      tbDecision: {
        ...payload.tbDecision,
        clinicalFlags: requiresDrTbReferral
          ? [
              "Nepal NTP: relapse or treatment-after-failure requires GeneXpert DR-TB referral; do not use the removed Category 2 retreatment regimen.",
            ]
          : [],
      },
      userId: session.user.id,
    });
    if (!record) {
      return NextResponse.json(
        { error: "X-ray record not found." },
        { status: 404 }
      );
    }
    return NextResponse.json({ saved: true });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid patient metadata." },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: "Could not save clinical context." },
      { status: 500 }
    );
  }
}
