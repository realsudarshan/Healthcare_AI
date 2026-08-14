import { tool } from "ai";
import type { Session } from "next-auth";
import { z } from "zod";
import { getPatientXrayRecordsByPatientId } from "@/lib/db/queries";

function summarizeRecord(
  record: Awaited<ReturnType<typeof getPatientXrayRecordsByPatientId>>[number]
) {
  return {
    allProbabilities: record.allProbs,
    confidence: Number.parseFloat(record.confidence),
    createdAt: record.createdAt.toISOString(),
    diagnosis: record.prediction,
    patientId: record.patientId,
    patientMetadata: record.patientMetadata,
    patientName: record.patientName,
    recordId: record.id,
    tbDecision: record.tbDecision,
  };
}

export const getPatientRecords = ({ session }: { session: Session }) =>
  tool({
    description:
      "Look up the logged-in user's saved X-ray diagnostic records for a patient ID. Use when the user asks to show patient data, history, lab details, TB details, diagnosis, confidence, or clinical metadata for a specific patient ID.",
    execute: async ({ patientId }) => {
      if (!session.user?.id) {
        return { error: "Unauthorized." };
      }

      const records = await getPatientXrayRecordsByPatientId({
        patientId: patientId.trim(),
        userId: session.user.id,
      });

      return {
        count: records.length,
        patientId: patientId.trim(),
        records: records.map(summarizeRecord),
        reminder:
          "Image data is not included in chat responses. Open Patient Records and click View to inspect the X-ray image.",
        status: records.length > 0 ? "found" : "not_found",
      };
    },
    inputSchema: z.object({
      patientId: z
        .string()
        .min(1)
        .max(128)
        .describe("Exact patient ID to look up, for example 49."),
    }),
  });
