import { tool } from "ai";
import { z } from "zod";

export const analyzeXray = tool({
  description:
    "Use this tool when the user wants to analyze a Chest X-Ray for diseases (Pneumonia, Tuberculosis, Normal). It will trigger a generative UI form asking the user for the Patient ID, Patient Name, and the X-Ray image.",
  execute: async () => ({
    message:
      "Please fill out the patient details and upload the X-Ray image below.",
    status: "waiting_for_upload",
  }),
  inputSchema: z.object({
    intent: z
      .string()
      .optional()
      .describe("Optional note about what the user wants to look for."),
  }),
});
