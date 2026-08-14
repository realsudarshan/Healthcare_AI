import { NextResponse } from "next/server";
import { z } from "zod";
import { classifyTbFromLabs } from "@/lib/tb/diagnostic-classifier";

const labValue = z.enum([
  "NOT_DONE",
  "DETECTED",
  "NOT_DETECTED",
  "RESISTANT",
  "SUSCEPTIBLE",
  "INDETERMINATE",
]);
const schema = z.object({
  fluoroquinoloneResistance: labValue,
  groupAResistance: labValue,
  isoniazidResistance: labValue,
  mtbDetected: labValue,
  rifampicinResistance: labValue,
});

export async function POST(request: Request) {
  try {
    return NextResponse.json(
      classifyTbFromLabs(schema.parse(await request.json()))
    );
  } catch {
    return NextResponse.json(
      { error: "Invalid laboratory results." },
      { status: 400 }
    );
  }
}
