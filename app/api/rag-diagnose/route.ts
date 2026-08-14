import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/app/(auth)/auth";

const patientValue = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.null(),
  z.array(z.unknown()),
  z.record(z.string(), z.unknown()),
]);

const schema = z.object({
  patient: z.object({
    comorbidities: z.record(z.string(), patientValue).optional(),
    demographics: z.record(z.string(), patientValue).optional(),
    duration: z.record(z.string(), patientValue).optional(),
    lab_tests: z.record(z.string(), patientValue).optional(),
    microbiology: z.record(z.string(), patientValue).optional(),
    model_predictions: z.record(z.string(), patientValue).optional(),
    notes: z.string().optional(),
    previous_tb_history: z.record(z.string(), patientValue).optional(),
    risk_factors: z.record(z.string(), patientValue).optional(),
    symptoms: z.record(z.string(), patientValue).optional(),
    tb_tests: z.record(z.string(), patientValue).optional(),
    treatment_history: z.record(z.string(), patientValue).optional(),
    xray_findings: z.record(z.string(), patientValue).optional(),
  }),
  query: z.string().min(1).optional(),
  stream: z.boolean().optional(),
  top_k: z.number().int().min(1).max(12).optional(),
});

function getBackendBaseUrl() {
  return (process.env.LANGCHAIN_API_URL ?? "http://127.0.0.1:8001")
    .replace("localhost:8000", "localhost:8001")
    .replace("127.0.0.1:8000", "127.0.0.1:8001")
    .replace(/\/$/, "");
}

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const payload = schema.parse(await request.json());
    const baseURL = getBackendBaseUrl();
    const headers: HeadersInit = { "Content-Type": "application/json" };
    if (process.env.LANGCHAIN_API_KEY) {
      headers.Authorization = `Bearer ${process.env.LANGCHAIN_API_KEY}`;
    }

    const body = JSON.stringify({
      patient: payload.patient,
      query:
        payload.query ??
        "Use the collected TB type, lab results, X-ray result, and patient clinical details to provide TB diagnosis and medicine-support information from NNF 2018. Identify missing confirmation tests and safety cautions.",
      top_k: payload.top_k ?? 6,
    });

    // Streaming path — proxy the SSE stream directly to the client
    if (payload.stream) {
      const response = await fetch(`${baseURL}/rag/diagnose/stream`, {
        body,
        headers,
        method: "POST",
      });

      if (!response.ok || !response.body) {
        return NextResponse.json(
          { error: `RAG backend error: ${await response.text()}` },
          { status: 502 }
        );
      }

      return new Response(response.body, {
        headers: {
          "Cache-Control": "no-cache",
          "Content-Type": "text/event-stream",
          "X-Accel-Buffering": "no",
        },
      });
    }

    // Non-streaming path — return structured JSON as before
    const response = await fetch(`${baseURL}/rag/diagnose`, {
      body,
      headers,
      method: "POST",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `RAG backend error: ${await response.text()}` },
        { status: 502 }
      );
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid RAG diagnosis payload." },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: "Could not generate RAG diagnosis." },
      { status: 500 }
    );
  }
}
