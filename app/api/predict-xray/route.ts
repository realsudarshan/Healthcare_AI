import { type NextRequest, NextResponse } from "next/server";
import { auth } from "@/app/(auth)/auth";
import { savePatientXrayRecord } from "@/lib/db/queries";
import {
  getPublicBaseUrl,
  isStorageConfigured,
  putObject,
} from "@/lib/storage";
import { generateUUID } from "@/lib/utils";

function getBackendBaseUrl() {
  return (process.env.LANGCHAIN_API_URL ?? "http://127.0.0.1:8001")
    .replace("localhost:8000", "localhost:8001")
    .replace("127.0.0.1:8000", "127.0.0.1:8001")
    .replace(/\/$/, "");
}

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    const formData = await req.formData();
    const file = formData.get("file") as File;
    const patientName = formData.get("patientName") as string;
    const patientId = formData.get("patientId") as string;
    if (!file || !patientName || !patientId) {
      return NextResponse.json(
        { error: "Missing required fields." },
        { status: 400 }
      );
    }
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        { error: "Only JPEG or PNG chest X-ray images are accepted." },
        { status: 400 }
      );
    }
    const backendFormData = new FormData();
    backendFormData.append("file", file);
    const baseURL = getBackendBaseUrl();
    const response = await fetch(`${baseURL}/predict`, {
      body: backendFormData,
      method: "POST",
    });
    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend error: ${await response.text()}` },
        { status: 500 }
      );
    }
    const result = await response.json();
    if (result.error) {
      return NextResponse.json(result, { status: 500 });
    }

    // Upload the X-ray image to object storage (Backblaze B2) and persist only
    // the URL.  Fall back to null if storage is not configured so the rest of
    // the flow still works.
    let imageUrl: string | null = null;
    if (isStorageConfigured()) {
      const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
      const key = `xrays/${generateUUID()}/${safeName}`;
      const fileBuffer = new Uint8Array(await file.arrayBuffer());
      await putObject(key, fileBuffer, file.type || "image/jpeg");
      const publicBaseUrl = getPublicBaseUrl();
      const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
      imageUrl = publicBaseUrl
        ? `${publicBaseUrl}/${key}`
        : `${basePath}/api/files/${key}`;
    }

    const savedRecord = await savePatientXrayRecord({
      allProbs: result.all_probs,
      confidence: String(result.confidence),
      imageUrl,
      patientId,
      patientName,
      prediction: result.prediction,
      userId: session.user.id,
    });
    return NextResponse.json({ ...result, recordId: savedRecord.id });
  } catch (error) {
    console.error("Error in predict-xray route:", error);
    return NextResponse.json(
      { error: "Internal server error connecting to AI backend." },
      { status: 500 }
    );
  }
}
