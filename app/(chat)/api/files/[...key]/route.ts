import { NextResponse } from "next/server";

import { auth } from "@/app/(auth)/auth";
import { getObject, isStorageConfigured } from "@/lib/storage";

/**
 * Streams an uploaded object back through the app so the bucket can stay
 * private. Used when S3_PUBLIC_URL is not set.
 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ key: string[] }> }
) {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  if (!isStorageConfigured()) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const { key: segments } = await params;
  const key = segments.join("/");

  if (!key.startsWith("uploads/") && !key.startsWith("xrays/")) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  try {
    const object = await getObject(key);

    if (!object) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    return new Response(object.body, {
      headers: {
        "Cache-Control": "private, max-age=31536000, immutable",
        "Content-Type": object.contentType,
      },
    });
  } catch {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
}
