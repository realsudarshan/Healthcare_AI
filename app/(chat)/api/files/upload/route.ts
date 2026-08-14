import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@/app/(auth)/auth";
import {
  getPublicBaseUrl,
  isStorageConfigured,
  putObject,
} from "@/lib/storage";
import { generateUUID } from "@/lib/utils";

const FileSchema = z.object({
  file: z
    .any()
    .refine((file) => file && typeof file === "object" && "size" in file, {
      message: "Expected a file",
    })
    .refine((file) => file.size <= 5 * 1024 * 1024, {
      message: "File size should be less than 5MB",
    })
    .refine((file) => ["image/jpeg", "image/png"].includes(file.type), {
      message: "File type should be JPEG or PNG",
    }),
});

export async function POST(request: Request) {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  if (!isStorageConfigured()) {
    return NextResponse.json(
      {
        error:
          "File uploads are not configured on this server. Set the S3_* environment variables to enable them.",
      },
      { status: 501 }
    );
  }

  if (request.body === null) {
    return new Response("Request body is empty", { status: 400 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as Blob;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    const validatedFile = FileSchema.safeParse({ file });

    if (!validatedFile.success) {
      const errorMessage = validatedFile.error.issues
        .map((error) => error.message)
        .join(", ");

      return NextResponse.json({ error: errorMessage }, { status: 400 });
    }

    const filename = (formData.get("file") as File).name;
    const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
    const contentType = file.type || "application/octet-stream";
    // Prefix with a UUID so concurrent uploads of the same filename never
    // overwrite each other.
    const key = `uploads/${generateUUID()}/${safeName}`;
    const fileBuffer = new Uint8Array(await file.arrayBuffer());

    try {
      await putObject(key, fileBuffer, contentType);

      const publicBaseUrl = getPublicBaseUrl();
      const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

      return NextResponse.json({
        contentType,
        pathname: safeName,
        url: publicBaseUrl
          ? `${publicBaseUrl}/${key}`
          : `${basePath}/api/files/${key}`,
      });
    } catch (error) {
      console.error("Failed to store upload:", error);
      return NextResponse.json({ error: "Upload failed" }, { status: 500 });
    }
  } catch {
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 }
    );
  }
}
