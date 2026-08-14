import { NextResponse } from "next/server";
import { auth } from "@/app/(auth)/auth";
import { getPatientXrayRecordImage } from "@/lib/db/queries";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ recordId: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { recordId } = await params;
  const record = await getPatientXrayRecordImage({
    recordId,
    userId: session.user.id,
  });

  if (!record) {
    return NextResponse.json({ error: "Record not found." }, { status: 404 });
  }

  return NextResponse.json({ imageUrl: record.imageUrl });
}
