import { redirect } from "next/navigation";
import { auth } from "@/app/(auth)/auth";
import { PatientRecordsTable } from "@/components/chat/patient-records-table";
import { getPatientXrayRecords } from "@/lib/db/queries";

export default async function PatientsPage() {
  const session = await auth();

  if (!session?.user?.id) {
    redirect("/");
  }

  const records = await getPatientXrayRecords({ userId: session.user.id });

  return (
    <div className="flex h-full w-full flex-col overflow-y-auto p-4 md:p-8">
      <div className="mx-auto w-full max-w-6xl">
        <h1 className="mb-8 text-3xl font-bold tracking-tight">
          Patient Diagnostic Records
        </h1>

        {records.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center rounded-lg border border-dashed text-muted-foreground">
            <p>No patient records found.</p>
            <p className="text-sm mt-1">
              Use the AI chatbot to analyze an X-Ray first!
            </p>
          </div>
        ) : (
          <PatientRecordsTable records={records} />
        )}
      </div>
    </div>
  );
}
