"use client";

import { format } from "date-fns";
import { useCallback, useState } from "react";
import type { PatientXrayRecord } from "@/lib/db/schema";
import { cn } from "@/lib/utils";

type PatientXrayRecordListItem = Omit<PatientXrayRecord, "imageUrl">;

export function PatientRecordsTable({
  records,
}: {
  records: PatientXrayRecordListItem[];
}) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loadingImageId, setLoadingImageId] = useState<string | null>(null);

  const openPreview = useCallback(async (recordId: string) => {
    setLoadingImageId(recordId);
    try {
      const response = await fetch(
        `/api/patient-xray-records/${recordId}/image`
      );
      if (!response.ok) {
        return;
      }
      const data = (await response.json()) as { imageUrl?: string | null };
      if (data.imageUrl) {
        setSelectedImage(data.imageUrl);
      }
    } finally {
      setLoadingImageId(null);
    }
  }, []);

  const closePreview = useCallback(() => setSelectedImage(null), []);

  return (
    <>
      <div className="overflow-x-auto rounded-lg border bg-card text-card-foreground shadow-sm">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="border-b bg-muted/50">
            <tr className="text-left font-medium text-muted-foreground">
              <th className="px-6 py-4">X-Ray</th>
              <th className="px-6 py-4">Date</th>
              <th className="px-6 py-4">Patient Name</th>
              <th className="px-6 py-4">Patient ID</th>
              <th className="px-6 py-4">Diagnosis</th>
              <th className="px-6 py-4 text-right">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {records.map((record) => (
              <PatientRecordRow
                isLoadingImage={loadingImageId === record.id}
                key={record.id}
                onOpenPreview={openPreview}
                record={record}
              />
            ))}
          </tbody>
        </table>
      </div>

      {selectedImage ? (
        <button
          aria-label="Close X-Ray preview"
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/80 p-6 outline-none"
          onClick={closePreview}
          type="button"
        >
          {/* biome-ignore lint/performance/noImgElement: base64 data URL from DB */}
          <img
            alt="Full-size X-Ray"
            className="max-h-full max-w-full rounded-lg object-contain shadow-2xl"
            src={selectedImage}
          />
        </button>
      ) : null}
    </>
  );
}

function PatientRecordRow({
  isLoadingImage,
  onOpenPreview,
  record,
}: {
  isLoadingImage: boolean;
  onOpenPreview: (recordId: string) => void;
  record: PatientXrayRecordListItem;
}) {
  const handleOpenPreview = useCallback(() => {
    onOpenPreview(record.id);
  }, [onOpenPreview, record.id]);

  return (
    <tr className="transition-colors hover:bg-muted/50">
      <td className="px-6 py-4">
        <button
          className="h-8 rounded-md border px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
          disabled={isLoadingImage}
          onClick={handleOpenPreview}
          type="button"
        >
          {isLoadingImage ? "Loading..." : "View"}
        </button>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-muted-foreground">
        {format(new Date(record.createdAt), "MMM d, yyyy h:mm a")}
      </td>
      <td className="px-6 py-4 font-medium">{record.patientName}</td>
      <td className="px-6 py-4 text-muted-foreground">{record.patientId}</td>
      <td className="px-6 py-4">
        <span
          className={cn(
            "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
            record.prediction === "Normal"
              ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400"
              : "bg-destructive/10 text-destructive dark:bg-destructive/20"
          )}
        >
          {record.prediction}
        </span>
      </td>
      <td className="px-6 py-4 text-right tabular-nums font-medium">
        {(Number.parseFloat(record.confidence) * 100).toFixed(1)}%
      </td>
    </tr>
  );
}
