"use client";

import { useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import { Artifact } from "@/components/chat/create-artifact";
import { DownloadIcon, RedoIcon, UndoIcon } from "@/components/chat/icons";

type Metadata = Record<string, never>;

function openPrintWindow(content: string, title: string) {
  const printWindow = window.open("", "_blank");
  if (!printWindow) {
    toast.error(
      "Pop-up blocked — allow pop-ups for this site to download the PDF."
    );
    return;
  }
  printWindow.document.write(`<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${title}</title>
  <style>
    body { font-family: -apple-system, "Segoe UI", system-ui, sans-serif; margin: 0; padding: 2rem; color: #1a1a1a; background: #fff; }
    .report { max-width: 800px; margin: 0 auto; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; }
    th, td { border: 1px solid #ccc; padding: 7px 10px; font-size: 13px; text-align: left; }
    th { background: #f0f4f8; font-weight: 600; }
    h1 { font-size: 20px; font-weight: 700; text-align: center; border-bottom: 2px solid #333; padding-bottom: 6px; margin-bottom: 4px; }
    h2 { font-size: 14px; font-weight: 600; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 20px; }
    p, li { font-size: 13px; line-height: 1.6; margin: 4px 0; }
    ul { padding-left: 1.2rem; }
    hr { border: none; border-top: 1px solid #eee; margin: 1rem 0; }
    @media print { body { padding: 0.5cm; } @page { margin: 1.5cm; } }
  </style>
</head>
<body>${content}</body>
</html>`);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => {
    printWindow.print();
  }, 400);
  toast.success("Print dialog opened — choose 'Save as PDF'.");
}

function PdfPreview({
  content,
  status,
  title,
}: {
  content: string;
  status: string;
  title: string;
}) {
  const isStreaming = status === "streaming";
  const contentRef = useRef(content);
  const titleRef = useRef(title);
  contentRef.current = content;
  titleRef.current = title;

  useEffect(() => {
    const handler = () => openPrintWindow(contentRef.current, titleRef.current);
    window.addEventListener("pdf-artifact-download", handler);
    return () => window.removeEventListener("pdf-artifact-download", handler);
  }, []);

  const handlePrint = useCallback(() => {
    openPrintWindow(contentRef.current, titleRef.current);
  }, []);

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between border-b border-border/40 bg-muted/20 px-4 py-2">
        {isStreaming ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-block size-1.5 animate-pulse rounded-full bg-primary" />
            Generating report...
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">Clinical Report</span>
        )}
        <button
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
          disabled={isStreaming || !content}
          onClick={handlePrint}
          type="button"
        >
          <DownloadIcon size={13} />
          Download PDF
        </button>
      </div>
      <div
        className="min-h-0 flex-1 overflow-auto bg-white p-6"
        // biome-ignore lint/security/noDangerouslySetInnerHtml: LLM-generated clinical HTML report
        dangerouslySetInnerHTML={{
          __html:
            content ||
            "<p style='color:#999;padding:2rem'>Generating clinical report...</p>",
        }}
      />
    </div>
  );
}

export const pdfArtifact = new Artifact<"pdf", Metadata>({
  actions: [
    {
      description: "View Previous version",
      icon: <UndoIcon size={18} />,
      isDisabled: ({ currentVersionIndex }) => currentVersionIndex === 0,
      onClick: ({ handleVersionChange }) => {
        handleVersionChange("prev");
      },
    },
    {
      description: "View Next version",
      icon: <RedoIcon size={18} />,
      isDisabled: ({ isCurrentVersion }) => isCurrentVersion,
      onClick: ({ handleVersionChange }) => {
        handleVersionChange("next");
      },
    },
  ],
  content: ({ content, status, title }) => (
    <PdfPreview content={content} status={status} title={title} />
  ),
  description:
    "Patient clinical report with X-ray findings, TB classification, drug recommendations, and safety notes.",
  initialize: () => null,
  kind: "pdf",
  onStreamPart: ({ streamPart, setArtifact }) => {
    if (streamPart.type === "data-pdfDelta") {
      setArtifact((draftArtifact) => ({
        ...draftArtifact,
        content: streamPart.data,
        isVisible: true,
        status: "streaming",
      }));
    }
  },
  toolbar: [
    {
      description: "Download as PDF",
      icon: <DownloadIcon />,
      onClick: ({ sendMessage: _sendMessage }) => {
        // Trigger the download action by finding the download button in the actions bar
        // and dispatching a custom event the PdfPreview can listen to
        window.dispatchEvent(new CustomEvent("pdf-artifact-download"));
      },
    },
  ],
});
