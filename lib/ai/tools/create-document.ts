import { tool, type UIMessageStreamWriter } from "ai";
import type { Session } from "next-auth";
import { z } from "zod";
import {
  artifactKinds,
  documentHandlersByArtifactKind,
} from "@/lib/artifacts/server";
import type { ChatMessage } from "@/lib/types";
import { generateUUID } from "@/lib/utils";

type CreateDocumentProps = {
  session: Session;
  dataStream: UIMessageStreamWriter<ChatMessage>;
  modelId: string;
};

export const createDocument = ({
  session,
  dataStream,
  modelId,
}: CreateDocumentProps) =>
  tool({
    description:
      "Create an artifact. You MUST specify kind: use 'pdf' for patient clinical reports with X-ray results, TB classification, drug recommendations, and safety notes; 'text' for essays/writing/documents; 'sheet' for spreadsheets/data tables.",
    execute: async ({ title, kind }) => {
      const id = generateUUID();

      dataStream.write({
        data: kind,
        transient: true,
        type: "data-kind",
      });

      dataStream.write({
        data: id,
        transient: true,
        type: "data-id",
      });

      dataStream.write({
        data: title,
        transient: true,
        type: "data-title",
      });

      dataStream.write({
        data: null,
        transient: true,
        type: "data-clear",
      });

      const documentHandler = documentHandlersByArtifactKind.find(
        (documentHandlerByArtifactKind) =>
          documentHandlerByArtifactKind.kind === kind
      );

      if (!documentHandler) {
        throw new Error(`No document handler found for kind: ${kind}`);
      }

      await documentHandler.onCreateDocument({
        dataStream,
        id,
        modelId,
        session,
        title,
      });

      dataStream.write({ data: null, transient: true, type: "data-finish" });

      return {
        content:
          kind === "pdf"
            ? "A clinical PDF report was created and is now visible to the user. They can download it as a PDF."
            : "A document was created and is now visible to the user.",
        id,
        kind,
        title,
      };
    },
    inputSchema: z.object({
      kind: z
        .enum(artifactKinds)
        .describe(
          "REQUIRED. 'pdf' for patient clinical reports, 'text' for essays/writing/documents, 'sheet' for spreadsheets/data tables"
        ),
      title: z
        .string()
        .describe(
          "The title of the artifact — for pdf reports include the patient name e.g. 'Clinical Report — John Doe'"
        ),
    }),
  });
