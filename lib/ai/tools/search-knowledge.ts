import { tool } from "ai";
import { z } from "zod";
import { searchLocalTbKnowledge } from "@/lib/tb/knowledge";

type RagChunk = {
  id: string;
  content: string;
  score: number;
  source: string;
  metadata?: Record<string, unknown>;
};

export const searchKnowledge = tool({
  description:
    "Search the healthcare / product knowledge base (LangChain RAG backend). " +
    "Use for clinical imaging guidelines, architecture notes, or domain facts. " +
    "Cite returned sources in your answer.",
  execute: async ({ query, topK }) => {
    const localTbChunks = searchLocalTbKnowledge(query, topK ?? 4);
    const baseURL = (
      process.env.LANGCHAIN_API_URL ?? "http://localhost:8001"
    ).replace(/\/$/, "");

    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };
    if (process.env.LANGCHAIN_API_KEY) {
      headers.Authorization = `Bearer ${process.env.LANGCHAIN_API_KEY}`;
    }

    try {
      const response = await fetch(`${baseURL}/rag/search`, {
        body: JSON.stringify({ query, top_k: topK ?? 4 }),
        headers,
        method: "POST",
      });

      if (!response.ok) {
        const detail = await response.text();
        return {
          chunks: [] as RagChunk[],
          error: `RAG backend error (${response.status}): ${detail || response.statusText}`,
        };
      }

      const data = (await response.json()) as {
        query: string;
        chunks: RagChunk[];
      };

      return {
        chunks: [...localTbChunks, ...data.chunks].slice(0, topK ?? 4),
        citationHint:
          "Cite sources using [source path] from each chunk when answering.",
        query: data.query,
      };
    } catch (error) {
      return {
        chunks: localTbChunks as RagChunk[],
        error:
          error instanceof Error
            ? `Cannot reach LangChain backend at ${baseURL}: ${error.message}`
            : `Cannot reach LangChain backend at ${baseURL}`,
      };
    }
  },
  inputSchema: z.object({
    query: z
      .string()
      .min(1)
      .describe("Search query for guidelines or product knowledge"),
    topK: z
      .number()
      .int()
      .min(1)
      .max(10)
      .optional()
      .describe("Number of chunks to retrieve (default 4)"),
  }),
});
