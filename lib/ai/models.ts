export const DEFAULT_CHAT_MODEL = "mixroute/gpt-4o-mini";

export const titleModel = {
  description: "Fast model for title generation",
  id: "groq/llama-3.1-8b-instant",
  name: "Llama 3.1 8B Instant",
  provider: "groq",
};

export type ModelCapabilities = {
  tools: boolean;
  vision: boolean;
  reasoning: boolean;
};

export type ChatModel = {
  id: string;
  name: string;
  provider: string;
  description: string;
  reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high";
};

export const LANGCHAIN_MODEL_PREFIX = "langchain/";
export const GROQ_MODEL_PREFIX = "groq/";
export const MIXROUTE_MODEL_PREFIX = "mixroute/";

export function isLangChainModel(modelId: string) {
  return modelId.startsWith(LANGCHAIN_MODEL_PREFIX);
}

export function isGroqModel(modelId: string) {
  return modelId.startsWith(GROQ_MODEL_PREFIX);
}

export function isMixRouteModel(modelId: string) {
  return modelId.startsWith(MIXROUTE_MODEL_PREFIX);
}

/** Strip provider prefix for the underlying OpenAI-compatible API. */
export function toProviderModelId(modelId: string) {
  if (isLangChainModel(modelId)) {
    return modelId.replace(/^langchain\//, "");
  }
  if (isMixRouteModel(modelId)) {
    return modelId.replace(/^mixroute\//, "");
  }
  if (isGroqModel(modelId)) {
    return modelId.replace(/^groq\//, "");
  }
  if (modelId.startsWith("openai/")) {
    return modelId.replace(/^openai\//, "");
  }
  return modelId;
}

export const chatModels: ChatModel[] = [
  {
    description: "OpenAI GPT-4o mini via MixRoute (vision + tools)",
    id: "mixroute/gpt-4o-mini",
    name: "GPT-4o mini (MixRoute)",
    provider: "mixroute",
  },
  {
    description: "OpenAI GPT-4.1 mini via MixRoute",
    id: "mixroute/gpt-4.1-mini",
    name: "GPT-4.1 mini (MixRoute)",
    provider: "mixroute",
  },
  {
    description: "Google Gemini 2.5 Flash via MixRoute (vision)",
    id: "mixroute/gemini-2.5-flash",
    name: "Gemini 2.5 Flash (MixRoute)",
    provider: "mixroute",
  },
  {
    description: "Anthropic Claude Haiku 4.5 via MixRoute",
    id: "mixroute/claude-haiku-4-5",
    name: "Claude Haiku 4.5 (MixRoute)",
    provider: "mixroute",
  },
  {
    description: "DeepSeek V4 Flash via MixRoute",
    id: "mixroute/deepseek-v4-flash",
    name: "DeepSeek V4 Flash (MixRoute)",
    provider: "mixroute",
  },
  {
    description: "Vision + tools — upload X-rays and images (Groq Qwen 3.6)",
    id: "groq/qwen/qwen3.6-27b",
    name: "Qwen 3.6 27B Vision (Groq)",
    provider: "groq",
  },
  {
    description:
      "Custom LangChain backend with RAG (ai-backend /v1/chat/completions)",
    id: "langchain/healthcare-rag",
    name: "Healthcare RAG (LangChain)",
    provider: "langchain",
  },
  {
    description: "Fast Groq chat model with tool use (text only)",
    id: "groq/llama-3.3-70b-versatile",
    name: "Llama 3.3 70B (Groq)",
    provider: "groq",
  },
  {
    description: "Very fast Groq model for quick replies (text only)",
    id: "groq/llama-3.1-8b-instant",
    name: "Llama 3.1 8B Instant (Groq)",
    provider: "groq",
  },
];

const STATIC_CAPABILITIES: Record<string, ModelCapabilities> = {
  "groq/llama-3.1-8b-instant": {
    reasoning: false,
    tools: true,
    vision: false,
  },
  "groq/llama-3.3-70b-versatile": {
    reasoning: false,
    tools: true,
    vision: false,
  },
  "groq/qwen/qwen3.6-27b": {
    reasoning: true,
    tools: true,
    vision: true,
  },
  "langchain/healthcare-rag": {
    reasoning: false,
    tools: true,
    vision: false,
  },
  "mixroute/claude-haiku-4-5": {
    reasoning: false,
    tools: true,
    vision: true,
  },
  "mixroute/deepseek-v4-flash": {
    reasoning: false,
    tools: true,
    vision: false,
  },
  "mixroute/gemini-2.5-flash": {
    reasoning: false,
    tools: true,
    vision: true,
  },
  "mixroute/gpt-4.1-mini": {
    reasoning: false,
    tools: true,
    vision: true,
  },
  "mixroute/gpt-4o-mini": {
    reasoning: false,
    tools: true,
    vision: true,
  },
};

/** Sync capabilities for client UI (attachment button, model badges). */
export const modelCapabilities: Record<string, ModelCapabilities> =
  Object.fromEntries(
    chatModels.map((model) => [
      model.id,
      STATIC_CAPABILITIES[model.id] ?? {
        reasoning: false,
        tools: true,
        vision: false,
      },
    ])
  );

export function getCapabilities(): Record<string, ModelCapabilities> {
  return modelCapabilities;
}

export const isDemo = process.env.IS_DEMO === "1";

export type ModelWithCapabilities = ChatModel & {
  capabilities: ModelCapabilities;
};

/** The curated model list, annotated with capabilities for the client. */
export function getAvailableModels(): ModelWithCapabilities[] {
  const capabilities = getCapabilities();
  return chatModels.map((model) => ({
    ...model,
    capabilities: capabilities[model.id] ?? {
      reasoning: false,
      tools: true,
      vision: false,
    },
  }));
}

export function getActiveModels(): ChatModel[] {
  return chatModels;
}

export const allowedModelIds = new Set(chatModels.map((m) => m.id));

export const modelsByProvider = chatModels.reduce(
  (acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  },
  {} as Record<string, ChatModel[]>
);

export type ModelAvailability = "healthy" | "impacted" | "unknown";

export async function getModelAvailability(
  modelId: string
): Promise<ModelAvailability> {
  const model = chatModels.find((item) => item.id === modelId);

  if (!model) {
    return "unknown";
  }

  if (isLangChainModel(modelId)) {
    try {
      const baseURL = (
        process.env.LANGCHAIN_API_URL ?? "http://localhost:8001"
      ).replace(/\/$/, "");
      const res = await fetch(`${baseURL}/health`, {
        next: { revalidate: 30 },
      });
      return res.ok ? "healthy" : "impacted";
    } catch {
      return "impacted";
    }
  }

  if (isMixRouteModel(modelId)) {
    return process.env.MIXROUTE_API_KEY ? "healthy" : "impacted";
  }

  if (isGroqModel(modelId)) {
    return process.env.GROQ_API_KEY ? "healthy" : "impacted";
  }

  return "unknown";
}
