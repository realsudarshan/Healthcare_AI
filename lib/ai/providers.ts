import { createOpenAI } from "@ai-sdk/openai";
import { customProvider } from "ai";
import { isTestEnvironment } from "../constants";
import {
  isGroqModel,
  isLangChainModel,
  isMixRouteModel,
  titleModel,
  toProviderModelId,
} from "./models";

export const myProvider = isTestEnvironment
  ? (() => {
      const {
        chatModel,
        titleModel: mockTitleModel,
      } = require("./models.mock");
      return customProvider({
        languageModels: {
          "chat-model": chatModel,
          "title-model": mockTitleModel,
        },
      });
    })()
  : null;

function getGroqProvider() {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GROQ_API_KEY is not set. Add it to .env.local (https://console.groq.com)."
    );
  }

  return createOpenAI({
    apiKey,
    baseURL: "https://api.groq.com/openai/v1",
    name: "groq",
  });
}

function getMixRouteProvider() {
  const apiKey = process.env.MIXROUTE_API_KEY;
  if (!apiKey) {
    throw new Error(
      "MIXROUTE_API_KEY is not set. Add it to .env.local (https://mixroute.ai)."
    );
  }

  const baseURL = (
    process.env.MIXROUTE_BASE_URL ?? "https://api.mixroute.ai/v1"
  ).replace(/\/$/, "");

  return createOpenAI({
    apiKey,
    baseURL,
    name: "mixroute",
  });
}

function getLangChainProvider() {
  const baseURL = (
    process.env.LANGCHAIN_API_URL ?? "http://localhost:8001"
  ).replace(/\/$/, "");

  return createOpenAI({
    apiKey: process.env.LANGCHAIN_API_KEY || "not-needed",
    baseURL: `${baseURL}/v1`,
    name: "langchain-backend",
  });
}

function getOpenAIProvider() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is not set. Add it to .env.local to use OpenAI models."
    );
  }

  return createOpenAI({
    apiKey,
    name: "openai",
  });
}

export function getLanguageModel(modelId: string) {
  if (isTestEnvironment && myProvider) {
    return myProvider.languageModel(modelId);
  }

  const apiModelId = toProviderModelId(modelId);

  if (isLangChainModel(modelId)) {
    return getLangChainProvider().chat(apiModelId);
  }

  if (isMixRouteModel(modelId)) {
    return getMixRouteProvider().chat(apiModelId);
  }

  if (
    isGroqModel(modelId) ||
    modelId.startsWith("llama") ||
    !modelId.includes("/")
  ) {
    return getGroqProvider().chat(apiModelId);
  }

  if (modelId.startsWith("openai/")) {
    return getOpenAIProvider().chat(apiModelId);
  }

  // Default: Groq-compatible OpenAI API
  return getGroqProvider().chat(apiModelId);
}

export function getTitleModel() {
  if (isTestEnvironment && myProvider) {
    return myProvider.languageModel("title-model");
  }

  return getLanguageModel(titleModel.id);
}
