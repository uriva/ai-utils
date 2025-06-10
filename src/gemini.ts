import {
  type Content,
  type GenerateContentRequest,
  GoogleGenerativeAI,
  type ModelParams,
} from "@google/generative-ai";
import { context } from "context-inject";
import { coerce, empty, map, pipe, remove } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import type { z, ZodSchema } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { zodToGeminiParameters } from "./mcpFramework.ts";
import { replaceSystem, structuredMsgs } from "./openai.ts";
import type { FnToSameFn, ModelOpts, TokenInjection } from "./utils.ts";

const tokenInjection: TokenInjection = context((): string => {
  throw new Error("no gemini token injected");
});

export const accessGeminiToken = tokenInjection.access;
export const injectGeminiToken = (token: string): FnToSameFn =>
  tokenInjection.inject(() => token);

const openAiToGeminiMessage = pipe(
  map(({ role, content }: ChatCompletionMessageParam): Content => ({
    role: role === "user" ? role : "model",
    parts: [{
      text: typeof content === "string" ? content : coerce(content?.toString()),
    }].filter((x) => x.text),
  })),
  remove(({ parts }: Content) => empty(parts)),
);

export const geminiGenJsonFromConvo: <T extends ZodSchema>(
  { thinking, mini }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
) => Promise<z.infer<T>> = async <T extends ZodSchema>(
  { mini }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const cachedCall = makeCache(
    "geminiCompletionResponseText",
  )((modelParams: ModelParams, req: GenerateContentRequest) =>
    new GoogleGenerativeAI(tokenInjection.access()).getGenerativeModel(
      modelParams,
    ).generateContent(req).then((x) => x.response.text())
  );
  return JSON.parse(
    await cachedCall(
      {
        model: mini
          ? "gemini-2.5-flash"
          : "gemini-2.5-pro-preview-05-06",
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: zodToGeminiParameters(zodType),
        },
      },
      {
        contents: pipe(
          map(replaceSystem("assistant")),
          openAiToGeminiMessage,
        )(messages),
      },
    ),
  );
};

export const geminiGenJson =
  <T extends ZodSchema>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string): Promise<z.TypeOf<T>> =>
    geminiGenJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
    );
