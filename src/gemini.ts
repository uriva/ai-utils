import {
  type Content,
  type GenerateContentParameters,
  GoogleGenAI,
} from "@google/genai";
import { context } from "context-inject";
import { coerce, empty, map, pipe, remove } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import type { z, ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { zodToGeminiParameters } from "./geminiAgent.ts";
import { structuredMsgs } from "./openai.ts";
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
  remove(({ parts }: Content) => empty(parts ?? [])),
);

export const geminiProVersion = "gemini-2.5-pro-preview-06-05";

export const geminiGenJsonFromConvo: <T extends ZodType>(
  { thinking, mini }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
) => Promise<z.infer<T>> = async <T extends ZodType>(
  { mini }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const cacher = makeCache("geminiCompletionResponseText-v2");
  const cachedCall = cacher((req: GenerateContentParameters) =>
    new GoogleGenAI({ apiKey: tokenInjection.access() }).models.generateContent(
      req,
    ).then(({ text }) => text ?? "{}")
  );
  return JSON.parse(
    await cachedCall({
      model: mini ? "gemini-2.5-flash-preview-05-20" : geminiProVersion,
      config: {
        responseMimeType: "application/json",
        responseSchema: zodToGeminiParameters(zodType),
      },
      contents: pipe(openAiToGeminiMessage)(messages),
    }),
  );
};

export const geminiGenJson =
  <T extends ZodType>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string): Promise<z.TypeOf<T>> =>
    geminiGenJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
    );
