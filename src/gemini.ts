import {
  type Content,
  type GenerateContentRequest,
  GoogleGenerativeAI,
  type ModelParams,
} from "@google/generative-ai";
import { context } from "context-inject";
import { coerce, empty, map, pipe, remove, retry } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import type { z, ZodSchema } from "zod";
import { makeCache } from "./cacher.ts";
import { zodToGeminiParameters } from "./mcpFramework.ts";
import { extractJson, replaceSystem, structuredMsgs } from "./openai.ts";
import {
  appendTypingInstruction,
  FnToSameFn,
  ModelOpts,
  TokenInjection,
} from "./utils.ts";

const tokenInjection: TokenInjection = context((): string => {
  throw new Error("no gemini token injected");
});

export const accessGeminiToken = tokenInjection.access;
export const injectGeminiToken = (token: string): FnToSameFn =>
  tokenInjection.inject(() => token);

const openAiToGeminiMessage = pipe(
  map((
    { role, content }: ChatCompletionMessageParam,
  ): Content => ({
    role: role === "user" ? role : "model",
    parts: [{
      text: typeof content === "string" ? content : coerce(content?.toString()),
    }]
      .filter((x) => x.text),
  })),
  remove(({ parts }: Content) => empty(parts)),
);

export const geminiGenJsonFromConvo: <T extends ZodSchema>(
  { thinking, mini }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
) => Promise<z.infer<T>> = retry(
  10000,
  2,
  async <T extends ZodSchema>(
    { thinking, mini }: ModelOpts,
    messages: ChatCompletionMessageParam[],
    zodType: T,
  ): Promise<z.infer<T>> => {
    const cachedCall = makeCache(
      "geminiCompletionResponseText",
    )((modelParams: ModelParams, req: GenerateContentRequest) =>
      new GoogleGenerativeAI(tokenInjection.access()).getGenerativeModel(
        modelParams,
      )
        .generateContent(req).then((x) => x.response.text())
    );
    if (!thinking) {
      return (JSON.parse(
        await cachedCall(
          {
            model: mini
              ? "gemini-2.5-pro-preview-03-25"
              : "gemini-2.5-pro-preview-03-25",
            generationConfig: {
              responseMimeType: "application/json",
              responseSchema: zodToGeminiParameters(zodType),
            },
          },
          {
            contents: pipe(
              map(replaceSystem("assistant")),
              openAiToGeminiMessage,
            )(
              messages,
            ),
          },
        ),
      )) as z.infer<T>;
    }
    return extractJson(geminiGenJsonFromConvo)(zodType)(
      await cachedCall(
        { model: "gemini-2.5-pro-preview-03-25" },
        {
          contents: pipe(
            map(replaceSystem("assistant")),
            appendTypingInstruction(zodType, "assistant"),
            openAiToGeminiMessage,
          )(messages),
        },
      ),
    );
  },
);

export const geminiGenJson =
  <T extends ZodSchema>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string): Promise<z.TypeOf<T>> =>
    geminiGenJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
    );
