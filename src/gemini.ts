import {
  type Content,
  type FunctionDeclaration,
  type GenerateContentParameters,
  GoogleGenAI,
} from "@google/genai";
import { context, type Injection, type Injector } from "@uri/inject";
import { coerce, empty, map, pipe, remove } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z, type ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { structuredMsgs } from "./openai.ts";
import type { ModelOpts } from "./utils.ts";

// deno-lint-ignore no-explicit-any
const isRedundantAnyMember = (x: any) =>
  Object.keys(x).length === 1 && typeof (x.not) === "object" &&
  Object.keys(x.not).length === 0;

// deno-lint-ignore no-explicit-any
const removeAdditionalProperties = <T>(obj: Record<string, any>) => {
  if (typeof obj === "object" && obj !== null) {
    let newObj = { ...obj };
    if (obj.anyOf) {
      // deno-lint-ignore no-explicit-any
      newObj.anyOf = obj.anyOf.filter((x: any) =>
        x.type !== "null" && !isRedundantAnyMember(x)
      ).map(removeAdditionalProperties);
    }
    if (newObj.anyOf?.length === 1) newObj = newObj.anyOf[0];
    if (Array.isArray(obj.type)) {
      if (obj.type.find((x: string) => x === "null")) {
        newObj.nullable = true;
      }
      newObj.type = obj.type.find((x) => x !== "null");
    }
    if ("additionalProperties" in newObj) {
      newObj.additionalProperties = undefined;
    }
    for (const key in newObj) {
      if (key in newObj) {
        if (Array.isArray(newObj[key])) {
          newObj[key] = newObj[key].map(removeAdditionalProperties);
        } else if (
          typeof newObj[key] === "object" && newObj[key] !== null
        ) {
          newObj[key] = removeAdditionalProperties(newObj[key]);
        }
      }
    }
    return newObj;
  }
  return obj;
};

export const zodToGeminiParameters = (zodObj: ZodType): FunctionDeclaration => {
  const jsonSchema = removeAdditionalProperties(z.toJSONSchema(zodObj));
  // deno-lint-ignore no-unused-vars
  const { $schema, ...rest } = jsonSchema;
  return rest;
};

const tokenInjection: Injection<() => string> = context((): string => {
  throw new Error("no gemini token injected");
});

export const accessGeminiToken = tokenInjection.access;
export const injectGeminiToken = (token: string): Injector =>
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
export const geminiFlashVersion = "gemini-2.5-flash-preview-05-20";
export const geminiFlashImageVersion = "gemini-2.5-flash-image";

export const geminiGenJsonFromConvo: <T extends ZodType>(
  { mini }: ModelOpts,
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
      model: mini ? geminiFlashVersion : geminiProVersion,
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
