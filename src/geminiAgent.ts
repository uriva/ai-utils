import {
  type Content,
  type FunctionCall,
  type FunctionDeclarationSchema,
  type FunctionResponsePart,
  type GenerateContentRequest,
  type GenerateContentResult,
  GoogleGenerativeAI,
  type ModelParams,
} from "@google/generative-ai";
import { context } from "context-inject";
import { coerce, empty, type Func, map, pipe, sideEffect } from "gamla";
import { zodToJsonSchema } from "npm:zod-to-json-schema@3.24.5";
import type { z, ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { accessGeminiToken, geminiProVersion } from "./gemini.ts";
import type { SomethingInjection } from "./utils.ts";

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

// deno-lint-ignore no-explicit-any
export const zodToGeminiParameters = (zodObj: any) => {
  const jsonSchema = removeAdditionalProperties(zodToJsonSchema(zodObj));
  // deno-lint-ignore no-unused-vars
  const { $schema, ...rest } = jsonSchema;
  return rest as FunctionDeclarationSchema;
};

export const systemUser = "system";

export type Action<T extends ZodType, O> = {
  description: string;
  name: string;
  parameters: T;
  handler: (params: z.infer<T>) => Promise<O>;
};

type GeminiOutput = {
  text: string;
  functionCalls: FunctionCall[];
};

const callGemini = (modelParams: ModelParams) =>
  makeCache("gemini response with function calls")((
    req: GenerateContentRequest,
  ): Promise<GeminiOutput> =>
    new GoogleGenerativeAI(accessGeminiToken())
      .getGenerativeModel(modelParams).generateContent(req).then((
        { response }: GenerateContentResult,
      ) => ({
        text: response.text(),
        functionCalls: response.functionCalls() ?? [],
      }))
  );

export type BotSpec = {
  // deno-lint-ignore no-explicit-any
  actions: Action<any, any>[];
  prompt: string;
};

// deno-lint-ignore no-explicit-any
const actionToTool = ({ name, description, parameters }: Action<any, any>) => ({
  name,
  description,
  parameters: zodToGeminiParameters(parameters),
});

const geminiInput = (
  systemInstruction: string,
  // deno-lint-ignore no-explicit-any
  actions: Action<any, any>[],
  contents: Content[],
): GenerateContentRequest => ({
  systemInstruction,
  tools: [{ functionDeclarations: actions.map(actionToTool) }],
  contents,
});

const callToResult =
  // deno-lint-ignore no-explicit-any
  (actions: Action<any, any>[]) =>
  async <T extends ZodType, O>(
    { name, args }: FunctionCall,
  ): Promise<FunctionResponsePart> => {
    const { handler, parameters }: Action<T, O> = coerce(
      actions.find(({ name: n }) => n === name),
    );
    return {
      functionResponse: {
        name,
        response: {
          result: await handler(parameters.parse(args)),
        },
      },
    };
  };

const debugLogsAfter = <F extends Func>(f: F) =>
  pipe(f, sideEffect(debugLogs.access<Awaited<ReturnType<F>>>));

export const runBot = async ({ actions, prompt }: BotSpec) => {
  let c = 0;
  while (true) {
    c++;
    if (c > 5) throw new Error("Too many iterations");
    const { text, functionCalls } = await pipe(
      debugLogsAfter(geminiInput),
      debugLogsAfter(callGemini({ model: geminiProVersion })),
    )(prompt, actions, await getHistory());
    if (text) await outputEvent({ role: "model", parts: [{ text }] });
    const calls = functionCalls ?? [];
    const results = await map(callToResult(actions))(calls);
    for (let i = 0; i < results.length; i++) {
      await outputEvent({ role: "model", parts: [{ functionCall: calls[i] }] });
      await outputEvent({ role: "user", parts: [results[i]] });
    }
    if (empty(functionCalls)) return;
  }
};

const debugLogs: SomethingInjection<<T>(t: T) => void> = context(
  <T>(_: T) => {},
);

export const injectedDebugLogs = debugLogs.inject;

const modelOutput: SomethingInjection<(event: Content) => Promise<void>> =
  context((_event: Content): Promise<void> => {
    throw new Error("output function not injected");
  });

export const outputEvent = modelOutput.access;
export const injectOutputEvent = modelOutput.inject;

const historyInjection: SomethingInjection<() => Promise<Content[]>> = context(
  (): Promise<Content[]> => {
    throw new Error("History not injected");
  },
);

export const getHistory = historyInjection.access;
export const injectAccessHistory = historyInjection.inject;
