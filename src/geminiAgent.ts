import {
  type Content,
  type FunctionCall,
  type FunctionDeclaration,
  type FunctionResponse,
  GenerateContentConfig,
  type GenerateContentParameters,
  type GenerateContentResponse,
  GoogleGenAI,
  type Part,
} from "@google/genai";

import { context } from "context-inject";
import { coerce, empty, type Func, map, pipe, sideEffect } from "gamla";
import { z, type ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { accessGeminiToken, geminiProVersion } from "./gemini.ts";
import type { FnToSameFn, SomethingInjection } from "./utils.ts";

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
  const jsonSchema = removeAdditionalProperties(z.toJSONSchema(zodObj));
  // deno-lint-ignore no-unused-vars
  const { $schema, ...rest } = jsonSchema;
  return rest as FunctionDeclaration;
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

const callGemini = (model: string) => ((
  req: Omit<GenerateContentParameters, "model">,
): Promise<GeminiOutput> =>
  new GoogleGenAI({ apiKey: accessGeminiToken() }).models.generateContent({
    model,
    ...req,
  }).then((
    { text, functionCalls }: GenerateContentResponse,
  ) => ({ text: text ?? "", functionCalls: functionCalls ?? [] })));

export type BotSpec = {
  // deno-lint-ignore no-explicit-any
  actions: Action<any, any>[];
  prompt: string;
  maxIterations: number;
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
): Omit<GenerateContentParameters, "model"> => ({
  config: {
    systemInstruction,
    tools: [{ functionDeclarations: actions.map(actionToTool) }],
    // Only set allowedFunctionNames if mode is ANY. Since mode is not set, omit allowedFunctionNames.
    toolConfig: {
      functionCallingConfig: {
        // allowedFunctionNames: actions.map((a) => a.name),
      },
    },
  },
  contents,
});

// deno-lint-ignore no-explicit-any
const parseWithCatch = <T extends ZodType>(parameters: T, args: any) => {
  try {
    return parameters.parse(args);
  } catch (error) {
    console.error("Error parsing function call arguments:", error);
    return null;
  }
};

const callToResult =
  // deno-lint-ignore no-explicit-any
  (actions: Action<any, any>[]) =>
  async <T extends ZodType, O>(
    { name, args }: FunctionCall,
  ): Promise<Part> => {
    const { handler, parameters }: Action<T, O> = coerce(
      actions.find(({ name: n }) => n === name),
    );
    const parsedArgs = parseWithCatch(parameters, args);
    return {
      functionResponse: {
        name,
        response: {
          result: parsedArgs
            ? await handler(parsedArgs)
            : `Invalid arguments for function`,
        },
      },
    };
  };

const debugLogsAfter = <F extends Func>(f: F) =>
  pipe(f, sideEffect(debugLogs.access<Awaited<ReturnType<F>>>));

export const functionCallTurn = (functionCall: FunctionCall) => ({
  role: "model",
  parts: [{ functionCall }],
});

export const functionResultTurn = (
  functionResponse: FunctionResponse,
): Content => ({
  role: "user",
  parts: [{ functionResponse }],
});

export const runBot = async ({ actions, prompt, maxIterations }: BotSpec) => {
  const cacher = makeCache("gemini response with function calls v2");
  let c = 0;
  while (true) {
    c++;
    if (c > maxIterations) throw new Error("Too many iterations");
    const history = await getHistory();
    if (empty(history) || (history[0].parts ?? [])[0].functionCall) {
      history.unshift({
        role: "user",
        parts: [{ text: "<conversation started>" }],
      });
    }
    const { text, functionCalls } = await pipe(
      debugLogsAfter(geminiInput),
      debugLogsAfter(cacher(callGemini)(geminiProVersion)),
    )(prompt, actions, history);
    if (text) await outputEvent({ role: "model", parts: [{ text }] });
    const calls = functionCalls ?? [];
    const results = await map(callToResult(actions))(calls);
    for (let i = 0; i < results.length; i++) {
      await outputEvent(functionCallTurn(calls[i]));
      await outputEvent(
        functionResultTurn(coerce(results[i].functionResponse)),
      );
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

export const injectInMemoryHistory = (inMemoryHistory: Content[]): FnToSameFn =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );
