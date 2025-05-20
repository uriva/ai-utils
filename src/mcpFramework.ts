import { context } from "context-inject";
import {
  coerce,
  empty,
  Func,
  init,
  last,
  logAfter,
  map,
  nonempty,
  pipe,
  sideEffect,
} from "gamla";
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
import { zodToJsonSchema } from "npm:zod-to-json-schema@3.24.5";
import type { z, ZodSchema } from "zod";
import { makeCache } from "./cacher.ts";
import { accessGeminiToken } from "./gemini.ts";
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

export type Action<T extends ZodSchema, O> = {
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
  botNameInHistory: string;
};

const combineConsecutiveModelMessages = (acc: Content[], curr: Content) =>
  nonempty(acc) && last(acc).role === curr.role
    ? [...init(acc), {
      ...last(acc),
      parts: [...last(acc).parts, ...curr.parts],
    }]
    : [...acc, curr];

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
  async <T extends ZodSchema, O>(
    { name, args }: FunctionCall,
  ): Promise<FunctionResponsePart> => {
    const { handler, parameters }: Action<T, O> = coerce(
      actions.find(({ name: n }) => n === name),
    );
    return {
      functionResponse: {
        name,
        response: {
          result: await handler(parameters.parse(args) as T),
        },
      },
    };
  };

const debugLogsAfter = <F extends Func>(f: F) =>
  pipe(f, sideEffect(debugLogs.access<Awaited<ReturnType<F>>>));

export const runBot = async (
  { actions, prompt, botNameInHistory }: BotSpec,
) => {
  let c = 0;
  const contents = (await getHistory()).map(({ from, text }) => ({
    role: from !== botNameInHistory && from !== systemUser ? "user" : "model",
    parts: [{ text }],
  })).reduce(combineConsecutiveModelMessages, []);
  const thoughts: Content[] = [];
  while (true) {
    c++;
    if (c > 5) throw new Error("Too many iterations");
    const { text, functionCalls } = await pipe(
      debugLogsAfter(geminiInput),
      debugLogsAfter(callGemini({ model: "gemini-2.5-pro-preview-03-25" })),
    )(prompt, actions, [...contents, ...thoughts]);
    if (text) await reply(text);
    const calls = functionCalls ?? [];
    const results = await map(callToResult(actions))(calls);
    for (let i = 0; i < results.length; i++) {
      agentSystemLog.access(
        [
          `Agent called tool:\n${JSON.stringify(calls[i], null, 2)}`,
          `Got result:\n${JSON.stringify(results[i], null, 2)}`,
        ].join("\n\n"),
      );
      const call = { role: "model", parts: [{ functionCall: calls[i] }] };
      const result = results[i];
      thoughts.push(call);
      thoughts.push({ role: "user", parts: [result] });
    }
    if (empty(functionCalls)) return;
  }
};

const agentSystemLog: SomethingInjection<(text: string) => void> = context(
  (_text: string) => {},
);

const debugLogs: SomethingInjection<<T>(t: T) => void> = context(
  <T>(_: T) => {},
);

export const injectedDebugLogs = debugLogs.inject;

export const injectAgentSystemLog = agentSystemLog.inject;

const historyInjection: SomethingInjection<() => Promise<HistoryEvent[]>> =
  context((): Promise<HistoryEvent[]> => {
    throw new Error("History not injected");
  });

export const getHistory = historyInjection.access;
export const injectAccessHistory = historyInjection.inject;

const replyInjection: SomethingInjection<(text: string) => Promise<void>> =
  context((_text: string): Promise<void> => {
    throw new Error("Reply not injected");
  });

export const reply = replyInjection.access;
export const injectReply = replyInjection.inject;

export type HistoryEvent = { text: string; from: string; time: number };

export type MessageDraft = { from: string; text: string };
