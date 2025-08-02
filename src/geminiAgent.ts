import {
  type Content,
  type FunctionCall,
  type FunctionDeclaration,
  type GenerateContentParameters,
  type GenerateContentResponse,
  GoogleGenAI,
} from "@google/genai";
import { context } from "context-inject";
import {
  coerce,
  each,
  empty,
  type Func,
  groupBy,
  last,
  pipe,
  sideEffect,
} from "gamla";
import { z, type ZodType } from "zod/v4";
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

export const zodToGeminiParameters = (zodObj: ZodType): FunctionDeclaration => {
  const jsonSchema = removeAdditionalProperties(z.toJSONSchema(zodObj));
  // deno-lint-ignore no-unused-vars
  const { $schema, ...rest } = jsonSchema;
  return rest;
};

export const systemUser = "system";

export type Tool<T extends ZodType> = {
  description: string;
  name: string;
  parameters: T;
  handler: (params: z.infer<T>) => Promise<string>;
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

export type AgentSpec = {
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[];
  prompt: string;
  maxIterations: number;
  // deno-lint-ignore no-explicit-any
  onMaxIterationsReached: () => any;
};

// deno-lint-ignore no-explicit-any
const actionToTool = ({ name, description, parameters }: Tool<any>) => ({
  name,
  description,
  parameters: zodToGeminiParameters(parameters),
});

const geminiInput = (
  systemInstruction: string,
  // deno-lint-ignore no-explicit-any
  actions: Tool<any>[],
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

const parseWithCatch = <T extends ZodType>(
  parameters: T,
  // deno-lint-ignore no-explicit-any
  args: any,
): { ok: false; error: Error } | { ok: true; result: z.infer<T> } => {
  try {
    return { ok: true, result: parameters.parse(args) };
  } catch (error) {
    console.error("Error parsing function call arguments:", error);
    return { ok: false, error: error as Error };
  }
};

const callToResult =
  // deno-lint-ignore no-explicit-any
  (actions: Tool<any>[]) =>
  async <T extends ZodType>(
    fc: FunctionCall,
  ): Promise<{ name: string; result: string }> => {
    await outputEvent(toolUseTurn(fc));
    const { name, args } = fc;
    const action: Tool<T> | undefined = actions.find(({ name: n }) =>
      n === name
    );
    if (!name) throw new Error("Function call name is missing");
    if (!action) return { name, result: `Function ${name} not found` };
    const { handler, parameters } = action;
    const parseResult = parseWithCatch(parameters, args);
    return {
      name,
      result: parseResult.ok
        ? await handler(parseResult.result)
        : `Invalid arguments: ${JSON.stringify(parseResult.error)}`,
    };
  };

const debugLogsAfter = <F extends Func>(f: F) =>
  pipe(f, sideEffect(debugLogs.access<Awaited<ReturnType<F>>>));

type SharedFields = { id: MessageId; timestamp: number; isOwn: boolean };

type MessageId = string;

export type ParticipantUtterance = {
  type: "participant_utterance";
  isOwn: false;
  name: string;
  text: string;
} & SharedFields;

export type OwnText = {
  isOwn: true;
  type: "own_utterance";
  text: string;
} & SharedFields;

export type ParticipantReaction = {
  type: "participant_reaction";
  reaction: string;
  isOwn: false;
  onMessage: MessageId;
} & SharedFields;

export type OwnReaction = {
  type: "own_reaction";
  isOwn: true;
  reaction: string;
  onMessage: MessageId;
} & SharedFields;

export type ToolUse<T> = {
  type: "tool_call";
  isOwn: true;
  name: string;
  parameters: T;
} & SharedFields;

export type ToolResult = {
  type: "tool_result";
  isOwn: true;
  name: string;
  result: string;
} & SharedFields;

export type DoNothing = { type: "do_nothing" } & SharedFields;

export type HistoryEvent =
  | ParticipantUtterance
  | OwnText
  | OwnReaction
  | ParticipantReaction
  | ToolUse<unknown>
  | ToolResult
  | DoNothing;

const idGeneration: SomethingInjection<() => string> = context((): MessageId =>
  crypto.randomUUID()
);
const timestampGeneration: SomethingInjection<() => number> = context(
  (): number => Date.now(),
);

export const overrideTime = timestampGeneration.inject;
export const overrideIdGenerator = idGeneration.inject;

const sharedFields = () => ({
  id: idGeneration.access(),
  timestamp: timestampGeneration.access(),
});

export const toolUseTurn = (
  { name, args }: FunctionCall,
): HistoryEvent => ({
  type: "tool_call",
  ...sharedFields(),
  isOwn: true,
  timestamp: timestampGeneration.access(),
  name: coerce(name),
  parameters: args,
});

export const participantUtteranceTurn = (
  { name, text }: { name: string; text: string },
): HistoryEvent => ({
  type: "participant_utterance",
  isOwn: false,
  name: coerce(name),
  text,
  ...sharedFields(),
});

export const ownUtteranceTurn = (text: string): HistoryEvent => ({
  type: "own_utterance",
  isOwn: true,
  text,
  ...sharedFields(),
});

export const toolResultTurn = (
  { name, result }: { name: string; result: string },
): HistoryEvent => ({
  ...sharedFields(),
  type: "tool_result",
  isOwn: true,
  name,
  result,
});

const doNothingEvent = (): HistoryEvent => ({
  type: "do_nothing",
  isOwn: true,
  ...sharedFields(),
});

const indexById = (events: HistoryEvent[]) => {
  const eventIdToEvents = groupBy((x: HistoryEvent) => x.id)(
    events,
  );
  return (id: MessageId) => coerce(eventIdToEvents[id]?.[0]);
};

const historyEventToContent = (events: HistoryEvent[]) => {
  const eventById = indexById(events);
  return (e: HistoryEvent): Content => {
    if (e.type === "participant_utterance") {
      return { role: "user", parts: [{ text: e.text }] };
    }
    if (e.type === "own_utterance") {
      return { role: "model", parts: [{ text: e.text }] };
    }
    if (e.type === "tool_call") {
      return {
        role: "model",
        parts: [{
          functionCall: {
            name: e.name,
            args: e.parameters as Record<string, unknown>,
          },
        }],
      };
    }
    if (e.type === "tool_result") {
      return {
        role: "user",
        parts: [{
          functionResponse: {
            name: e.name,
            response: { result: e.result },
          },
        }],
      };
    }
    if (e.type === "participant_reaction" || e.type === "own_reaction") {
      const msg = eventById(e.onMessage);
      const text = typeof msg === "object" && "text" in msg
        ? (msg.text as string)
        : "";
      const role = e.type === "participant_reaction" ? "user" : "model";
      return {
        role,
        parts: [{ text: `reacted: ${e.reaction} to: ${text.slice(0, 100)}` }],
      };
    }
    if (e.type === "do_nothing") {
      return { role: "model", parts: [{ text: "" }] };
    }
    throw new Error(
      `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
    );
  };
};

export const runAgent = async (
  { tools, prompt, maxIterations, onMaxIterationsReached }: AgentSpec,
) => {
  const cacher = makeCache("gemini response with function calls v2");
  let c = 0;
  while (true) {
    c++;
    if (c > maxIterations) {
      onMaxIterationsReached();
      return;
    }
    const historyOuter = await getHistory();
    const history = historyOuter.map(historyEventToContent(historyOuter));
    if (empty(history) || (history[0].parts ?? [])[0].functionCall) {
      history.unshift({
        role: "user",
        parts: [{ text: "<conversation started>" }],
      });
    }
    const { text, functionCalls } = await pipe(
      debugLogsAfter(geminiInput),
      debugLogsAfter(cacher(callGemini(geminiProVersion))),
    )(prompt, tools, history);
    if (text) await outputEvent(ownUtteranceTurn(text));
    await each(pipe(callToResult(tools), toolResultTurn, outputEvent))(
      functionCalls,
    );
    if (empty(functionCalls) && !text) outputEvent(doNothingEvent());
    const newHistory = await getHistory();
    if (empty(functionCalls) && last(newHistory).isOwn) return;
  }
};

const debugLogs: SomethingInjection<<T>(t: T) => void> = context(
  <T>(_: T) => {},
);

export const injectDebugger = debugLogs.inject;

const modelOutput: SomethingInjection<(event: HistoryEvent) => Promise<void>> =
  context((_event: HistoryEvent): Promise<void> => {
    throw new Error("output function not injected");
  });

export const outputEvent = modelOutput.access;
export const injectOutputEvent = modelOutput.inject;

const historyInjection: SomethingInjection<() => Promise<HistoryEvent[]>> =
  context((): Promise<HistoryEvent[]> => {
    throw new Error("History not injected");
  });

export const getHistory = historyInjection.access;
export const injectAccessHistory = historyInjection.inject;
