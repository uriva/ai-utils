import {
  type Content,
  type FunctionCall,
  type FunctionDeclaration,
  type GenerateContentParameters,
  type GenerateContentResponse,
  GoogleGenAI,
  type Part,
} from "@google/genai";
import { context } from "context-inject";
import {
  coerce,
  each,
  empty,
  filter,
  groupBy,
  last,
  map,
  pipe,
  retry,
  sideEffect,
  timeit,
} from "gamla";
import { z, type ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import {
  accessGeminiToken,
  geminiFlashVersion,
  geminiProVersion,
} from "./gemini.ts";
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

type GeminiFunctiontoolPart = {
  type: "function_call";
  functionCall: FunctionCall;
  thoughtSignature?: string;
};

type GeminiPartOfInterest =
  | { type: "text"; text: string; thoughtSignature?: string }
  | GeminiFunctiontoolPart;

type GeminiOutput = GeminiPartOfInterest[];

const callGemini = (model: string) => ((
  req: Omit<GenerateContentParameters, "model">,
): Promise<GeminiOutput> =>
  new GoogleGenAI({ apiKey: accessGeminiToken() }).models.generateContent({
    model,
    ...req,
  })
    .then((resp: GenerateContentResponse): GeminiOutput =>
      (resp.candidates?.[0]?.content?.parts ?? [])
        .flatMap((part: Part): GeminiOutput => {
          const { text, functionCall, thoughtSignature } = part;
          if (functionCall) {
            return [{ type: "function_call", functionCall, thoughtSignature }];
          }
          if (typeof text === "string") {
            return [{ type: "text", text, thoughtSignature }];
          }
          return [];
        })
    ));

export type AgentSpec = {
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[];
  prompt: string;
  maxIterations: number;
  // deno-lint-ignore no-explicit-any
  onMaxIterationsReached: () => any;
  lightModel?: boolean;
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

const runSideEffectAfter = <T>(
  side: (x: T) => void | Promise<void>,
) =>
// deno-lint-ignore no-explicit-any
<F extends (...xs: any[]) => T | Promise<T>>(f: F) =>
  // @ts-expect-error cannot infer correctness
  pipe(f, sideEffect(side));

type SharedFields = { id: MessageId; timestamp: number; isOwn: boolean };

type MessageId = string;

type ParticipantDetail = { name: string };

export type ParticipantUtterance =
  & {
    type: "participant_utterance";
    isOwn: false;
    text: string;
  }
  & ParticipantDetail
  & SharedFields;

export type OwnUtterance = {
  isOwn: true;
  modelMetadata?: ModelMetadata;
  type: "own_utterance";
  text: string;
} & SharedFields;

export type ParticipantReaction =
  & {
    type: "participant_reaction";
    reaction: string;
    isOwn: false;
    onMessage: MessageId;
  }
  & ParticipantDetail
  & SharedFields;

export type OwnReaction = {
  type: "own_reaction";
  isOwn: true;
  modelMetadata?: ModelMetadata;
  reaction: string;
  onMessage: MessageId;
} & SharedFields;

export type ToolUse<T> = {
  type: "tool_call";
  isOwn: true;
  name: string;
  modelMetadata?: ModelMetadata;
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
  | OwnUtterance
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

type ModelMetadata = { type: "gemini"; thoughtSignature: string };

const toolUseTurnWithMetadata = (
  { name, args }: FunctionCall,
  modelMetadata: ModelMetadata | undefined,
): HistoryEvent => ({
  type: "tool_call",
  ...sharedFields(),
  isOwn: true,
  timestamp: timestampGeneration.access(),
  name: coerce(name),
  parameters: args,
  modelMetadata,
});

export const toolUseTurn = ({ name, args }: FunctionCall): HistoryEvent =>
  toolUseTurnWithMetadata({ name, args }, undefined);

export const participantUtteranceTurn = (
  { name, text }: { name: string; text: string },
): HistoryEvent => ({
  type: "participant_utterance",
  isOwn: false,
  name: coerce(name),
  text,
  ...sharedFields(),
});

const ownUtteranceTurnWithMetadata = (
  text: string,
  modelMetadata: ModelMetadata | undefined,
): HistoryEvent => ({
  type: "own_utterance",
  isOwn: true,
  modelMetadata,
  text,
  ...sharedFields(),
});

export const ownUtteranceTurn = (text: string): HistoryEvent =>
  ownUtteranceTurnWithMetadata(text, undefined);

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
      return { role: "user", parts: [{ text: `${e.name}: ${e.text}` }] };
    }
    if (e.type === "own_utterance") {
      return {
        role: "model",
        parts: [{
          thoughtSignature: e.modelMetadata?.thoughtSignature,
          text: e.text,
        }],
      };
    }
    if (e.type === "tool_call") {
      return {
        role: "model",
        parts: [{
          thoughtSignature: e.modelMetadata?.thoughtSignature,
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
    if (e.type === "own_reaction") {
      const msg = eventById(e.onMessage);
      const text = typeof msg === "object" && "text" in msg ? msg.text : "";
      return {
        role: "model",
        parts: [{
          thoughtSignature: e.modelMetadata?.thoughtSignature,
          text: `You reacted ${e.reaction} to message: ${text.slice(0, 100)}`,
        }],
      };
    }
    if (e.type === "participant_reaction") {
      const msg = eventById(e.onMessage);
      const text = typeof msg === "object" && "text" in msg ? msg.text : "";
      return {
        role: "user",
        parts: [{
          text: `${e.name} reacted ${e.reaction} to message: ${
            text.slice(0, 100)
          }`,
        }],
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
  { tools, prompt, maxIterations, onMaxIterationsReached, lightModel }:
    AgentSpec,
) => {
  const cacher = makeCache("gemini response with function calls v4");
  let c = 0;
  while (true) {
    c++;
    if (c > maxIterations) {
      onMaxIterationsReached();
      return;
    }
    const historyOuter = await getHistory();
    const history = historyOuter.map(historyEventToContent(historyOuter));
    if (empty(history) || history[0].role !== "user") {
      history.unshift({
        role: "user",
        parts: [{ text: "<conversation started>" }],
      });
    }
    const parts = await pipe(
      geminiInput,
      runSideEffectAfter(debugModelOutput.access)(
        cacher(
          // August 31st 2025, gemini returns frequent 500 errors.
          retry(
            1000,
            1,
            timeit(debugTimeElapsedMs.access, callGemini)(
              lightModel ? geminiFlashVersion : geminiProVersion,
            ),
          ),
        ),
      ),
    )(prompt, tools, sideEffect(debugHistory.access)(history));
    await handleInitialOutputEvent(parts);
    await handleFunctionCalls(tools)(parts);
    const sawFunction = parts.some(({ type }: GeminiPartOfInterest) =>
      type === "function_call"
    );
    if (
      !sawFunction &&
      !parts.some((p: GeminiPartOfInterest) => p.type === "text" && p.text)
    ) outputEvent(doNothingEvent());
    const newHistory = await getHistory();
    if (!sawFunction && last(newHistory).isOwn) return;
  }
};

const makeDebugLogger = <Input>(): SomethingInjection<
  (inp: Input) => void | Promise<void>
> => context((_) => {});

const debugHistory: SomethingInjection<
  (inp: Content[]) => void | Promise<void>
> = makeDebugLogger<Content[]>();
const debugModelOutput: SomethingInjection<
  (inp: GeminiOutput) => void | Promise<void>
> = makeDebugLogger<GeminiOutput>();
const debugTimeElapsedMs: SomethingInjection<
  (inp: number) => void | Promise<void>
> = makeDebugLogger<number>();

export const injectTimerMs = debugTimeElapsedMs.inject;
export const injectDebugHistory = debugHistory.inject;
export const injectDebugOutput = debugModelOutput.inject;

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

const handleInitialOutputEvent = each(
  pipe(
    (p: GeminiPartOfInterest) => {
      if (p.type === "text" && p.text) {
        return p.thoughtSignature
          ? ownUtteranceTurnWithMetadata(p.text, {
            type: "gemini",
            thoughtSignature: p.thoughtSignature,
          })
          : ownUtteranceTurn(p.text);
      }
      if (p.type === "function_call") {
        return p.thoughtSignature
          ? toolUseTurnWithMetadata(p.functionCall, {
            type: "gemini",
            thoughtSignature: p.thoughtSignature,
          })
          : toolUseTurn(p.functionCall);
      }
      throw new Error(`Unknown part type: ${JSON.stringify(p)}`);
    },
    outputEvent,
  ),
);

// deno-lint-ignore no-explicit-any
const handleFunctionCalls = (tools: Tool<any>[]) =>
  pipe(
    filter((p: GeminiPartOfInterest): p is GeminiFunctiontoolPart =>
      p.type === "function_call"
    ),
    map(({ functionCall }: GeminiFunctiontoolPart) => functionCall),
    each(pipe(callToResult(tools), toolResultTurn, outputEvent)),
  );
