import { context, type Injection } from "@uri/inject";
import {
  coerce,
  each,
  filter,
  last,
  map,
  pipe,
  sideEffect,
  timeit,
} from "gamla";
import type { z, ZodType } from "zod/v4";

export type MediaAttachment =
  | { kind: "inline"; mimeType: string; dataBase64: string, caption?: string }
  | { kind: "file"; mimeType: string; fileUri: string, caption?: string };

export type ToolReturn = { result: string; attachments?: MediaAttachment[] };

export type Tool<T extends ZodType> = {
  description: string;
  name: string;
  parameters: T;
  handler: (params: z.infer<T>) => Promise<string | ToolReturn>;
};

type SharedFields = { id: MessageId; timestamp: number; isOwn: boolean };

export type MessageId = string;

type ParticipantDetail = { name: string };

export type ParticipantUtterance =
  & {
    type: "participant_utterance";
    isOwn: false;
    text: string;
    attachments?: MediaAttachment[];
  }
  & ParticipantDetail
  & SharedFields;

export type OwnUtterance<ModelMetadata> = {
  isOwn: true;
  modelMetadata?: ModelMetadata;
  type: "own_utterance";
  text: string;
  attachments?: MediaAttachment[];
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

export type OwnReaction<ModelMetadata> = {
  type: "own_reaction";
  isOwn: true;
  modelMetadata?: ModelMetadata;
  reaction: string;
  onMessage: MessageId;
} & SharedFields;

type ToolUseWithMetadata<T, ModelMetadata> = {
  type: "tool_call";
  isOwn: true;
  name: string;
  modelMetadata?: ModelMetadata;
  parameters: T;
} & SharedFields;

export type ToolUse<T> = ToolUseWithMetadata<T, unknown>;

export type ToolResult = {
  type: "tool_result";
  isOwn: true;
  name: string;
  result: string;
  attachments?: MediaAttachment[];
} & SharedFields;

export type DoNothing<ModelMetadata> = {
  type: "do_nothing";
  modelMetadata?: ModelMetadata;
} & SharedFields;

export type HistoryEventWithMetadata<ModelMetadata> =
  | ParticipantUtterance
  | OwnUtterance<ModelMetadata>
  | OwnReaction<ModelMetadata>
  | ParticipantReaction
  | ToolUseWithMetadata<unknown, ModelMetadata>
  | ToolResult
  | DoNothing<ModelMetadata>;

export type HistoryEvent = HistoryEventWithMetadata<unknown>;

const idGeneration: Injection<() => string> = context((): MessageId =>
  crypto.randomUUID()
);
const timestampGeneration: Injection<() => number> = context(
  (): number => Date.now(),
);

type FunctionCall = {
  /** The unique id of the function call. If populated, the client to execute the
     `function_call` and return the response with the matching `id`. */
  id?: string;
  /** Optional. The function parameters and values in JSON object format. See [FunctionDeclaration.parameters] for parameter details. */
  args?: Record<string, unknown>;
  /** Required. The name of the function to call. Matches [FunctionDeclaration.name]. */
  name?: string;
};

const makeDebugLogger = <Input>(): Injection<
  (inp: Input) => void | Promise<void>
> => context((_) => {});

const debugHistory: Injection<
  (inp: HistoryEvent[]) => void | Promise<void>
> = makeDebugLogger<HistoryEvent[]>();
const debugTimeElapsedMs: Injection<
  (inp: number) => void | Promise<void>
> = makeDebugLogger<number>();

export const injectTimerMs = debugTimeElapsedMs.inject;
const reportTimeElapsedMs = debugTimeElapsedMs.access;
export const injectDebugHistory = debugHistory.inject;
const reportHistoryForDebug = debugHistory.access;

const modelOutput: Injection<(event: HistoryEvent) => Promise<void>> = context(
  (_event: HistoryEvent): Promise<void> => {
    throw new Error("output function not injected");
  },
);

const outputEvent = modelOutput.access;
export const injectOutputEvent = modelOutput.inject;

const historyInjection: Injection<() => Promise<HistoryEvent[]>> = context(
  (): Promise<HistoryEvent[]> => {
    throw new Error("History not injected");
  },
);

const getHistory = historyInjection.access;
export const injectAccessHistory = historyInjection.inject;

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
  (actions: Tool<any>[]) => async <T extends ZodType>(fc: FunctionCall) => {
    const { name, args } = fc;
    const action: Tool<T> | undefined = actions.find(({ name: n }) =>
      n === name
    );
    if (!name) throw new Error("Function call name is missing");
    if (!action) return { name, result: `Function ${name} not found` };
    const { handler, parameters } = action;
    const parseResult = parseWithCatch(parameters, args);
    if (!parseResult.ok) {
      return {
        name,
        result: `Invalid arguments: ${JSON.stringify(parseResult.error)}`,
      };
    }
    const out = await handler(parseResult.result);
    return typeof out === "string"
      ? { name, result: out }
      : { name, result: out.result, attachments: out.attachments };
  };

export const toolUseTurnWithMetadata = <Metadata>(
  { name, args }: FunctionCall,
  modelMetadata: Metadata | undefined,
): HistoryEventWithMetadata<Metadata> => ({
  type: "tool_call",
  ...sharedFields(),
  isOwn: true,
  timestamp: timestampGeneration.access(),
  name: coerce(name),
  parameters: args,
  modelMetadata,
});

export const toolUseTurn = <Metadata>(
  { name, args }: FunctionCall,
): HistoryEventWithMetadata<Metadata> =>
  toolUseTurnWithMetadata({ name, args }, undefined);

export const participantUtteranceTurn = (
  { name, text, attachments }: {
    name: string;
    text: string;
    attachments?: MediaAttachment[];
  },
): HistoryEvent => ({
  type: "participant_utterance",
  isOwn: false,
  name: coerce(name),
  text,
  attachments,
  ...sharedFields(),
});

export const ownUtteranceTurnWithMetadata = <Metadata>(
  text: string,
  modelMetadata: Metadata | undefined,
  attachments?: MediaAttachment[],
): HistoryEventWithMetadata<Metadata> => ({
  type: "own_utterance",
  isOwn: true,
  modelMetadata,
  text,
  attachments,
  ...sharedFields(),
});

export const ownUtteranceTurn = <Metadata>(
  text: string,
  attachments?: MediaAttachment[],
): HistoryEventWithMetadata<Metadata> =>
  ownUtteranceTurnWithMetadata(text, undefined, attachments);

const sharedFields = () => ({
  id: idGeneration.access(),
  timestamp: timestampGeneration.access(),
});

export const toolResultTurn = (
  { name, result, attachments }: {
    name: string;
    result: string;
    attachments?: MediaAttachment[];
  },
): HistoryEvent => ({
  ...sharedFields(),
  type: "tool_result",
  isOwn: true,
  name,
  result,
  attachments,
});

export const doNothingEvent = <Metadata>(
  modelMetadata?: Metadata,
): HistoryEventWithMetadata<
  Metadata
> => ({
  type: "do_nothing",
  isOwn: true,
  modelMetadata,
  ...sharedFields(),
});

export const overrideTime = timestampGeneration.inject;
export const overrideIdGenerator = idGeneration.inject;
export const generateId = idGeneration.access;

// deno-lint-ignore no-explicit-any
const handleFunctionCalls = (tools: Tool<any>[]) =>
  pipe(
    // deno-lint-ignore no-explicit-any
    filter((p: HistoryEvent): p is ToolUse<any> => p.type === "tool_call"),
    // deno-lint-ignore no-explicit-any
    map((t: ToolUse<any>): FunctionCall => ({
      name: t.name,
      args: t.parameters,
    })),
    each(pipe(callToResult(tools), toolResultTurn, outputEvent)),
  );

export type AgentSpec = {
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[];
  prompt: string;
  maxIterations: number;
  // deno-lint-ignore no-explicit-any
  onMaxIterationsReached: () => any;
  lightModel?: boolean;
  provider?: "gemini";
  imageGen?: boolean;
};

export const runAbstractAgent = async (
  { maxIterations, tools, onMaxIterationsReached }: AgentSpec,
  callModel: (history: HistoryEvent[]) => Promise<HistoryEvent[]>,
) => {
  let c = 0;
  while (true) {
    c++;
    if (c > maxIterations) {
      onMaxIterationsReached();
      return;
    }
    const output = await pipe(
      getHistory,
      sideEffect(reportHistoryForDebug),
      timeit(reportTimeElapsedMs, callModel),
      sideEffect(each(outputEvent)),
    )();
    await handleFunctionCalls(tools)(output);
    if (
      !(output.some((ev: HistoryEvent) => ev.type === "tool_call")) &&
      last(await getHistory()).isOwn
    ) return;
  }
};

// --- Token estimation -------------------------------------------------------
// A lightweight, overridable heuristic for estimating the token cost of
// processing a single HistoryEvent. This intentionally avoids binding to any
// provider-specific tokenizer (so the library stays dependency‑light) while
// still giving callers a way to reason about budget / pruning.
//
// Rough heuristic: ~1 token per ~4 characters (English) with a 30% buffer.
// For base64 media we count each 4 chars as 1 token (very rough) – callers
// relying on precise billing should override.
const approxTextTokens = (text: string | undefined): number => {
  if (!text) return 0;
  return Math.max(1, Math.ceil((text.length / 4) * 1.3));
};

const approxJsonTokens = (obj: unknown): number => {
  try {
    return approxTextTokens(JSON.stringify(obj));
  } catch (_) {
    return 10; // fallback small constant
  }
};

const attachmentTokens = (
  attachments: MediaAttachment[] | undefined,
): number => {
  if (!attachments || attachments.length === 0) return 0;
  return attachments.reduce((sum, a) => {
    if (a.kind === "inline") {
      // base64 length / 4 (very rough) with small buffer
      return sum + Math.ceil(a.dataBase64.length / 4 * 1.1);
    }
    // file references assumed minimal (URI + metadata)
    return sum + approxTextTokens(a.fileUri) + approxTextTokens(a.mimeType);
  }, 0);
};

const assertNever = (x: never): never => {
  throw new Error(
    `Unhandled HistoryEvent variant in token estimator: ${JSON.stringify(x)}`,
  );
};

export const estimateTokens = (e: HistoryEvent): number => {
  if (e.type === "participant_utterance") {
    return approxTextTokens(e.name) + approxTextTokens(e.text) +
      attachmentTokens(e.attachments) + 2;
  }
  if (e.type === "own_utterance") {
    return approxTextTokens(e.text) +
      attachmentTokens(e.attachments) + 2;
  }
  if (e.type === "tool_call") {
    return approxTextTokens(e.name) + approxJsonTokens(e.parameters) + 4;
  }
  if (e.type === "tool_result") {
    return approxTextTokens(e.name) + approxTextTokens(e.result) +
      attachmentTokens(e.attachments) + 4;
  }
  if (e.type === "participant_reaction") {
    return approxTextTokens(e.name) + approxTextTokens(e.reaction) + 2;
  }
  if (e.type === "own_reaction") {
    return approxTextTokens(e.reaction) + 2;
  }
  if (e.type === "do_nothing") {
    return 1;
  }
  return assertNever(e);
};
