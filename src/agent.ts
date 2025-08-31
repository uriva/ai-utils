import { context } from "context-inject";
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
import type { SomethingInjection } from "./utils.ts";

export type Tool<T extends ZodType> = {
  description: string;
  name: string;
  parameters: T;
  handler: (params: z.infer<T>) => Promise<string>;
};

type SharedFields = { id: MessageId; timestamp: number; isOwn: boolean };

export type MessageId = string;

type ParticipantDetail = { name: string };

export type ParticipantUtterance =
  & {
    type: "participant_utterance";
    isOwn: false;
    text: string;
  }
  & ParticipantDetail
  & SharedFields;

export type OwnUtterance<ModelMetadata> = {
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
} & SharedFields;

export type DoNothing = { type: "do_nothing" } & SharedFields;

export type HistoryEventWithMetadata<ModelMetadata> =
  | ParticipantUtterance
  | OwnUtterance<ModelMetadata>
  | OwnReaction<ModelMetadata>
  | ParticipantReaction
  | ToolUseWithMetadata<unknown, ModelMetadata>
  | ToolResult
  | DoNothing;

export type HistoryEvent = HistoryEventWithMetadata<unknown>;

const idGeneration: SomethingInjection<() => string> = context((): MessageId =>
  crypto.randomUUID()
);
const timestampGeneration: SomethingInjection<() => number> = context(
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

const makeDebugLogger = <Input>(): SomethingInjection<
  (inp: Input) => void | Promise<void>
> => context((_) => {});

const debugHistory: SomethingInjection<
  (inp: HistoryEvent[]) => void | Promise<void>
> = makeDebugLogger<HistoryEvent[]>();
const debugTimeElapsedMs: SomethingInjection<
  (inp: number) => void | Promise<void>
> = makeDebugLogger<number>();

export const injectTimerMs = debugTimeElapsedMs.inject;
const reportTimeElapsedMs = debugTimeElapsedMs.access;
export const injectDebugHistory = debugHistory.inject;
const reportHistoryForDebug = debugHistory.access;

const modelOutput: SomethingInjection<(event: HistoryEvent) => Promise<void>> =
  context((_event: HistoryEvent): Promise<void> => {
    throw new Error("output function not injected");
  });

const outputEvent = modelOutput.access;
export const injectOutputEvent = modelOutput.inject;

const historyInjection: SomethingInjection<() => Promise<HistoryEvent[]>> =
  context((): Promise<HistoryEvent[]> => {
    throw new Error("History not injected");
  });

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
  { name, text }: { name: string; text: string },
): HistoryEvent => ({
  type: "participant_utterance",
  isOwn: false,
  name: coerce(name),
  text,
  ...sharedFields(),
});

export const ownUtteranceTurnWithMetadata = <Metadata>(
  text: string,
  modelMetadata: Metadata | undefined,
): HistoryEventWithMetadata<Metadata> => ({
  type: "own_utterance",
  isOwn: true,
  modelMetadata,
  text,
  ...sharedFields(),
});

export const ownUtteranceTurn = <Metadata>(
  text: string,
): HistoryEventWithMetadata<Metadata> =>
  ownUtteranceTurnWithMetadata(text, undefined);

const sharedFields = () => ({
  id: idGeneration.access(),
  timestamp: timestampGeneration.access(),
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

export const doNothingEvent = <Metadata>(): HistoryEventWithMetadata<
  Metadata
> => ({
  type: "do_nothing",
  isOwn: true,
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
      !(output.some(({ type }) => type === "tool_call")) &&
      last(await getHistory()).isOwn
    ) return;
  }
};
