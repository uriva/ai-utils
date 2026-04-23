import { context, type Injection } from "@uri/inject";
import { coerce, each, empty, filter, last, nonempty, timeit } from "gamla";
import { z, type ZodType } from "zod/v4";
import { zodToTypingString } from "./toolTyping.ts";
import {
  hasInternalSentTimestampSuffix,
  stripInternalSentTimestampSuffix,
} from "./internalMessageMetadata.ts";
import { isEmojiFlood } from "./utils.ts";
export type MediaAttachment =
  | { kind: "inline"; mimeType: string; dataBase64: string; caption?: string }
  | { kind: "file"; mimeType: string; fileUri: string; caption?: string };

const mediaAttachmentSchema: z.ZodType<MediaAttachment> = z.union([
  z.object({
    kind: z.literal("inline"),
    mimeType: z.string(),
    dataBase64: z.string(),
    caption: z.string().optional(),
  }),
  z.object({
    kind: z.literal("file"),
    mimeType: z.string(),
    fileUri: z.string(),
    caption: z.string().optional(),
  }),
]);

export type ToolReturn = { result: string; attachments?: MediaAttachment[] };

const maxToolOutputChars = 20_000;

const truncateToolOutput = (s: string): string =>
  s.length <= maxToolOutputChars
    ? s
    : s.slice(0, maxToolOutputChars) + "\n[...output truncated]";

const toolReturnSchema: z.ZodType<string | ToolReturn> = z.union([
  z.string(),
  z.object({
    result: z.string(),
    attachments: z.array(mediaAttachmentSchema).optional(),
  }),
]);

type ToolBase<T extends ZodType> = {
  description: string;
  name: string;
  parameters: T;
};

export type Tool<T extends ZodType> = ToolBase<T> & {
  handler: (
    params: z.infer<T>,
    toolCallId: string,
  ) => Promise<string | ToolReturn | void>;
};

/** @deprecated Use Tool directly — deferred vs regular is determined by handler return value */
export type RegularTool<T extends ZodType> = Tool<T>;
/** @deprecated Use Tool directly — deferred vs regular is determined by handler return value */
export type DeferredTool<T extends ZodType> = Tool<T>;

export type Skill = {
  name: string;
  description: string;
  instructions: string;
  // deno-lint-ignore no-explicit-any
  tools: RegularTool<any>[];
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
  truncated?: boolean;
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

export type ParticipantEditMessage =
  & Omit<ParticipantUtterance, "type">
  & { type: "participant_edit_message"; onMessage: MessageId };

export type OwnEditMessage<ModelMetadata> =
  & Omit<OwnUtterance<ModelMetadata>, "type">
  & { type: "own_edit_message"; onMessage: MessageId };

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
  toolCallId?: string;
  result: string;
  attachments?: MediaAttachment[];
} & SharedFields;

export type OwnThought<ModelMetadata> = {
  type: "own_thought";
  isOwn: true;
  modelMetadata?: ModelMetadata;
  text: string;
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
  | ParticipantEditMessage
  | OwnEditMessage<ModelMetadata>
  | ToolUseWithMetadata<unknown, ModelMetadata>
  | ToolResult
  | OwnThought<ModelMetadata>
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

const toolNotFoundInjection: Injection<
  (toolName: string) => void | Promise<void>
> = makeDebugLogger<string>();

export const injectToolNotFound = toolNotFoundInjection.inject;
const reportToolNotFound = toolNotFoundInjection.access;

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
export const accessOutputEvent = modelOutput.access;

const streamChunkInjection: Injection<(chunk: string) => Promise<void> | void> =
  context((_chunk: string) => {});
export const injectStreamChunk = streamChunkInjection.inject;
export const accessStreamChunk = streamChunkInjection.access;
export const getStreamChunk = streamChunkInjection.getStore;

const streamThinkingChunkInjection: Injection<
  (chunk: string) => Promise<void> | void
> = context((_chunk: string) => {});
export const injectStreamThinkingChunk = streamThinkingChunkInjection.inject;
export const getStreamThinkingChunk = streamThinkingChunkInjection.getStore;

const abortInjection: Injection<() => Promise<boolean>> = context(
  () => Promise.resolve(false),
);
export const injectShouldAbort = abortInjection.inject;
const shouldAbort = abortInjection.access;

const historyInjection: Injection<() => Promise<HistoryEvent[]>> = context(
  (): Promise<HistoryEvent[]> => {
    throw new Error("History not injected");
  },
);

const getHistory = historyInjection.access;
export const injectAccessHistory = historyInjection.inject;
export const accessHistory = historyInjection.access;

export type CallModel = (events: HistoryEvent[]) => Promise<HistoryEvent[]>;

const callModelInjection: Injection<CallModel> = context(
  (_events: HistoryEvent[]): Promise<HistoryEvent[]> => {
    throw new Error(
      "no callModel injected; runAgent usually wires this from the provider",
    );
  },
);

export const injectCallModel = callModelInjection.inject;
export const accessCallModel = callModelInjection.access;

// Wraps the resolved CallModel. Used e.g. by test_helpers to add rmmbr
// caching around whatever provider caller runAgent picks. The wrapper gets
// the provider name so it can key caches per-provider.
export type Provider = "google" | "moonshot" | "anthropic" | undefined;

export type CallModelWrapper = (args: {
  provider: Provider;
  inner: CallModel;
}) => CallModel;

const callModelWrapperInjection: Injection<CallModelWrapper> = context(
  ({ inner }) => inner,
);

export const injectCallModelWrapper = callModelWrapperInjection.inject;
export const accessCallModelWrapper = callModelWrapperInjection.access;

const parseWithCatch = <T extends ZodType>(
  parameters: T,
  // deno-lint-ignore no-explicit-any
  args: any,
): { ok: false; error: Error } | { ok: true; result: z.infer<T> } => {
  try {
    const p = "strict" in parameters &&
        typeof (parameters as unknown as { strict?: () => ZodType }).strict ===
          "function"
      ? (parameters as unknown as { strict: () => ZodType }).strict()
      : parameters;
    return { ok: true, result: p.parse(args) as z.infer<T> };
  } catch (error) {
    return { ok: false, error: error as Error };
  }
};

export const callToResult =
  // deno-lint-ignore no-explicit-any
  (actions: Tool<any>[]) =>
  async <T extends ZodType>(fc: FunctionCall): Promise<
    | {
      toolCallId: string | undefined;
      result: string;
      attachments?: MediaAttachment[];
    }
    | undefined
  > => {
    const { name, args, id } = fc;
    const toolCallId = id;
    if (!name) throw new Error("Function call name is missing");
    const directMatch: Tool<T> | undefined = actions.find((
      { name: n },
    ) => n === name);
    const [action, effectiveArgs] = directMatch
      ? [directMatch, args]
      : name.includes("/")
      ? [
        actions.find(({ name: n }) => n === runCommandToolName) as
          | Tool<T>
          | undefined,
        { command: name, params: args },
      ]
      : [undefined, args];
    if (!action) {
      reportToolNotFound(name);
      return {
        toolCallId,
        result:
          `Tool not found. You may have misspelled it, or you need to call learn_skill to discover available tools. If you see this tool in your history, it may also be that this tool is no longer available or has changed names.`,
      };
    }
    const { handler, parameters } = action;
    const parseResult = parseWithCatch(parameters, effectiveArgs);
    if (!parseResult.ok) {
      return {
        toolCallId,
        result: `Invalid arguments: ${
          parseResult.error instanceof z.ZodError
            ? parseResult.error.issues
              .map((i) =>
                `${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`
              )
              .join(", ")
            : parseResult.error.message
        }`,
      };
    }
    const out = await handler(parseResult.result, toolCallId ?? "");
    if (out === undefined) return undefined;
    const parsed = parseWithCatch(toolReturnSchema, out);
    if (!parsed.ok) {
      throw new Error(
        `Tool "${name}" handler returned invalid value (args: ${
          JSON.stringify(args)
        }): ${
          parsed.error instanceof z.ZodError
            ? parsed.error.issues.map((i) =>
              `${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`
            ).join(", ")
            : parsed.error.message
        }`,
      );
    }
    const validated = parsed.result;
    return typeof validated === "string"
      ? { toolCallId, result: truncateToolOutput(validated) }
      : {
        toolCallId,
        result: truncateToolOutput(validated.result),
        attachments: validated.attachments,
      };
  };

export const toolUseTurnWithMetadata = <Metadata>(
  { name, args }: FunctionCall,
  modelMetadata: Metadata | undefined,
): HistoryEventWithMetadata<Metadata> => ({
  type: "tool_call",
  ...sharedFields(),
  isOwn: true,
  name: coerce(name),
  parameters: args,
  modelMetadata,
});

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

export const ownThoughtTurn = <Metadata>(
  text: string,
): HistoryEventWithMetadata<Metadata> => ({
  type: "own_thought",
  isOwn: true,
  text,
  ...sharedFields(),
});

export const ownThoughtTurnWithMetadata = <Metadata>(
  text: string,
  modelMetadata: Metadata | undefined,
): HistoryEventWithMetadata<Metadata> => ({
  type: "own_thought",
  isOwn: true,
  modelMetadata,
  text,
  ...sharedFields(),
});

const sharedFields = () => ({
  id: idGeneration.access(),
  timestamp: timestampGeneration.access(),
});

export const toolResultTurn = (
  { result, attachments, toolCallId }: {
    result: string;
    attachments?: MediaAttachment[];
    toolCallId?: string;
  },
): HistoryEvent => ({
  ...sharedFields(),
  type: "tool_result",
  isOwn: true,
  result,
  attachments,
  toolCallId,
});

export const participantEditMessageTurn = (
  { name, text, onMessage, attachments }: {
    name: string;
    text: string;
    onMessage: MessageId;
    attachments?: MediaAttachment[];
  },
): HistoryEvent => ({
  type: "participant_edit_message",
  isOwn: false,
  name,
  text,
  onMessage,
  attachments,
  ...sharedFields(),
});

export const ownEditMessageTurnWithMetadata = <Metadata>(
  { text, onMessage, modelMetadata, attachments }: {
    text: string;
    onMessage: MessageId;
    modelMetadata?: Metadata;
    attachments?: MediaAttachment[];
  },
): HistoryEventWithMetadata<Metadata> => ({
  type: "own_edit_message",
  isOwn: true,
  modelMetadata,
  text,
  onMessage,
  attachments,
  ...sharedFields(),
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

export const modelOutputLeaksInternalSentTimestamp = (
  output: HistoryEvent[],
): boolean =>
  output.some((event) =>
    (event.type === "own_utterance" || event.type === "own_edit_message") &&
    hasInternalSentTimestampSuffix(event.text)
  );

const sanitizeInternalSentTimestampLeak = (
  output: HistoryEvent[],
): HistoryEvent[] =>
  output.map((event) =>
    event.type === "own_utterance"
      ? { ...event, text: stripInternalSentTimestampSuffix(event.text) }
      : event.type === "own_edit_message"
      ? { ...event, text: stripInternalSentTimestampSuffix(event.text) }
      : event
  );

const internalThoughtPattern =
  /^\[Internal thought, visible only to you: ([\s\S]*?)\]$/;

const reclassifyLeakedThoughts = (output: HistoryEvent[]): HistoryEvent[] =>
  output.map((event) => {
    if (event.type !== "own_utterance") return event;
    const match = stripInternalSentTimestampSuffix(event.text).match(
      internalThoughtPattern,
    );
    return match
      ? { ...event, type: "own_thought" as const, text: match[1] }
      : event;
  });

const noResponsePattern = /^\[no response\]\s*$/i;

const isNoResponseUtterance = (event: HistoryEvent) =>
  (event.type === "own_utterance" || event.type === "own_edit_message") &&
  noResponsePattern.test(event.text.trim());

const reclassifyNoResponse = (output: HistoryEvent[]): HistoryEvent[] =>
  output.map((event) =>
    isNoResponseUtterance(event)
      ? doNothingEvent(
        (event as Extract<HistoryEvent, { type: "own_utterance" }>)
          .modelMetadata ??
          undefined,
      )
      : event
  );

const isEmptyUtterance = (event: HistoryEvent) => {
  if (event.type !== "own_utterance" && event.type !== "own_edit_message") {
    return false;
  }
  return !event.text.trim() && empty(event.attachments ?? []);
};

const reclassifyEmptyUtterances = (output: HistoryEvent[]): HistoryEvent[] =>
  output.filter((event) => !isEmptyUtterance(event));

const participantNamesFromHistory = (history: HistoryEvent[]): Set<string> =>
  new Set(
    history
      .filter((e): e is ParticipantUtterance | ParticipantEditMessage =>
        e.type === "participant_utterance" ||
        e.type === "participant_edit_message"
      )
      .map((e) => e.name),
  );

const fabricatedUserMessagePattern = (participantNames: Set<string>) => {
  if (participantNames.size === 0) return null;
  const escaped = [...participantNames].map((n) =>
    n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );
  return new RegExp(`^(${escaped.join("|")}):\\s`, "m");
};

export const stripFabricatedUserMessages = (
  participantNames: Set<string>,
  output: HistoryEvent[],
): HistoryEvent[] => {
  const pattern = fabricatedUserMessagePattern(participantNames);
  if (!pattern) return output;
  return output.map((event) => {
    if (event.type !== "own_utterance") return event;
    const text = stripInternalSentTimestampSuffix(event.text);
    if (!pattern.test(text)) return event;
    console.warn(
      "[fabrication-guard] model fabricated user message in own_utterance",
      { text: text.slice(0, 200) },
    );
    const lines = text.split("\n");
    const clean = lines.filter((line) => !pattern.test(line)).join("\n").trim();
    return clean.length > 0
      ? { ...event, text: clean }
      : { ...event, type: "own_thought" as const, text };
  });
};

export const maxUtteranceChars = 4000;

const findSplitIndex = (text: string): number => {
  const window = text.slice(0, maxUtteranceChars);
  const minAccept = Math.floor(maxUtteranceChars / 2);
  const paragraphIdx = window.lastIndexOf("\n\n");
  if (paragraphIdx >= minAccept) return paragraphIdx + 2;
  const newlineIdx = window.lastIndexOf("\n");
  if (newlineIdx >= minAccept) return newlineIdx + 1;
  const sentenceMatch = [...window.matchAll(/[.!?](?:\s|$)/g)].at(-1);
  if (sentenceMatch && sentenceMatch.index >= minAccept) {
    return sentenceMatch.index + sentenceMatch[0].length;
  }
  const whitespaceIdx = window.search(/\s\S*$/);
  if (whitespaceIdx >= minAccept) return whitespaceIdx + 1;
  return maxUtteranceChars;
};

const splitLongUtteranceText = (text: string): string[] => {
  if (text.length <= maxUtteranceChars) return [text];
  const idx = findSplitIndex(text);
  const head = text.slice(0, idx).trimEnd();
  const tail = text.slice(idx).trimStart();
  return tail === "" ? [head] : [head, ...splitLongUtteranceText(tail)];
};

const splitOversizedUtterance = (
  event: Extract<HistoryEvent, { type: "own_utterance" }>,
): HistoryEvent[] =>
  splitLongUtteranceText(event.text).map((chunk, i) => ({
    ...event,
    text: chunk,
    id: i === 0 ? event.id : generateId(),
    timestamp: event.timestamp + i,
  }));

const splitOversizedUtterances = (output: HistoryEvent[]): HistoryEvent[] =>
  output.flatMap((event) =>
    event.type === "own_utterance" && event.text.length > maxUtteranceChars
      ? splitOversizedUtterance(event)
      : [event]
  );

export const sanitizeModelOutput = (
  history: HistoryEvent[],
  output: HistoryEvent[],
): { emit: HistoryEvent[]; internal: HistoryEvent[] } => {
  const sanitized = modelOutputLeaksInternalSentTimestamp(output)
    ? sanitizeInternalSentTimestampLeak(output)
    : output;
  const withoutFabrications = stripFabricatedUserMessages(
    participantNamesFromHistory(history),
    sanitized,
  );
  const withoutNoResponse = reclassifyNoResponse(withoutFabrications);
  const withoutEmpty = reclassifyEmptyUtterances(withoutNoResponse);
  const reclassified = reclassifyLeakedThoughts(withoutEmpty);
  const safe = splitOversizedUtterances(reclassified);
  return { emit: safe, internal: safe };
};

const hasToolCall = (history: HistoryEvent[], toolCallId: string): boolean =>
  history.some((event) =>
    event.type === "tool_call" && event.id === toolCallId
  );

const toolResultsByCallId = (
  history: HistoryEvent[],
): Map<string, ToolResult[]> =>
  history.reduce((acc, event) => {
    if (event.type !== "tool_result" || !event.toolCallId) return acc;
    const existing = acc.get(event.toolCallId) ?? [];
    return acc.set(event.toolCallId, [...existing, event]);
  }, new Map<string, ToolResult[]>());

export const normalizeHistoryForModel = (
  history: HistoryEvent[],
): HistoryEvent[] => {
  const groupedResults = toolResultsByCallId(history);
  const consumedResultIds = new Set<string>();

  const interleaved = history.reduce<HistoryEvent[]>((acc, event) => {
    if (event.type === "tool_result") return acc;
    if (event.type !== "tool_call") return [...acc, event];
    const matchedResults = (groupedResults.get(event.id) ?? [])
      .filter((result) => !consumedResultIds.has(result.id));
    matchedResults.forEach((result) => consumedResultIds.add(result.id));
    if (nonempty(matchedResults)) {
      return [...acc, event, ...matchedResults];
    }
    const syntheticResult: ToolResult = {
      type: "tool_result",
      isOwn: true,
      id: `${event.id}-synthetic-result`,
      timestamp: event.timestamp,
      result: "[Tool result unavailable]",
      toolCallId: event.id,
    };
    return [...acc, event, syntheticResult];
  }, []);

  const orphanedResults = history.filter((event): event is ToolResult => {
    if (event.type !== "tool_result") return false;
    if (consumedResultIds.has(event.id)) return false;
    if (!event.toolCallId) return true;
    return !hasToolCall(history, event.toolCallId);
  });

  return [...interleaved, ...orphanedResults];
};

export const handleFunctionCalls =
  // deno-lint-ignore no-explicit-any
  (tools: Tool<any>[], onToolResult?: (event: HistoryEvent) => void) =>
  async (output: HistoryEvent[]): Promise<boolean> => {
    // deno-lint-ignore no-explicit-any
    const toolCalls = filter((p: HistoryEvent): p is ToolUse<any> =>
      p.type === "tool_call"
    )(output);
    let hadDeferred = false;
    await each(async (t: ToolUse<Record<string, unknown>>) => {
      if (t.name === doNothingToolName) {
        hadDeferred = true;
        await outputEvent(doNothingEvent(undefined));
        return;
      }
      const fc: FunctionCall = { name: t.name, args: t.parameters, id: t.id };
      const callResult = await callToResult(tools)(fc);
      if (callResult === undefined) {
        hadDeferred = true;
        return;
      }
      const result = toolResultTurn(callResult);
      await outputEvent(result);
      onToolResult?.(result);
    })(toolCalls);
    return hadDeferred;
  };

export const runCommandToolName = "run_command";
export const learnSkillToolName = "learn_skill";

export const doNothingToolName = "do_nothing";

export const doNothingTool: Tool<
  z.ZodObject<{ reason: z.ZodOptional<z.ZodString> }>
> = {
  name: doNothingToolName,
  description:
    "Call this tool when you have nothing to say and should not respond. Use this instead of writing an empty message, HTML comment, or any placeholder text.",
  parameters: z.object({ reason: z.string().optional() }),
  handler: () => Promise.resolve(""),
};

export const tool = <ParametersSchema extends z.ZodObject<z.ZodRawShape>>(
  tool: Tool<ParametersSchema>,
): Tool<ParametersSchema> => ({
  ...tool,
  handler: (
    params: z.infer<ParametersSchema>,
    toolCallId: string,
  ): ReturnType<typeof tool.handler> => tool.handler(params, toolCallId),
});

// deno-lint-ignore no-explicit-any
export const createSkillTools = (skills: Skill[]): RegularTool<any>[] => {
  const skillMap = Object.fromEntries(skills.map((s) => [s.name, s]));
  const toolMap = Object.fromEntries(
    skills.flatMap((skill) =>
      skill.tools.map((tool) => [`${skill.name}/${tool.name}`, tool])
    ),
  );
  const skillNames = skills.map((s) => s.name).join(", ");
  return [
    tool({
      name: runCommandToolName,
      description:
        "Execute a tool from a specific skill. Format: skillName/toolName",
      parameters: z.object({
        command: z.string().describe(
          "The command in format skillName/toolName",
        ),
        params: z.any().describe("The parameters for the tool"),
      }),
      handler: async ({ command, params }, toolCallId) => {
        const lastSlash = command.lastIndexOf("/");
        if (lastSlash === -1) {
          return `Invalid command format. Expected "skillName/toolName", got "${command}". Available skills: ${skillNames}`;
        }
        const skillName = command.slice(0, lastSlash);
        const toolName = command.slice(lastSlash + 1);
        if (!skillMap[skillName]) {
          return `Skill "${skillName}" not found. Available skills: ${skillNames}`;
        }
        const fullToolName = `${skillName}/${toolName}`;
        const tool = toolMap[fullToolName];
        if (!tool) {
          return `Tool "${toolName}" not found in skill "${skillName}". Please call ${learnSkillToolName}.`;
        }
        const parseResult = parseWithCatch(tool.parameters, params);
        if (!parseResult.ok) {
          return `Invalid parameters for ${fullToolName}: ${
            parseResult.error instanceof z.ZodError
              ? parseResult.error.issues.map((i) =>
                `${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`
              ).join(", ")
              : parseResult.error.message
          }`;
        }
        return await tool.handler(parseResult.result, toolCallId);
      },
    }),
    tool({
      name: learnSkillToolName,
      description:
        "Get detailed information about a skill including its instructions and available tools",
      parameters: z.object({
        skillName: z.string().describe("The name of the skill to learn about"),
      }),
      handler: ({ skillName }) => {
        const skill = skillMap[skillName];
        if (!skill) {
          return Promise.resolve(
            `Skill "${skillName}" not found. Available skills: ${skillNames}`,
          );
        }
        return Promise.resolve(JSON.stringify(
          {
            name: skill.name,
            description: skill.description,
            instructions: skill.instructions,
            tools: skill.tools.map((tool) => ({
              name: tool.name,
              description: tool.description,
              parameters: zodToTypingString(tool.parameters),
            })),
          },
          null,
          2,
        ));
      },
    }),
  ];
};

export type AgentSpec = {
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[];
  skills?: Skill[];
  prompt: string;
  onOutputEvent?: (event: HistoryEvent) => Promise<void>;
  onStreamChunk?: (chunk: string) => Promise<void> | void;
  onStreamThinkingChunk?: (chunk: string) => Promise<void> | void;
  maxIterations: number;
  // deno-lint-ignore no-explicit-any
  onMaxIterationsReached: () => any;
  lightModel?: boolean;
  disableStreaming?: boolean;
  provider?: "google" | "moonshot" | "anthropic";
  imageGen?: boolean;
  rewriteHistory: (replacements: Record<string, HistoryEvent>) => Promise<void>;
  timezoneIANA: string;
  maxOutputTokens?: number;
  transport?: {
    kind: "audio";
    endpoint: import("./duplex.ts").DuplexEndpoint;
    voiceName: string;
    participantName: string;
  };
};

const hasEmojiFlood = (events: HistoryEvent[]) =>
  events.some((e) => e.type === "own_utterance" && isEmojiFlood(e.text));

const maxEmojiFloodRetries = 3;

const maxTruncationRetries = 2;

const findTruncatedUtterance = (events: HistoryEvent[]) =>
  events.find(
    (e): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
      e.type === "own_utterance" && e.truncated === true,
  );

const truncationCorrectionText = (partialText: string) => {
  const tail = partialText.slice(-400);
  return `Your previous response hit the output token budget and was cut off mid-way. You had written: "${tail}". Restart the response from the beginning — keep it significantly more concise and keep any internal reasoning brief so the full answer fits within the budget.`;
};

const stripTruncatedFlag = (events: HistoryEvent[]): HistoryEvent[] =>
  events.map((e) =>
    e.type === "own_utterance" && e.truncated
      ? { ...e, truncated: undefined }
      : e
  );

export const runAbstractAgent = async (
  { maxIterations, tools, skills, onMaxIterationsReached, prompt: _prompt }:
    AgentSpec,
  callModel: (history: HistoryEvent[]) => Promise<HistoryEvent[]>,
) => {
  const allTools = skills && skills.length > 0
    ? [...tools, ...createSkillTools(skills)]
    : tools;
  let c = 0;
  let emojiFloodRetries = 0;
  let truncationRetries = 0;
  let ephemeralHistory: HistoryEvent[] = [];
  while (true) {
    if (await shouldAbort()) return;
    c++;
    if (c > maxIterations) {
      onMaxIterationsReached();
      return;
    }
    const history = await getHistory();
    const effectiveHistory = [...history, ...ephemeralHistory];
    const normalizedHistory = normalizeHistoryForModel(effectiveHistory);
    await reportHistoryForDebug(normalizedHistory);
    const rawModelResponse = await timeit(reportTimeElapsedMs, callModel)(
      normalizedHistory,
    );
    if (hasEmojiFlood(rawModelResponse)) {
      emojiFloodRetries++;
      console.warn(
        `[emoji-flood] detected emoji flood in model response (attempt ${emojiFloodRetries}/${maxEmojiFloodRetries})`,
      );
      if (emojiFloodRetries >= maxEmojiFloodRetries) {
        throw new Error("model keeps producing emoji flood responses");
      }
      continue;
    }
    const truncated = findTruncatedUtterance(rawModelResponse);
    if (truncated && truncationRetries < maxTruncationRetries) {
      truncationRetries++;
      console.warn(
        `[max-tokens] model response truncated (attempt ${truncationRetries}/${maxTruncationRetries}); retrying with correctional thought`,
      );
      ephemeralHistory = [
        ...ephemeralHistory,
        ownThoughtTurn(truncationCorrectionText(truncated.text)),
      ];
      continue;
    }
    const modelResponse = stripTruncatedFlag(rawModelResponse);
    const { emit, internal } = sanitizeModelOutput(
      normalizedHistory,
      modelResponse,
    );

    // Process what needs to be emitted
    if (emit.length > 0) {
      await each(outputEvent)(emit);
      const hadDeferred = await handleFunctionCalls(allTools)(emit);
      if (hadDeferred) return;

      // We actually yielded things to the outside world, reset ephemeral history
      ephemeralHistory = [];

      const updatedHistory = await getHistory();
      if (
        !(emit.some((ev: HistoryEvent) => ev.type === "tool_call")) &&
        nonempty(updatedHistory) &&
        last(updatedHistory).isOwn &&
        !emit.every((ev: HistoryEvent) => ev.type === "own_thought")
      ) return;
    } else {
      // Nothing was emitted to the outside world, accumulate the internal state (e.g., thoughts)
      ephemeralHistory = [...ephemeralHistory, ...internal];
    }
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
  if (
    e.type === "participant_utterance" || e.type === "participant_edit_message"
  ) {
    return approxTextTokens(e.name) + approxTextTokens(e.text) +
      attachmentTokens(e.attachments) + 2;
  }
  if (e.type === "own_utterance" || e.type === "own_edit_message") {
    return approxTextTokens(e.text) +
      attachmentTokens(e.attachments) + 2;
  }
  if (e.type === "tool_call") {
    return approxTextTokens(e.name) + approxJsonTokens(e.parameters) + 4;
  }
  if (e.type === "tool_result") {
    return approxTextTokens(e.result) +
      attachmentTokens(e.attachments) + 4;
  }
  if (e.type === "own_thought") {
    return approxTextTokens(e.text) + 2;
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
