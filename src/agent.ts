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
import { z, type ZodType } from "zod/v4";
import { zodToGeminiParameters } from "./gemini.ts";

const mediaAttachmentSchema = z.union([
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

export type MediaAttachment = z.infer<typeof mediaAttachmentSchema>;

const toolReturnSchema = z.union([
  z.string(),
  z.object({
    result: z.string(),
    attachments: z.array(mediaAttachmentSchema).optional(),
  }),
]);

export type ToolReturn = z.infer<typeof toolReturnSchema>;

export type Tool<T extends ZodType> = {
  description: string;
  name: string;
  parameters: T;
  handler: (params: z.infer<T>) => Promise<string | ToolReturn>;
};

export type Skill = {
  name: string;
  description: string;
  instructions: string;
  // deno-lint-ignore no-explicit-any
  tools: Tool<any>[];
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
  name: string;
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
    return { ok: false, error: error as Error };
  }
};

const callToResult =
  // deno-lint-ignore no-explicit-any
  (actions: Tool<any>[]) => async <T extends ZodType>(fc: FunctionCall) => {
    const { name, args, id } = fc;
    const toolCallId = id;
    const action: Tool<T> | undefined = actions.find(({ name: n }) =>
      n === name
    );
    if (!name) throw new Error("Function call name is missing");
    if (!action) {
      return { toolCallId, name, result: `Function ${name} not found` };
    }
    const { handler, parameters } = action;
    const parseResult = parseWithCatch(parameters, args);
    if (!parseResult.ok) {
      return {
        toolCallId,
        name,
        result: `Invalid arguments: ${JSON.stringify(parseResult.error)}`,
      };
    }
    const out = await handler(parseResult.result);
    const parsed = parseWithCatch(toolReturnSchema, out);
    if (!parsed.ok) {
      throw new Error(
        `Tool "${name}" handler returned invalid value: ${JSON.stringify(parsed.error)}`,
      );
    }
    const validated = parsed.result;
    return typeof validated === "string"
      ? { toolCallId, name, result: validated }
      : {
        toolCallId,
        name,
        result: validated.result,
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
  timestamp: timestampGeneration.access(),
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

const sharedFields = () => ({
  id: idGeneration.access(),
  timestamp: timestampGeneration.access(),
});

const toolResultTurn = (
  { name, result, attachments, toolCallId }: {
    name: string;
    result: string;
    attachments?: MediaAttachment[];
    toolCallId?: string;
  },
): HistoryEvent => ({
  ...sharedFields(),
  type: "tool_result",
  isOwn: true,
  name,
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

// deno-lint-ignore no-explicit-any
const handleFunctionCalls = (tools: Tool<any>[]) =>
  pipe(
    // deno-lint-ignore no-explicit-any
    filter((p: HistoryEvent): p is ToolUse<any> => p.type === "tool_call"),
    // deno-lint-ignore no-explicit-any
    map((t: ToolUse<any>): FunctionCall => ({
      name: t.name,
      args: t.parameters,
      id: t.id,
    })),
    each(pipe(callToResult(tools), toolResultTurn, outputEvent)),
  );

export const runCommandToolName = "run_command";
export const learnSkillToolName = "learn_skill";

export const tool = <ParametersSchema extends z.ZodObject<z.ZodRawShape>>(
  tool: Tool<ParametersSchema>,
) => ({
  ...tool,
  handler: (params: z.infer<ParametersSchema>): ReturnType<
    typeof tool.handler
  > => tool.handler(params),
});

// deno-lint-ignore no-explicit-any
export const createSkillTools = (skills: Skill[]): Tool<any>[] => {
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
      handler: async ({ command, params }) => {
        if (!command.includes("/")) {
          return `Invalid command format. Expected "skillName/toolName", got "${command}". Available skills: ${skillNames}`;
        }
        const [skillName, toolName] = command.split("/");
        if (!skillMap[skillName]) {
          return `Skill "${skillName}" not found. Available skills: ${skillNames}`;
        }
        const fullToolName = `${skillName}/${toolName}`;
        const tool = toolMap[fullToolName];
        if (!tool) {
          return `Tool "${toolName}" not found in skill "${skillName}". Please call ${learnSkillToolName}.`;
        }
        const parseResult = parseWithCatch(tool.parameters.strict(), params);
        if (!parseResult.ok) {
          return `Invalid parameters for ${fullToolName}: ${parseResult.error.message}`;
        }
        return await tool.handler(parseResult.result);
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
              parameters: zodToGeminiParameters(tool.parameters),
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
  maxIterations: number;
  // deno-lint-ignore no-explicit-any
  onMaxIterationsReached: () => any;
  lightModel?: boolean;
  provider?: "gemini";
  imageGen?: boolean;
  rewriteHistory: (replacements: Record<string, HistoryEvent>) => Promise<void>;
  timezoneIANA: string;
  maxOutputTokens?: number;
};

export const runAbstractAgent = async (
  { maxIterations, tools, skills, onMaxIterationsReached }: AgentSpec,
  callModel: (history: HistoryEvent[]) => Promise<HistoryEvent[]>,
) => {
  const allTools = skills && skills.length > 0
    ? [...tools, ...createSkillTools(skills)]
    : tools;
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
    await handleFunctionCalls(allTools)(output);
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
  if (e.type === "participant_utterance" || e.type === "participant_edit_message") {
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
    return approxTextTokens(e.name) + approxTextTokens(e.result) +
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
