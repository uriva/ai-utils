import {
  type Content,
  type FunctionCall,
  type GenerateContentParameters,
  type GenerateContentResponseUsageMetadata,
  GoogleGenAI,
  type Part,
} from "@google/genai";
import { context, type Injection } from "@uri/inject";
import {
  conditionalRetry,
  empty,
  filter,
  groupBy,
  map,
  pipe,
  sum,
} from "gamla";
import type { ZodType } from "zod/v4";
import {
  type AgentSpec,
  createSkillTools,
  doNothingEvent,
  estimateTokens,
  generateId,
  getStreamChunk,
  type HistoryEventWithMetadata,
  type MediaAttachment,
  type MessageId,
  type OwnEditMessage,
  ownThoughtTurnWithMetadata,
  type OwnUtterance,
  ownUtteranceTurnWithMetadata,
  type ParticipantEditMessage,
  type ParticipantUtterance,
  type Tool,
  type ToolResult,
  toolUseTurnWithMetadata,
} from "./agent.ts";
import {
  accessGeminiToken,
  attachmentsToParts,
  geminiFlashImageVersion,
  geminiFlashVersion,
  geminiProImageVersion,
  geminiProVersion,
  zodToGeminiParameters,
} from "./gemini.ts";
import {
  appendInternalSentTimestamp,
  stripInternalSentTimestampSuffix,
} from "./internalMessageMetadata.ts";

const normalizeError = (error: unknown): Error => {
  if (error instanceof Error) return error;
  if (typeof error === "string") return new Error(error);
  if (typeof error === "object" && error !== null) {
    const err = new Error(
      (error as { message?: string }).message || JSON.stringify(error),
    );
    Object.assign(err, error);
    return err;
  }
  return new Error(String(error));
};

const isServerError = (error: unknown) =>
  error instanceof Error && "status" in error &&
  (error as { status: number }).status >= 500;

const alternateModel = (model: string) =>
  model === geminiProVersion
    ? geminiFlashVersion
    : model === geminiFlashVersion
    ? geminiProVersion
    : model === geminiProImageVersion
    ? geminiFlashImageVersion
    : model === geminiFlashImageVersion
    ? geminiProImageVersion
    : model;

const geminiError: Injection<
  (_1: Error, _2: GenerateContentParameters) => void
> = context((_1: Error, _2: GenerateContentParameters) => {});

export const injectGeminiErrorLogger = geminiError.inject;

export type TokenUsage = GenerateContentResponseUsageMetadata;

const tokenUsage: Injection<
  (usage: TokenUsage, model: string) => void | Promise<void>
> = context((_: TokenUsage, _2: string) => {});

export const injectTokenUsage = tokenUsage.inject;

type GeminiOutput = GeminiPartOfInterest[];

export const extractFileIdFromError = (error: Error) => {
  const match = error.message.match(/File\s+([a-zA-Z0-9]+)/);
  return match ? match[1] : null;
};

export const is403PermissionError = (error: Error) => {
  if ("status" in error && (error as { status: number }).status === 403) {
    return true;
  }
  return error.message.includes("403") &&
    error.message.includes("PERMISSION_DENIED");
};

const isFileNotActiveError = (error: Error) =>
  error.message.includes("not in an ACTIVE state") ||
  (error.message.includes("FAILED_PRECONDITION") &&
    error.message.includes("File"));

const isUnsupportedMimeTypeError = (error: Error) =>
  error.message.includes("Unsupported MIME type");

const isTokenLimitExceeded = (error: Error) =>
  "status" in error && (error as { status: number }).status === 400 &&
  error.message.includes("token count exceeds");

const dropOldestHalf = <T extends { type: string }>(events: T[]): T[] => {
  if (events.length <= 2) return events;
  const half = Math.floor(events.length / 2);
  return events.slice(half);
};

export const capEventsToTokenBudget = (maxTokens: number) =>
(
  events: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => {
  const tokenCounts = map(estimateTokens)(events);
  const total = sum(tokenCounts);
  if (total <= maxTokens) return events;
  let cumulative = 0;
  const keepFromIndex = tokenCounts.findIndex((t: number) => {
    cumulative += t;
    return total - cumulative <= maxTokens;
  });
  const sliced = keepFromIndex < 0
    ? events.slice(-1)
    : events.slice(keepFromIndex + 1);
  return filterOrphanedToolResults(sliced);
};

const extractUnsupportedMimeType = (error: Error): string | undefined => {
  const match = error.message.match(/Unsupported MIME type:\s*([^"\\\s},]+)/);
  return match ? match[1] : undefined;
};

const getExpiredMediaText = (attachments: MediaAttachment[]) =>
  !empty(attachments)
    ? ` <media file expired: ${
      attachments.map((a: MediaAttachment) => a.caption || a.mimeType)
        .join(", ")
    }>`
    : "";

export const stripExpiredFile = (
  error: Error,
  events: GeminiHistoryEvent[],
) => {
  const fileId = extractFileIdFromError(error);
  if (!fileId) return undefined;
  const matchesFile = hasFileAttachment(fileId);
  const replacements = pipe(
    filter((event): event is EventWithAttachments => matchesFile(event)),
    map((event): [string, GeminiHistoryEvent] => {
      const placeholder = getExpiredMediaText(
        event.attachments?.filter((att: MediaAttachment) =>
          att.kind === "file" && att.fileUri.includes(fileId)
        ) ?? [],
      );
      return [event.id, stripFileFromEvent(fileId, placeholder)(event)];
    }),
    Object.fromEntries<GeminiHistoryEvent>,
  )(events);
  return {
    updatedHistory: map((event: GeminiHistoryEvent) =>
      event.id in replacements ? replacements[event.id] : event
    )(events),
    replacements,
  };
};

const rawCallGemini = async ({
  req,
  disableStreaming,
}: {
  req: GenerateContentParameters;
  disableStreaming?: boolean;
}): Promise<GeminiOutput> => {
  const handleStreamChunk = getStreamChunk();
  const sdk = new GoogleGenAI({ apiKey: accessGeminiToken() });
  let finalUsageMetadata: TokenUsage | undefined;
  const accumulatedParts: Part[] = [];

  if (disableStreaming) {
    const response = await sdk.models.generateContent(req);
    finalUsageMetadata = response.usageMetadata;
    const parts = response.candidates?.[0]?.content?.parts ?? [];
    for (const part of parts) {
      if (
        typeof part.text === "string" && !part.thought
      ) {
        await handleStreamChunk(part.text);
      }
      accumulatedParts.push(part);
    }
  } else {
    const responseStream = await sdk.models.generateContentStream(req);

    for await (const chunk of responseStream) {
      if (chunk.usageMetadata) {
        finalUsageMetadata = chunk.usageMetadata;
      }
      const parts = chunk.candidates?.[0]?.content?.parts ?? [];
      for (const part of parts) {
        if (
          typeof part.text === "string" && !part.thought &&
          !part.thoughtSignature
        ) {
          await handleStreamChunk(part.text);
        }

        if (typeof part.text === "string") {
          const lastPart = accumulatedParts[accumulatedParts.length - 1];
          if (
            lastPart &&
            typeof lastPart.text === "string" &&
            lastPart.thought === part.thought &&
            lastPart.thoughtSignature === part.thoughtSignature
          ) {
            lastPart.text += part.text;
          } else {
            accumulatedParts.push({ ...part });
          }
        } else if (part.functionCall) {
          // Assume functionCalls are fully formed or overwrite previous partial ones of the same name
          const lastPart = accumulatedParts[accumulatedParts.length - 1];
          if (
            lastPart && lastPart.functionCall &&
            lastPart.functionCall.name === part.functionCall.name
          ) {
            // If the SDK streams function calls by updating the object, we just replace it
            lastPart.functionCall = part.functionCall;
            if (part.thoughtSignature) {
              lastPart.thoughtSignature = part.thoughtSignature;
            }
          } else {
            accumulatedParts.push({ ...part });
          }
        } else {
          accumulatedParts.push(part);
        }
      }
    }
  }

  if (finalUsageMetadata) {
    tokenUsage.access(finalUsageMetadata, req.model);
  }

  return accumulatedParts.flatMap((part: Part): GeminiOutput => {
    const {
      text,
      functionCall,
      thoughtSignature,
      inlineData,
      fileData,
      thought,
    } = part;
    if (functionCall) {
      return [{ type: "function_call", functionCall, thoughtSignature }];
    }
    if (inlineData) {
      return [{ type: "inline_data", inlineData, thoughtSignature }];
    }
    if (fileData) {
      return [{ type: "file_data", fileData, thoughtSignature }];
    }
    if (typeof text === "string") {
      return [{ type: "text", text, thoughtSignature, thought }];
    }
    return [];
  });
};

const callGeminiWithRetry = conditionalRetry(isServerError)(
  1000,
  4,
  rawCallGemini,
);

const callGemini = (
  req: GenerateContentParameters,
  disableStreaming?: boolean,
): Promise<GeminiOutput> =>
  callGeminiWithRetry({ req, disableStreaming }).catch((err) => {
    if (!isServerError(err)) throw err;
    return rawCallGemini({
      req: {
        ...req,
        model: alternateModel(req.model),
      },
      disableStreaming,
    });
  });

const actionToTool = ({ name, description, parameters }: Tool<ZodType>) => ({
  name,
  description,
  parameters: zodToGeminiParameters(parameters),
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const optionalThoughtSignature = (sig: string | undefined) =>
  sig ? { thoughtSignature: sig } : {};

const attachmentsToPartsOrEmpty = (attachments?: MediaAttachment[]): Part[] =>
  attachmentsToParts(attachments ?? []);

const referencedMessageText =
  (eventById: (id: string) => GeminiHistoryEvent | undefined) =>
  (onMessage: MessageId): string => {
    const msg = eventById(onMessage);
    return typeof msg === "object" && "text" in msg ? msg.text : "";
  };

const historyEventToContent = (
  eventById: (id: string) => GeminiHistoryEvent | undefined,
  timezoneIANA: string,
) =>
(e: GeminiHistoryEvent): Content => {
  const getRefText = referencedMessageText(eventById);
  const stampText = (text: string) =>
    appendInternalSentTimestamp(text, e.timestamp, timezoneIANA);
  if (
    e.type === "participant_utterance" ||
    e.type === "participant_edit_message"
  ) {
    const text = e.type === "participant_edit_message"
      ? `${e.name} edited message "${
        getRefText(e.onMessage).slice(0, 100)
      }" to: ${e.text}`
      : e.text
      ? `${e.name}: ${e.text}`
      : "";
    return wrapUserContent([
      text
        ? {
          text: stampText(text),
        }
        : undefined,
      ...attachmentsToPartsOrEmpty(e.attachments),
    ].filter((x): x is Part => !!x));
  }
  if (e.type === "own_utterance" || e.type === "own_edit_message") {
    const text = e.type === "own_edit_message"
      ? `You edited message "${
        getRefText(e.onMessage).slice(0, 100)
      }" to: ${e.text}`
      : e.text;
    const parts: Part[] = [];
    if (text) {
      parts.push({
        ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
        text,
      });
    }
    if (e.attachments && !empty(e.attachments)) {
      parts.push(...attachmentsToParts(e.attachments));
    }
    return wrapModelContent(
      !empty(parts) ? parts : [{
        ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
        text: "",
      }],
    );
  }
  if (e.type === "tool_call") {
    return wrapModelContent([{
      ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
      functionCall: {
        name: e.name,
        args: isRecord(e.parameters) ? e.parameters : {},
      },
    }]);
  }
  if (e.type === "tool_result") {
    const toolCall = e.toolCallId ? eventById(e.toolCallId) : undefined;
    const name = toolCall && "name" in toolCall ? toolCall.name : "unknown";
    const parts: Part[] = [
      {
        functionResponse: {
          name,
          response: {
            result: stampText(e.result),
          },
        },
      },
      ...attachmentsToPartsOrEmpty(e.attachments),
    ];
    return wrapUserContent(parts);
  }
  if (e.type === "own_thought") {
    return e.modelMetadata?.thoughtSignature
      ? wrapModelContent([{
        text: e.text,
        thought: true,
        thoughtSignature: e.modelMetadata.thoughtSignature,
      }])
      : e.modelMetadata
      ? wrapModelContent([{ text: "" }])
      : wrapUserContent([{
        text: stampText(`[System notification: ${e.text}]`),
      }]);
  }
  if (e.type === "own_reaction") {
    return wrapModelContent([{
      ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
      text: `You reacted ${e.reaction} to message: ${
        getRefText(e.onMessage).slice(0, 100)
      }`,
    }]);
  }
  if (e.type === "participant_reaction") {
    return wrapUserContent([{
      text: `${e.name} reacted ${e.reaction} to message: ${
        getRefText(e.onMessage).slice(0, 100)
      }`,
    }]);
  }
  if (e.type === "do_nothing") {
    return wrapModelContent([{
      text: "",
      ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
    }]);
  }
  throw new Error(
    `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
  );
};

const combineContent = (contents: Content[]): Content => {
  const parts = contents.flatMap((c) => c.parts ?? []);
  const signature = parts.find((p) => p.thoughtSignature)?.thoughtSignature;
  return {
    role: contents.some((c) => c.role === "model") ? "model" : "user",
    parts: signature
      ? parts.map((p) => ({ ...p, thoughtSignature: signature }))
      : parts,
  };
};

const wrapRole = (role: "user" | "model") => (parts: Part[]): Content => ({
  role,
  parts,
});

const wrapModelContent = wrapRole("model");

const wrapUserContent = wrapRole("user");

const getOriginalId = (e: GeminiHistoryEvent): string =>
  "modelMetadata" in e ? e.modelMetadata?.responseId ?? e.id : e.id;

const fixStart = (history: Content[]) =>
  (empty(history) || history[0].role !== "user")
    ? [
      { role: "user", parts: [{ text: "<conversation started>" }] },
      ...history,
    ]
    : history;

export const buildReq = (
  imageGen: boolean | undefined,
  lightModel: boolean | undefined,
  prompt: string,
  tools: Tool<ZodType>[],
  timezoneIANA: string,
  maxOutputTokens: number | undefined,
) =>
(events: GeminiHistoryEvent[]): GenerateContentParameters => ({
  model: imageGen
    ? (lightModel ? geminiFlashImageVersion : geminiProImageVersion)
    : (lightModel ? geminiFlashVersion : geminiProVersion),
  config: {
    systemInstruction: prompt,
    tools: [{ functionDeclarations: tools.map(actionToTool) }],
    toolConfig: { functionCallingConfig: {} },
    thinkingConfig: { includeThoughts: true },
    ...(maxOutputTokens ? { maxOutputTokens } : {}),
  },
  contents: pipe(
    groupBy(getOriginalId),
    Object.values<GeminiHistoryEvent[]>,
    map(
      pipe(
        map(historyEventToContent(indexById(events), timezoneIANA)),
        combineContent,
      ),
    ),
    fixStart,
  )(events),
});

const indexById = (events: GeminiHistoryEvent[]) => {
  const eventIdToEvents = groupBy(({ id }: GeminiHistoryEvent) => id)(events);
  return (id: MessageId) => eventIdToEvents[id]?.[0];
};

type GeminiFunctiontoolPart = {
  type: "function_call";
  functionCall: FunctionCall;
  thoughtSignature?: string;
};

type GeminiInlinePart = {
  type: "inline_data";
  inlineData: NonNullable<Part["inlineData"]>;
  thoughtSignature?: string;
};

type GeminiFilePart = {
  type: "file_data";
  fileData: NonNullable<Part["fileData"]>;
  thoughtSignature?: string;
};

type GeminiMetadata = {
  type: "gemini";
  thoughtSignature: string;
  responseId: string;
};

type GeminiHistoryEvent = HistoryEventWithMetadata<GeminiMetadata>;

type GeminiPartOfInterest =
  | { type: "text"; text: string; thoughtSignature?: string; thought?: boolean }
  | GeminiFunctiontoolPart
  | GeminiInlinePart
  | GeminiFilePart;

const sawFunction = (output: GeminiOutput) =>
  output.some(({ type }: GeminiPartOfInterest) => type === "function_call");

const didNothing = (output: GeminiOutput) =>
  !sawFunction(output) &&
  !output.some((p: GeminiPartOfInterest) =>
    (p.type === "text" && !p.thought &&
      p.text.replace(/[\s\u200B\u200C\u200D\uFEFF]/g, "")) ||
    p.type === "inline_data" ||
    p.type === "file_data"
  );

const isUnsupportedGeminiMimeType = (mimeType: string | undefined): boolean => {
  if (!mimeType) return true;
  const normalized = mimeType.trim().toLowerCase();
  if (!normalized) return true;
  if (normalized === "application/octet-stream") return true;
  return false;
};

const filterDoNothing = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => history.filter((e) => e.type !== "do_nothing");

const filterUnsupportedGeminiAttachments = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] =>
  history.map((event) => {
    if (!("attachments" in event)) return event;
    const attachments = event.attachments ?? [];
    if (empty(attachments)) return event;
    const kept = attachments.filter((att) =>
      !isUnsupportedGeminiMimeType(att.mimeType)
    );
    if (kept.length === attachments.length) return event;
    const removed = attachments.filter((att) =>
      isUnsupportedGeminiMimeType(att.mimeType)
    );
    if (!empty(removed)) {
      console.warn(
        `Warning: Filtering out unsupported Gemini attachment mime types on event ${event.id}: ${
          removed.map((att) => att.mimeType).join(", ")
        }`,
      );
    }
    return {
      ...event,
      attachments: empty(kept) ? undefined : kept,
    };
  });

export const filterOrphanedToolResults = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => {
  const allCalls = history.filter((e) => e.type === "tool_call");
  const processedCallIds = new Set<string>();

  return history.filter((e) => {
    if (e.type !== "tool_result") return true;

    if (!e.toolCallId) {
      console.warn(
        `Warning: Filtering out orphaned tool_result (id: ${e.id}) with no toolCallId.`,
      );
      return false;
    }

    const call = allCalls.find((c) =>
      c.id === e.toolCallId && !processedCallIds.has(c.id)
    );
    if (call) {
      processedCallIds.add(call.id);
      return true;
    }
    console.warn(
      `Warning: Filtering out orphaned tool_result (toolCallId: ${e.toolCallId}). ` +
        `No matching tool_call found with that ID.`,
    );
    return false;
  });
};

export const filterInvalidToolCalls = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] =>
  history.filter((e) => {
    if (e.type === "tool_call" && !e.modelMetadata?.thoughtSignature?.trim()) {
      console.warn(
        `Warning: Filtering out tool_call "${e.name}" (id: ${e.id}) with missing or empty thoughtSignature. ` +
          `This would cause Gemini API error: "Function call is missing a thought_signature in functionCall parts".`,
      );
      return false;
    }
    return true;
  });

const toolCallToOwnThought = (e: GeminiHistoryEvent): GeminiHistoryEvent => ({
  type: "own_thought",
  isOwn: true,
  id: e.id,
  timestamp: e.timestamp,
  text: `[Removed tool call "${
    "name" in e ? e.name : "unknown"
  }" due to missing thought signature. Parameters: ${
    JSON.stringify("parameters" in e ? e.parameters : {})
  }]`,
});

const toolResultToOwnThought = (e: GeminiHistoryEvent): GeminiHistoryEvent => ({
  type: "own_thought",
  isOwn: true,
  id: e.id,
  timestamp: e.timestamp,
  text: `[Removed tool result: ${"result" in e ? e.result : ""}]`,
});

const textToOwnThought = (e: GeminiHistoryEvent): GeminiHistoryEvent => ({
  type: "own_thought",
  isOwn: true,
  id: e.id,
  timestamp: e.timestamp,
  text: `[Removed ${e.type} from response containing invalid tool call: ${
    "text" in e ? (e.text as string).slice(0, 200) : ""
  }]`,
});

export const filterAndRewriteInvalidToolCalls =
  (rewriteHistory: AgentSpec["rewriteHistory"]) =>
  (history: GeminiHistoryEvent[]): GeminiHistoryEvent[] => {
    // Group all tool calls by their original response ID
    const toolCallsByResponseId = new Map<string, GeminiHistoryEvent[]>();
    for (const e of history) {
      if (e.type === "tool_call") {
        const id = getOriginalId(e);
        const group = toolCallsByResponseId.get(id) || [];
        group.push(e);
        toolCallsByResponseId.set(id, group);
      }
    }

    // A response group is tainted if it contains tool calls, but NONE of them
    // have a thoughtSignature. (In parallel calls, only the first gets a signature).
    const taintedResponseIds = new Set<string>();
    for (const [responseId, toolCalls] of toolCallsByResponseId.entries()) {
      const hasSignature = toolCalls.some((e) =>
        "modelMetadata" in e && e.modelMetadata?.thoughtSignature?.trim()
      );
      if (!hasSignature) {
        taintedResponseIds.add(responseId);
      }
    }

    if (taintedResponseIds.size === 0) return history;

    const isTainted = (e: GeminiHistoryEvent) =>
      taintedResponseIds.has(getOriginalId(e));

    const replacements: Record<string, GeminiHistoryEvent> = {};
    const filtered = history.filter((e) => {
      if (e.type === "tool_call" && isTainted(e)) {
        replacements[e.id] = toolCallToOwnThought(e);
        return false;
      }
      if (
        e.type === "tool_result" && "toolCallId" in e && e.toolCallId
      ) {
        // Find if the parent tool call was tainted
        const parentCall = history.find((h) => h.id === e.toolCallId);
        if (parentCall && isTainted(parentCall)) {
          replacements[e.id] = toolResultToOwnThought(e);
          return false;
        }
      }
      if (
        (e.type === "own_utterance" || e.type === "own_thought") && isTainted(e)
      ) {
        replacements[e.id] = textToOwnThought(e);
        return false;
      }
      return true;
    });

    rewriteHistory(replacements).catch((err) =>
      console.warn("Failed to rewrite history for invalid tool calls:", err)
    );
    return filtered;
  };

const hasFileAttachment =
  (fileId: string) =>
  (event: GeminiHistoryEvent): event is EventWithAttachments =>
    "attachments" in event &&
    !!event.attachments?.some((att) =>
      att.kind === "file" && att.fileUri.includes(fileId)
    );

type EventWithAttachments =
  | ParticipantUtterance
  | OwnUtterance<GeminiMetadata>
  | ParticipantEditMessage
  | OwnEditMessage<GeminiMetadata>
  | ToolResult;

const stripFileFromEvent =
  (fileId: string, placeholder: string) =>
  (event: EventWithAttachments): EventWithAttachments => ({
    ...event,
    ...event.type === "tool_result"
      ? { result: event.result + placeholder }
      : { text: (event.text ?? "") + placeholder },
    attachments: event.attachments?.filter((att) =>
      att.kind === "inline" ||
      (att.kind === "file" && !att.fileUri.includes(fileId))
    ),
  });

const stripAttachmentsByMimeType = (
  mimeType: string,
  events: GeminiHistoryEvent[],
): {
  updatedHistory: GeminiHistoryEvent[];
  replacements: Record<string, GeminiHistoryEvent>;
} => {
  const replacements: Record<string, GeminiHistoryEvent> = {};
  const updatedHistory = map(
    (event: GeminiHistoryEvent): GeminiHistoryEvent => {
      if (!("attachments" in event) || !event.attachments) return event;
      const kept = event.attachments.filter((att) => att.mimeType !== mimeType);
      if (kept.length === event.attachments.length) return event;
      const placeholder = ` <unsupported file type removed: ${mimeType}>`;
      const updated = {
        ...event,
        ...event.type === "tool_result"
          ? { result: event.result + placeholder }
          : { text: ((event as { text?: string }).text ?? "") + placeholder },
        attachments: empty(kept) ? undefined : kept,
      } as GeminiHistoryEvent;
      replacements[event.id] = updated;
      return updated;
    },
  )(events);
  return { updatedHistory, replacements };
};

const replaceFileWithProcessingPlaceholder = (
  fileId: string,
  events: GeminiHistoryEvent[],
) => {
  const shouldReplace = hasFileAttachment(fileId);
  const replace = stripFileFromEvent(
    fileId,
    " <user sent a file which is still being processed>",
  );
  return map((event: GeminiHistoryEvent): GeminiHistoryEvent =>
    shouldReplace(event) ? replace(event) : event
  )(events);
};

const handleFileNotActiveError = (
  error: Error,
  events: GeminiHistoryEvent[],
) => {
  if (!isFileNotActiveError(error)) return undefined;
  const fileId = extractFileIdFromError(error);
  if (!fileId) return undefined;
  return replaceFileWithProcessingPlaceholder(fileId, events);
};

const stripAllNotActiveFiles = async (
  events: GeminiHistoryEvent[],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  disableStreaming?: boolean,
): Promise<GeminiOutput> => {
  let currentEvents = events;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      return await callGemini(eventsToRequest(currentEvents), disableStreaming);
    } catch (error) {
      const err = normalizeError(error);
      const fixed = handleFileNotActiveError(err, currentEvents);
      if (!fixed) throw err;
      currentEvents = fixed;
    }
  }
  return callGemini(eventsToRequest(currentEvents), disableStreaming);
};

const stripAllUnsupportedMimeTypes = async (
  initialError: Error,
  events: GeminiHistoryEvent[],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  rewriteHistory: AgentSpec["rewriteHistory"],
  disableStreaming?: boolean,
): Promise<GeminiOutput> => {
  let currentEvents = events;
  let currentError = initialError;
  const allReplacements: Record<string, GeminiHistoryEvent> = {};
  for (let attempt = 0; attempt < 5; attempt++) {
    const mimeType = extractUnsupportedMimeType(currentError);
    if (!mimeType) throw currentError;
    console.warn(
      `Stripping unsupported MIME type from history: ${mimeType}`,
    );
    const { updatedHistory, replacements } = stripAttachmentsByMimeType(
      mimeType,
      currentEvents,
    );
    Object.assign(allReplacements, replacements);
    currentEvents = updatedHistory;
    try {
      const result = await callGemini(
        eventsToRequest(currentEvents),
        disableStreaming,
      );
      await rewriteHistory(allReplacements);
      return result;
    } catch (error) {
      const err = normalizeError(error);
      if (!isUnsupportedMimeTypeError(err)) throw err;
      currentError = err;
    }
  }
  return callGemini(eventsToRequest(currentEvents), disableStreaming);
};

const stripAllFileAttachments = (
  events: GeminiHistoryEvent[],
): {
  updatedHistory: GeminiHistoryEvent[];
  replacements: Record<string, GeminiHistoryEvent>;
} => {
  const replacements: Record<string, GeminiHistoryEvent> = {};
  const updatedHistory = map(
    (event: GeminiHistoryEvent): GeminiHistoryEvent => {
      if (!("attachments" in event) || !event.attachments) return event;
      const fileAttachments = event.attachments.filter((att) =>
        att.kind === "file"
      );
      if (empty(fileAttachments)) return event;
      const placeholder = getExpiredMediaText(fileAttachments);
      const kept = event.attachments.filter((att) => att.kind !== "file");
      const updated = {
        ...event,
        ...event.type === "tool_result"
          ? { result: event.result + placeholder }
          : { text: ((event as { text?: string }).text ?? "") + placeholder },
        attachments: empty(kept) ? undefined : kept,
      } as GeminiHistoryEvent;
      replacements[event.id] = updated;
      return updated;
    },
  )(events);
  return { updatedHistory, replacements };
};

export const stripAllExpiredFiles = async (
  initialError: Error,
  events: GeminiHistoryEvent[],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  rewriteHistory: AgentSpec["rewriteHistory"],
  disableStreaming?: boolean,
): Promise<GeminiOutput> => {
  let currentEvents = events;
  let currentError = initialError;
  const allReplacements: Record<string, GeminiHistoryEvent> = {};
  for (let attempt = 0; attempt < 5; attempt++) {
    const fixed = stripExpiredFile(currentError, currentEvents);
    if (!fixed) throw currentError;
    if (empty(Object.keys(fixed.replacements))) {
      console.warn(
        `Could not find file referenced in 403 error in any attachment. Stripping all file attachments as fallback.`,
      );
      const nuclear = stripAllFileAttachments(currentEvents);
      Object.assign(allReplacements, nuclear.replacements);
      currentEvents = nuclear.updatedHistory;
      try {
        const result = await callGemini(
          eventsToRequest(currentEvents),
          disableStreaming,
        );
        await rewriteHistory(allReplacements);
        return result;
      } catch (nuclearError) {
        const err = normalizeError(nuclearError);
        throw new Error(
          `403 persists after stripping all file attachments: ${err.message}`,
        );
      }
    }
    Object.assign(allReplacements, fixed.replacements);
    currentEvents = fixed.updatedHistory;
    try {
      const result = await callGemini(
        eventsToRequest(currentEvents),
        disableStreaming,
      );
      await rewriteHistory(allReplacements);
      return result;
    } catch (error) {
      const err = normalizeError(error);
      if (!is403PermissionError(err)) throw err;
      currentError = err;
    }
  }
  throw new Error(
    `403 persists after 5 attempts to strip expired files: ${currentError.message}`,
  );
};

const callGeminiWithFixHistory = (
  rewriteHistory: AgentSpec["rewriteHistory"],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  disableStreaming?: boolean,
) =>
async (events: GeminiHistoryEvent[]): Promise<GeminiOutput> => {
  try {
    try {
      return await callGemini(eventsToRequest(events), disableStreaming);
    } catch (error) {
      const err = normalizeError(error);
      if (isTokenLimitExceeded(err)) {
        const totalTokens = sum(map(estimateTokens)(events));
        console.warn(
          `Token limit exceeded (estimated ${totalTokens} tokens, ${events.length} events). Dropping oldest half.`,
        );
        const truncated = dropOldestHalf(events);
        if (truncated.length === events.length) throw err;
        return callGemini(eventsToRequest(truncated), disableStreaming);
      }
      if (isFileNotActiveError(err)) {
        return stripAllNotActiveFiles(
          events,
          eventsToRequest,
          disableStreaming,
        );
      }
      if (isUnsupportedMimeTypeError(err)) {
        return stripAllUnsupportedMimeTypes(
          err,
          events,
          eventsToRequest,
          rewriteHistory,
          disableStreaming,
        );
      }
      if (!is403PermissionError(err)) throw err;
      return stripAllExpiredFiles(
        err,
        events,
        eventsToRequest,
        rewriteHistory,
        disableStreaming,
      );
    }
  } catch (terminalError) {
    const err = normalizeError(terminalError);
    geminiError.access(err, eventsToRequest(events));
    throw err;
  }
};

const maxHistoryTokens = 800_000;

export const geminiAgentCaller = ({
  lightModel,
  prompt,
  tools,
  skills,
  imageGen,
  rewriteHistory,
  timezoneIANA,
  maxOutputTokens,
  disableStreaming,
}: AgentSpec) =>
(events: GeminiHistoryEvent[]): Promise<GeminiHistoryEvent[]> =>
  pipe(
    filterAndRewriteInvalidToolCalls(rewriteHistory),
    filterOrphanedToolResults,
    filterDoNothing,
    filterUnsupportedGeminiAttachments,
    capEventsToTokenBudget(maxHistoryTokens),
    callGeminiWithFixHistory(
      rewriteHistory,
      buildReq(
        imageGen,
        lightModel,
        skills && skills.length > 0
          ? `${prompt}\n\nAvailable skills:\n${
            skills.map((skill) => `- ${skill.name}: ${skill.description}`)
              .join("\n")
          }`
          : prompt,
        [
          ...tools,
          ...(skills && skills.length > 0 ? createSkillTools(skills) : []),
        ],
        timezoneIANA,
        maxOutputTokens,
      ),
      disableStreaming,
    ),
    (geminiOutput: GeminiOutput): GeminiHistoryEvent[] => {
      const responseId = generateId();
      if (didNothing(geminiOutput)) {
        const textPart = geminiOutput.find((p) =>
          p.type === "text" && p.thoughtSignature
        );
        return [doNothingEvent(
          textPart?.thoughtSignature
            ? {
              type: "gemini",
              responseId,
              thoughtSignature: textPart.thoughtSignature,
            }
            : undefined,
        )];
      }
      return geminiOutput.map(geminiOutputPartToHistoryEvent(responseId));
    },
  )(events);

const embeddedThoughtPattern =
  /\[Internal thought, visible only to you: [\s\S]*?\]/g;

export const stripEmbeddedThoughtPatterns = (text: string): string =>
  text.replace(embeddedThoughtPattern, "").trim();

const geminiOutputPartToHistoryEvent =
  (responseId: string) => (p: GeminiPartOfInterest): GeminiHistoryEvent => {
    if (p.type === "text") {
      const metadata: GeminiMetadata = {
        type: "gemini",
        responseId,
        thoughtSignature: p.thoughtSignature ?? "",
      };
      const text = typeof p.text === "string" ? p.text : "";

      const stripped = stripInternalSentTimestampSuffix(text);
      const thoughtRegex =
        /^\[Internal thought, visible only to you: ([\s\S]*?)\]$/;
      const match = stripped.match(thoughtRegex);

      if (match) {
        return ownThoughtTurnWithMetadata<GeminiMetadata>(match[1], metadata);
      }

      const cleanedText = stripEmbeddedThoughtPatterns(stripped);

      return p.thought
        ? ownThoughtTurnWithMetadata<GeminiMetadata>(cleanedText, metadata)
        : ownUtteranceTurnWithMetadata<GeminiMetadata>(cleanedText, metadata);
    }
    if (p.type === "function_call") {
      return toolUseTurnWithMetadata(p.functionCall, {
        type: "gemini",
        responseId,
        thoughtSignature: p.thoughtSignature ?? "",
      });
    }
    if (p.type === "inline_data") {
      const { data, mimeType } = p.inlineData;
      if (!data) {
        return ownUtteranceTurnWithMetadata<GeminiMetadata>("", {
          type: "gemini",
          responseId,
          thoughtSignature: "",
        });
      }
      return ownUtteranceTurnWithMetadata<GeminiMetadata>(
        "",
        {
          type: "gemini",
          responseId,
          thoughtSignature: p.thoughtSignature ?? "",
        },
        [{
          kind: "inline",
          mimeType: mimeType ?? "application/octet-stream",
          dataBase64: data,
        }],
      );
    }
    if (p.type === "file_data") {
      const { fileUri, mimeType } = p.fileData;
      return fileUri
        ? ownUtteranceTurnWithMetadata<GeminiMetadata>(
          "",
          {
            type: "gemini",
            responseId,
            thoughtSignature: p.thoughtSignature ?? "",
          },
          [{
            kind: "file",
            mimeType: mimeType ?? "application/octet-stream",
            fileUri,
          }],
        )
        : ownUtteranceTurnWithMetadata<GeminiMetadata>("", {
          type: "gemini",
          responseId,
          thoughtSignature: "",
        });
    }
    throw new Error(`Unknown part type: ${JSON.stringify(p)}`);
  };
