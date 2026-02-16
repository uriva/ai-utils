import {
  type Content,
  type FunctionCall,
  type GenerateContentParameters,
  type GenerateContentResponse,
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
} from "gamla";
import type { ZodType } from "zod/v4";
import {
  type AgentSpec,
  createSkillTools,
  doNothingEvent,
  generateId,
  type HistoryEventWithMetadata,
  type MediaAttachment,
  type MessageId,
  type OwnEditMessage,
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
  geminiFlashImageVersion,
  geminiFlashVersion,
  geminiProImageVersion,
  geminiProVersion,
  zodToGeminiParameters,
} from "./gemini.ts";

const is500Error = (error: unknown) =>
  error instanceof Error && "status" in error &&
  (error as { status: number }).status === 500;

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

type GeminiOutput = GeminiPartOfInterest[];

const extractFileIdFromError = (error: Error) => {
  const match = error.message.match(/File\s+([a-zA-Z0-9]+)/);
  return match ? match[1] : null;
};

const is403PermissionError = (error: Error) => {
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

const extractUnsupportedMimeType = (error: Error): string | undefined => {
  const match = error.message.match(/Unsupported MIME type:\s*([^"\s}]+)/);
  return match ? match[1] : undefined;
};

const getExpiredMediaText = (attachments: MediaAttachment[]) =>
  !empty(attachments)
    ? ` <media file expired: ${
      attachments.map((a: MediaAttachment) => a.caption || a.mimeType)
        .join(", ")
    }>`
    : "";

const stripExpiredFile = (error: Error, events: GeminiHistoryEvent[]) => {
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

const rawCallGemini = (
  req: GenerateContentParameters,
): Promise<GeminiOutput> =>
  new GoogleGenAI({ apiKey: accessGeminiToken() }).models.generateContent(req)
    .then((resp: GenerateContentResponse): GeminiOutput =>
      (resp.candidates?.[0]?.content?.parts ?? [])
        .flatMap((part: Part): GeminiOutput => {
          const { text, functionCall, thoughtSignature, inlineData, fileData } =
            part;
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
            return [{ type: "text", text, thoughtSignature }];
          }
          return [];
        })
    ).catch((err) => {
      geminiError.access(err, req);
      throw err;
    });

const callGeminiWithRetry = conditionalRetry(is500Error)(
  1000,
  4,
  rawCallGemini,
);

const callGemini = (req: GenerateContentParameters): Promise<GeminiOutput> =>
  callGeminiWithRetry(req).catch((err) => {
    if (!is500Error(err)) throw err;
    return rawCallGemini({
      ...req,
      model: alternateModel(req.model),
    });
  });

const actionToTool = ({ name, description, parameters }: Tool<ZodType>) => ({
  name,
  description,
  parameters: zodToGeminiParameters(parameters),
});

const formatTimestamp = (ts: number, timezoneIANA: string): string =>
  new Date(ts).toLocaleString("en-US", {
    timeZone: timezoneIANA,
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const attachmentsToParts = (attachments?: MediaAttachment[]): Part[] =>
  (attachments ?? []).flatMap((a): Part[] => {
    const mediaPart: Part = a.kind === "inline"
      ? { inlineData: { data: a.dataBase64, mimeType: a.mimeType } }
      : { fileData: { fileUri: a.fileUri, mimeType: a.mimeType } };
    const parts: Part[] = [mediaPart];
    if (a.caption && a.caption.trim()) {
      parts.push({ text: a.caption });
    }
    return parts;
  });

const referencedMessageText =
  (eventById: (id: string) => GeminiHistoryEvent | undefined) =>
  (onMessage: MessageId): string => {
    const msg = eventById(onMessage);
    return typeof msg === "object" && "text" in msg ? msg.text : "";
  };

const historyEventToContent =
  (
    eventById: (id: string) => GeminiHistoryEvent | undefined,
    timezoneIANA: string,
  ) =>
  (e: GeminiHistoryEvent): Content => {
    const getRefText = referencedMessageText(eventById);
    const stampText = (text: string) =>
      `[${formatTimestamp(e.timestamp, timezoneIANA)}] ${text}`;
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
        ...attachmentsToParts(e.attachments),
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
          thoughtSignature: e.modelMetadata?.thoughtSignature,
          text,
        });
      }
      if (e.attachments && !empty(e.attachments)) {
        parts.push(...attachmentsToParts(e.attachments));
      }
      return wrapModelContent(
        !empty(parts) ? parts : [{
          thoughtSignature: e.modelMetadata?.thoughtSignature,
          text: "",
        }],
      );
    }
    if (e.type === "tool_call") {
      return wrapModelContent([{
        thoughtSignature: e.modelMetadata?.thoughtSignature,
        functionCall: {
          name: e.name,
          args: isRecord(e.parameters) ? e.parameters : {},
        },
      }]);
    }
    if (e.type === "tool_result") {
      const parts: Part[] = [
        {
          functionResponse: {
            name: e.name,
            response: {
              result:
                stampText(e.result),
            },
          },
        },
        ...attachmentsToParts(e.attachments),
      ];
      return wrapUserContent(parts);
    }
    if (e.type === "own_thought") {
      return wrapUserContent([{
        text: stampText(`[Internal thought, visible only to you: ${e.text}]`),
      }]);
    }
    if (e.type === "own_reaction") {
      return wrapModelContent([{
        thoughtSignature: e.modelMetadata?.thoughtSignature,
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
      // Carry thoughtSignature if available (assume only one text part)
      return wrapModelContent([{
        text: "",
        thoughtSignature: e.modelMetadata?.thoughtSignature,
      }]);
    }
    throw new Error(
      `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
    );
  };

const combineContent = (contents: Content[]): Content => ({
  role: contents.some((c) => c.role === "model") ? "model" : "user",
  parts: contents.flatMap((c) => c.parts ?? []),
});

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

const buildReq = (
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
  | { type: "text"; text: string; thoughtSignature?: string }
  | GeminiFunctiontoolPart
  | GeminiInlinePart
  | GeminiFilePart;

const sawFunction = (output: GeminiOutput) =>
  output.some(({ type }: GeminiPartOfInterest) => type === "function_call");

const didNothing = (output: GeminiOutput) =>
  !sawFunction(output) &&
  !output.some((p: GeminiPartOfInterest) =>
    (p.type === "text" &&
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

  // Reservartion pass: mark calls that are claimed by strict ID matching
  const reservedCallIds = new Set<string>();
  history.forEach((e) => {
    if (e.type === "tool_result" && e.toolCallId) {
      const call = allCalls.find((c) => c.id === e.toolCallId);
      if (call) reservedCallIds.add(call.id);
    }
  });

  const processedStrictCalls = new Set<string>();
  const processedLegacyCalls = new Set<string>();

  return history.filter((e) => {
    if (e.type !== "tool_result") return true;

    if (e.toolCallId) {
      // Strict match: must match an ID not yet fully processed/duplicated
      const call = allCalls.find((c) =>
        c.id === e.toolCallId && !processedStrictCalls.has(c.id)
      );
      if (call) {
        processedStrictCalls.add(call.id);
        return true;
      }
      console.warn(
        `Warning: Filtering out orphaned tool_result for "${e.name}" (toolCallId: ${e.toolCallId}). ` +
          `No matching tool_call found with that ID.`,
      );
      return false;
    }

    // Legacy match: find available call (not reserved, not used)
    const call = allCalls.find((c) =>
      c.name === e.name &&
      c.timestamp < e.timestamp &&
      !reservedCallIds.has(c.id) &&
      !processedLegacyCalls.has(c.id)
    );

    if (call) {
      processedLegacyCalls.add(call.id);
      return true;
    }

    console.warn(
      `Warning: Filtering out orphaned tool_result for "${e.name}" (id: ${e.id}). ` +
        `No matching tool_call found.`,
    );
    return false;
  });
};

export const filterInvalidToolCalls = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => {
  return history.filter((e) => {
    if (e.type === "tool_call" && !e.modelMetadata?.thoughtSignature?.trim()) {
      console.warn(
        `Warning: Filtering out tool_call "${e.name}" (id: ${e.id}) with missing or empty thoughtSignature. ` +
          `This would cause Gemini API error: "Function call is missing a thought_signature in functionCall parts".`,
      );
      return false;
    }
    return true;
  });
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
      const kept = event.attachments.filter((att) =>
        att.mimeType !== mimeType
      );
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
): Promise<GeminiOutput> => {
  let currentEvents = events;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      return await callGemini(eventsToRequest(currentEvents));
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      const fixed = handleFileNotActiveError(err, currentEvents);
      if (!fixed) throw err;
      currentEvents = fixed;
    }
  }
  return callGemini(eventsToRequest(currentEvents));
};

const stripAllUnsupportedMimeTypes = async (
  initialError: Error,
  events: GeminiHistoryEvent[],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  rewriteHistory: AgentSpec["rewriteHistory"],
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
      const result = await callGemini(eventsToRequest(currentEvents));
      await rewriteHistory(allReplacements);
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      if (!isUnsupportedMimeTypeError(err)) throw err;
      currentError = err;
    }
  }
  return callGemini(eventsToRequest(currentEvents));
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

const stripAllExpiredFiles = async (
  initialError: Error,
  events: GeminiHistoryEvent[],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  rewriteHistory: AgentSpec["rewriteHistory"],
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
      const result = await callGemini(eventsToRequest(currentEvents));
      await rewriteHistory(allReplacements);
      return result;
    }
    Object.assign(allReplacements, fixed.replacements);
    currentEvents = fixed.updatedHistory;
    try {
      const result = await callGemini(eventsToRequest(currentEvents));
      await rewriteHistory(allReplacements);
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      if (!is403PermissionError(err)) throw err;
      currentError = err;
    }
  }
  return callGemini(eventsToRequest(currentEvents));
};

const callGeminiWithFixHistory = (
  rewriteHistory: AgentSpec["rewriteHistory"],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
) =>
async (events: GeminiHistoryEvent[]): Promise<GeminiOutput> => {
  try {
    return await callGemini(eventsToRequest(events));
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    if (isFileNotActiveError(err)) {
      return stripAllNotActiveFiles(events, eventsToRequest);
    }
    if (isUnsupportedMimeTypeError(err)) {
      return stripAllUnsupportedMimeTypes(
        err,
        events,
        eventsToRequest,
        rewriteHistory,
      );
    }
    if (!is403PermissionError(err)) throw err;
    return stripAllExpiredFiles(err, events, eventsToRequest, rewriteHistory);
  }
};

export const geminiAgentCaller = ({
  lightModel,
  prompt,
  tools,
  skills,
  imageGen,
  rewriteHistory,
  timezoneIANA,
  maxOutputTokens,
}: AgentSpec) =>
(events: GeminiHistoryEvent[]): Promise<GeminiHistoryEvent[]> =>
  pipe(
    filterOrphanedToolResults,
    filterInvalidToolCalls,
    filterUnsupportedGeminiAttachments,
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

const geminiOutputPartToHistoryEvent =
  (responseId: string) => (p: GeminiPartOfInterest): GeminiHistoryEvent => {
    if (p.type === "text") {
      return ownUtteranceTurnWithMetadata<GeminiMetadata>(
        typeof p.text === "string" ? p.text : "",
        {
          type: "gemini",
          responseId,
          thoughtSignature: p.thoughtSignature ?? "",
        },
      );
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
