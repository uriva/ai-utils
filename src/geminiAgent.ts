import {
  type Content,
  type FunctionCall,
  FunctionCallingConfigMode,
  type FunctionDeclaration,
  type GenerateContentParameters,
  type GenerateContentResponseUsageMetadata,
  GoogleGenAI,
  HarmBlockThreshold,
  HarmCategory,
  type Part,
} from "@google/genai";
import { context, type Injection } from "@uri/inject";
import {
  conditionalRetryExponential,
  empty,
  filter,
  groupBy,
  map,
  pipe,
  sum,
} from "gamla";
import { collapseDuplicatedText } from "./utils.ts";
import type { ZodType } from "zod/v4";
import {
  accessMetadataStore,
  type AgentSpec,
  createSkillTools,
  doNothingEventWithMetadata,
  estimateAgentInputTokens,
  estimateTokens,
  externalEventPrefix,
  generateId,
  getStreamChunk,
  getStreamThinkingChunk,
  type HistoryEventWithMetadata,
  historyHasPendingDeferredUserWaitingNudge,
  isRecord,
  type MediaAttachment,
  type MessageId,
  noResponseTag,
  type OwnEditMessage,
  ownThoughtTurnWithMetadata,
  type OwnUtterance,
  ownUtteranceTurnWithMetadata,
  type ParticipantEditMessage,
  type ParticipantUtterance,
  systemInstructionTail,
  systemNotificationPrefix,
  thinkingTokenExhaustionWarningText,
  type Tool,
  type ToolResult,
  toolUseTurnWithMetadata,
} from "./agent.ts";
import {
  accessGeminiToken,
  alternateGeminiModelVersion,
  attachmentsToParts,
  ensureGeminiAttachmentIsLink,
  geminiModelVersion,
  geminiThinkingConfig,
  isGeminiFileUri,
  zodToGeminiParameters,
} from "./gemini.ts";
import {
  appendInternalSentTimestamp,
  stripInternalSentTimestampSuffix,
} from "./internalMessageMetadata.ts";
import { inspectMediaUrlToolName } from "./inspectMediaTool.ts";
import {
  extractJsonThought,
  internalThoughtMarker,
  stripJsonThought,
} from "./jsonThought.ts";
import {
  is403PermissionError,
  isInvalidArgumentError,
  isRetryableError,
  isSyntheticTimeoutError,
  normalizeError,
  stripAnsi,
  syntheticTimeoutMarker,
} from "./utils.ts";
import { assertNoScriptDrift } from "./scriptDriftGuard.ts";
import { isCompactedSummaryText } from "./compaction.ts";
import {
  getOrCreateGeminiContextCache,
  invalidateGeminiContextCache,
  isGeminiContextCachingEnabled,
} from "./geminiContextCache.ts";

// Verbose [gemini-step]/[gemini-diag] request logging. On by
// default; tests and noise-sensitive consumers can silence or redirect it.
const geminiLog: Injection<(line: string) => void> = context((line: string) =>
  console.log(line)
);

const logGemini = geminiLog.access;

const geminiError: Injection<
  (_1: Error, _2: GenerateContentParameters) => void
> = context((_1: Error, _2: GenerateContentParameters) => {});

export const injectGeminiErrorLogger = geminiError.inject;

const promptBlocked: Injection<
  (_1: string, _2: GenerateContentParameters) => void
> = context((_1: string, _2: GenerateContentParameters) => {});

// Fires when Gemini rejects the entire prompt (`promptFeedback.blockReason`,
// zero candidates). Consumers can use this to alert the bot operator that their
// persona/history is driving content blocks, since the end user only sees the
// generic safety notice and cannot edit the prompt themselves.
export const injectPromptBlockedLogger = promptBlocked.inject;

export type TokenUsage = GenerateContentResponseUsageMetadata;

const tokenUsage: Injection<
  (usage: TokenUsage, model: string) => void | Promise<void>
> = context((_: TokenUsage, _2: string) => {});

export const injectTokenUsage = tokenUsage.inject;

const finishReasonSink: Injection<(reason: string) => void> = context(
  (_: string) => {},
);

export type GeminiOutput = GeminiPartOfInterest[];

export const extractFileIdFromError = (error: Error) => {
  const match = error.message.match(/File\s+([a-zA-Z0-9]+)/);
  return match ? match[1] : null;
};

const geminiMalformedFunctionCallReason = "MALFORMED_FUNCTION_CALL";

export const geminiMalformedFunctionCallError = (parts: Part[]) => {
  const err = new Error(
    `Gemini returned ${geminiMalformedFunctionCallReason} with ${parts.length} parts`,
  );
  Object.assign(err, { status: 503 });
  return err;
};

// Gemini may return parts alongside MALFORMED_FUNCTION_CALL: an empty-args
// functionCall shell plus the raw arg-JSON as text fragments. Those parts are
// garbage — emitting them leaks JSON fragments to the user and executes a
// broken tool call — so reject unconditionally and let retry/fallback run.
export const rejectMalformedFunctionCall = (
  finishReason: string | undefined,
  parts: Part[],
) => {
  if (finishReason !== geminiMalformedFunctionCallReason) return;
  throw geminiMalformedFunctionCallError(parts);
};

export { is403PermissionError };

const isFileNotActiveError = (error: Error) =>
  error.message.includes("not in an ACTIVE state") ||
  (error.message.includes("FAILED_PRECONDITION") &&
    error.message.includes("File"));

const isUnsupportedMimeTypeError = (error: Error) =>
  error.message.includes("Unsupported MIME type");

const isTokenLimitExceeded = (error: Error) =>
  "status" in error && (error as { status: number }).status === 400 &&
  error.message.includes("token count exceeds");

const isImageProcessingOrInternalError = (error: Error) =>
  error.message.includes("Unable to process input image") ||
  error.message.includes("Internal error encountered") ||
  isInvalidArgumentError(error);

const isRecoverableError = (error: Error) =>
  isFileNotActiveError(error) ||
  isUnsupportedMimeTypeError(error) ||
  isImageProcessingOrInternalError(error) ||
  is403PermissionError(error) ||
  isTokenLimitExceeded(error);

export const capEventsToTokenBudget = (maxTokens: number) =>
async (
  events: GeminiHistoryEvent[],
): Promise<GeminiHistoryEvent[]> => {
  const tokenCounts = await map(estimateTokens)(events);
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

export const getUnprocessableMediaText = (attachment: MediaAttachment) =>
  ` <media file could not be processed: ${
    attachment.caption || attachment.mimeType
  }>`;

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

const modelCallTimeoutMs: Injection<() => number> = context(() => 60_000);

export const injectGeminiModelCallTimeoutMs = modelCallTimeoutMs.inject;

const errorDetails = (error: unknown) => {
  const status = (error && typeof error === "object" && "status" in error)
    ? (error as { status: unknown }).status
    : undefined;
  const name = (error instanceof Error) ? error.name : typeof error;
  const message = (error instanceof Error)
    ? error.message.slice(0, 200)
    : String(error).slice(0, 200);
  return { status, name, message };
};

const withTimeout = <Args extends unknown[], Result>(
  fn: (signal: AbortSignal, ...args: Args) => Promise<Result>,
) =>
(...args: Args): Promise<Result> =>
  new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const controller = new AbortController();
    const timeoutMs = modelCallTimeoutMs.access();
    const timer = setTimeout(() => {
      console.warn(
        `[gemini-step] model-call-timeout after ${timeoutMs}ms`,
      );
      controller.abort();
      const err = new Error("Model call timed out");
      Object.assign(err, { status: 503, [syntheticTimeoutMarker]: true });
      reject(err);
    }, timeoutMs);
    logGemini("[gemini-step] rawCallGemini-start");
    fn(controller.signal, ...args).then(
      (result) => {
        clearTimeout(timer);
        console.log(
          `[gemini-step] rawCallGemini-ok elapsedMs=${Date.now() - startedAt}`,
        );
        resolve(result);
      },
      (error) => {
        clearTimeout(timer);
        const { status, name, message } = errorDetails(error);
        const logFn = isRecoverableError(normalizeError(error))
          ? logGemini
          : console.warn;
        logFn(
          `[gemini-step] rawCallGemini-error elapsedMs=${
            Date.now() - startedAt
          } status=${String(status)} name=${name} msg=${message}`,
        );
        reject(error);
      },
    );
  });

const requestDiag = (
  req: GenerateContentParameters,
  disableStreaming: boolean | undefined,
) => {
  const firstTool = req.config?.tools?.[0];
  const decls = firstTool && "functionDeclarations" in firstTool
    ? (firstTool.functionDeclarations ?? [])
    : [];
  const sysInstr = typeof req.config?.systemInstruction === "string"
    ? req.config.systemInstruction
    : "";
  const contents = Array.isArray(req.contents) ? req.contents : [];
  // Cheap size estimate: summing part payload lengths avoids serializing the
  // whole request (inline base64 blobs make JSON.stringify cost megabytes per call).
  const estimatePartChars = (p: Part): number =>
    (typeof p.text === "string" ? p.text.length : 0) +
    (p.inlineData?.data?.length ?? 0);
  const reqChars = contents.reduce(
    (n, c) =>
      n +
      (c && typeof c === "object" && "parts" in c && Array.isArray(c.parts)
        ? c.parts.reduce((m, p) => m + estimatePartChars(p), 0)
        : 0),
    0,
  );
  logGemini(
    `[gemini-diag] pre-call model=${req.model} mode=${
      disableStreaming ? "buffered" : "stream"
    } contents=${contents.length} tools=${decls.length} sysInstrLen=${sysInstr.length} reqChars=${reqChars}`,
  );
};

const usageDiag = (
  mode: "buffered" | "stream",
  usage: TokenUsage | undefined,
  finishReason: string | undefined,
  parts: Part[],
) => {
  const fnCalls = parts.filter((p) => p.functionCall).length;
  const thoughtChars = parts
    .filter((p) => p.thought && typeof p.text === "string")
    .reduce((n, p) => n + (p.text?.length ?? 0), 0);
  const textChars = parts
    .filter((p) => !p.thought && typeof p.text === "string" && !p.functionCall)
    .reduce((n, p) => n + (p.text?.length ?? 0), 0);
  const sigCount = parts.filter((p) => p.thoughtSignature).length;
  logGemini(
    `[gemini-diag] post-call mode=${mode} finish=${finishReason} parts=${parts.length} fnCalls=${fnCalls} thoughtChars=${thoughtChars} textChars=${textChars} sigCount=${sigCount} promptTok=${usage?.promptTokenCount} candTok=${usage?.candidatesTokenCount} thoughtTok=${usage?.thoughtsTokenCount} cachedTok=${usage?.cachedContentTokenCount}`,
  );
};

const logFunctionCallsMissingThoughtSignature = (
  model: string,
  finishReason: string | undefined,
  parts: Part[],
) => {
  const missing = parts
    .map((part, index) => ({ part, index }))
    .filter(({ part }) => part.functionCall && !part.thoughtSignature?.trim())
    .map(({ part, index }) => ({
      index,
      name: part.functionCall?.name,
      id: part.functionCall?.id,
      argBytes: part.functionCall?.args
        ? JSON.stringify(part.functionCall.args).length
        : 0,
    }));
  if (empty(missing)) return;
  console.warn(
    `[gemini-diag] response function calls missing thoughtSignature model=${model} finish=${finishReason} calls=${
      JSON.stringify(missing)
    }`,
  );
};

const withAbortSignal = (
  signal: AbortSignal,
  req: GenerateContentParameters,
): GenerateContentParameters => ({
  ...req,
  config: { ...(req.config ?? {}), abortSignal: signal },
});

type GeminiSdkExchangeResult = {
  parts: Part[];
  finishReason?: string;
  usageMetadata?: TokenUsage;
  promptBlockReason?: string;
};

const geminiSdkExchange = async (
  signal: AbortSignal,
  { req: rawReq, disableStreaming }: {
    req: GenerateContentParameters;
    disableStreaming?: boolean;
  },
): Promise<GeminiSdkExchangeResult> => {
  const req = withAbortSignal(signal, rawReq);
  requestDiag(req, disableStreaming);
  const handleStreamChunk = getStreamChunk();
  const handleStreamThinkingChunk = getStreamThinkingChunk();
  const sdk = new GoogleGenAI({ apiKey: accessGeminiToken() });

  const rawSysInstr = req.config?.systemInstruction;
  const rawTools = req.config?.tools as {
    functionDeclarations?: FunctionDeclaration[];
  }[] | undefined;
  const rawToolConfig = req.config?.toolConfig;
  let executionReq = req;
  let usedCacheName: string | null = null;

  if (
    isGeminiContextCachingEnabled() &&
    typeof rawSysInstr === "string" &&
    !req.config?.cachedContent
  ) {
    const cacheName = await getOrCreateGeminiContextCache(
      sdk,
      req.model,
      rawSysInstr,
      rawTools,
      rawToolConfig,
    );
    if (cacheName) {
      usedCacheName = cacheName;
      const {
        systemInstruction: _removedSys,
        tools: _removedTools,
        toolConfig: _removedToolConfig,
        ...restConfig
      } = req.config ?? {};
      executionReq = {
        ...req,
        config: {
          ...restConfig,
          cachedContent: cacheName,
        },
      };
    }
  }

  let finalUsageMetadata: TokenUsage | undefined;
  let finalFinishReason: string | undefined;
  let promptBlockReason: string | undefined;
  const accumulatedParts: Part[] = [];

  try {
    if (disableStreaming) {
      const response = await sdk.models.generateContent(executionReq);
      finalUsageMetadata = response.usageMetadata;
      finalFinishReason = response.candidates?.[0]?.finishReason;
      promptBlockReason = response.promptFeedback?.blockReason;
      const parts = response.candidates?.[0]?.content?.parts ?? [];
      usageDiag("buffered", finalUsageMetadata, finalFinishReason, parts);
      logFunctionCallsMissingThoughtSignature(
        req.model,
        finalFinishReason,
        parts,
      );
      for (const part of parts) {
        if (
          typeof part.text === "string" && !part.thought
        ) {
          await handleStreamChunk(part.text);
        }
        if (typeof part.text === "string" && part.thought) {
          await handleStreamThinkingChunk(part.text);
        }
        accumulatedParts.push(part);
      }
    } else {
      logGemini("[gemini-step] stream-await-start");
      const responseStream = await sdk.models.generateContentStream(
        executionReq,
      );
      logGemini("[gemini-step] stream-await-ok");

      let chunkCount = 0;
      for await (const chunk of responseStream) {
        chunkCount++;
        if (chunkCount === 1) logGemini("[gemini-step] stream-first-chunk");
        if (chunk.usageMetadata) {
          finalUsageMetadata = chunk.usageMetadata;
        }
        if (chunk.promptFeedback?.blockReason) {
          promptBlockReason = chunk.promptFeedback.blockReason;
        }
        const chunkFinishReason = chunk.candidates?.[0]?.finishReason;
        if (chunkFinishReason) finalFinishReason = chunkFinishReason;
        const parts = chunk.candidates?.[0]?.content?.parts ?? [];
        for (const part of parts) {
          if (
            typeof part.text === "string" && !part.thought &&
            !part.thoughtSignature
          ) {
            await handleStreamChunk(part.text);
          }
          if (typeof part.text === "string" && part.thought) {
            await handleStreamThinkingChunk(part.text);
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
      logGemini(`[gemini-step] stream-done chunks=${chunkCount}`);
      usageDiag(
        "stream",
        finalUsageMetadata,
        finalFinishReason,
        accumulatedParts,
      );
      logFunctionCallsMissingThoughtSignature(
        req.model,
        finalFinishReason,
        accumulatedParts,
      );
    }
  } catch (err: unknown) {
    if (usedCacheName && typeof rawSysInstr === "string") {
      const errorMsg = String(err);
      if (
        errorMsg.includes("NOT_FOUND") ||
        errorMsg.includes("404") ||
        errorMsg.includes("cachedContent") ||
        errorMsg.includes("Cached content")
      ) {
        await invalidateGeminiContextCache(req.model, rawSysInstr, rawTools);
        return await geminiSdkExchange(signal, {
          req: rawReq,
          disableStreaming,
        });
      }
    }
    throw err;
  }

  return {
    parts: accumulatedParts,
    finishReason: finalFinishReason,
    usageMetadata: finalUsageMetadata,
    promptBlockReason,
  };
};

const geminiSdkExchangeInjection: Injection<typeof geminiSdkExchange> = context(
  geminiSdkExchange,
);

// Test seam: script the exact Gemini exchange (parts + finishReason) without
// hitting the API, e.g. to reproduce malformed function-call responses.
export const injectGeminiSdkExchange = geminiSdkExchangeInjection.inject;

const rawCallGemini = async (
  signal: AbortSignal,
  args: {
    req: GenerateContentParameters;
    disableStreaming?: boolean;
  },
): Promise<GeminiOutput> => {
  const { parts, finishReason, usageMetadata, promptBlockReason } =
    await geminiSdkExchangeInjection.access(signal, args);

  if (usageMetadata) {
    tokenUsage.access(usageMetadata, args.req.model);
  }

  // A prompt-level block (`promptFeedback.blockReason`) returns zero candidates,
  // so `finishReason` is undefined and the turn yields no parts. Route it
  // through the same sink the candidate `finishReason` uses.
  const isPromptBlock = !!promptBlockReason && empty(parts);
  if (isPromptBlock) {
    promptBlocked.access(promptBlockReason, args.req);
  }
  const effectiveFinishReason = finishReason ??
    (isPromptBlock ? promptBlockReasonPrefix + promptBlockReason : undefined);

  if (effectiveFinishReason) {
    finishReasonSink.access(effectiveFinishReason);
  }

  rejectMalformedFunctionCall(finishReason, parts);

  return parts.flatMap((part: Part): GeminiOutput => {
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

const callGeminiWithRetry = conditionalRetryExponential(isRetryableError)(
  1000,
  16000,
  5,
  withTimeout(rawCallGemini),
);

const fallbackModelRetry = conditionalRetryExponential(isRetryableError)(
  1000,
  16000,
  4,
  withTimeout(rawCallGemini),
);

const callGemini = (
  req: GenerateContentParameters,
  disableStreaming?: boolean,
): Promise<GeminiOutput> =>
  callGeminiWithRetry({ req, disableStreaming }).catch((err: unknown) => {
    // INVALID_ARGUMENT is nominally a client error, but Gemini intermittently
    // rejects valid requests with it during serving incidents — one attempt on
    // the alternate model recovers those runs; a genuinely malformed request
    // just 400s once more and propagates.
    if (
      !isRetryableError(err) && !isSyntheticTimeoutError(err) &&
      !isInvalidArgumentError(normalizeError(err))
    ) throw err;
    return fallbackModelRetry({
      req: {
        ...req,
        model: alternateGeminiModelVersion(req.model),
      },
      disableStreaming,
    });
  });

const actionToTool = ({ name, description, parameters }: Tool<ZodType>) => ({
  name,
  description,
  parameters: zodToGeminiParameters(parameters),
});

const optionalThoughtSignature = (sig: string | undefined) =>
  sig ? { thoughtSignature: sig } : {};

const attachmentsToPartsOrEmpty = (attachments?: MediaAttachment[]): Part[] =>
  attachmentsToParts(attachments ?? []);

const referencedMessageText =
  (eventById: (id: string) => GeminiHistoryEvent | undefined) =>
  (onMessage: MessageId): string => {
    const msg = eventById(onMessage);
    return typeof msg === "object" && "text" in msg &&
        typeof msg.text === "string"
      ? msg.text
      : "";
  };

const isInspectMediaToolResult = (
  eventById: (id: MessageId) => GeminiHistoryEvent | undefined,
) =>
(event: GeminiHistoryEvent) => {
  const toolCall = event.type === "tool_result" && event.toolCallId
    ? eventById(event.toolCallId)
    : undefined;
  return toolCall?.type === "tool_call" &&
    toolCall.name === inspectMediaUrlToolName;
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
    // When the utterance has no real content we emit a `" "` placeholder. A
    // thoughtSignature must travel with the exact text Gemini signed, so it must
    // NOT be attached to this synthesized placeholder (Gemini can reject a
    // signature on content it never produced).
    return wrapModelContent(
      !empty(parts) ? parts : [{ text: " " }],
    );
  }
  if (e.type === "tool_call") {
    return wrapModelContent([{
      ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
      functionCall: {
        id: e.id,
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
          id: e.toolCallId,
          name,
          response: {
            result: stampText(stripAnsi(e.result)),
          },
        },
      },
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
      ? wrapModelContent([{ text: " " }])
      : wrapUserContent([
        {
          text: isCompactedSummaryText(e.text)
            ? stampText(e.text)
            : stampText(`${systemNotificationPrefix} ${e.text}]`),
        },
        ...attachmentsToPartsOrEmpty(e.attachments),
      ]);
  }
  if (e.type === "external_event") {
    return wrapUserContent([
      { text: stampText(`${externalEventPrefix} ${e.text}]`) },
      ...attachmentsToPartsOrEmpty(e.attachments),
    ]);
  }
  if (e.type === "own_reaction") {
    return wrapModelContent([{
      ...optionalThoughtSignature(e.modelMetadata?.thoughtSignature),
      text: `You reacted ${e.reaction}`,
    }]);
  }
  if (e.type === "participant_reaction") {
    return wrapUserContent([{
      text: `${e.name} reacted ${e.reaction}`,
    }]);
  }
  if (e.type === "do_nothing") {
    // `do_nothing` renders as a synthesized `" "` placeholder. Do not attach a
    // thoughtSignature to it — the signature belongs to the exact text Gemini
    // produced, not to this placeholder, and Gemini can reject the mismatch.
    return wrapModelContent([{ text: " " }]);
  }
  throw new Error(
    `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
  );
};

// A thoughtSignature is only valid on the exact part Gemini returned it on.
// Round-tripping means every part must carry back precisely its own signature —
// no more. The previous implementation smeared one part's signature onto ALL
// parts in the combined content, which attaches signatures to synthesized
// placeholder text parts (e.g. the " " emitted for a signature-less thought or
// do_nothing) and to plain utterances that never had one. Gemini rejects a
// signature on such a part with 400 INVALID_ARGUMENT.
//
// The only legitimate repair is for a functionCall part that lost its own
// signature during grouping: it may inherit a sibling's signature, since a
// functionCall always requires one. Everything else keeps exactly its own.
const isFunctionCallPart = (part: Part): boolean =>
  "functionCall" in part && !!part.functionCall;

const combineContent = (contents: Content[]): Content => {
  const parts = contents.flatMap((c) => c.parts ?? []);
  const siblingSignature = parts.find((p) => p.thoughtSignature)
    ?.thoughtSignature;
  const repair = (part: Part): Part =>
    isFunctionCallPart(part) && !part.thoughtSignature && siblingSignature
      ? { ...part, thoughtSignature: siblingSignature }
      : part;
  return {
    role: contents.some((c) => c.role === "model") ? "model" : "user",
    parts: parts.map(repair),
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

const fixEnd = (history: Content[]) => {
  if (empty(history)) return history;
  const last = history[history.length - 1];
  if (last.role === "model") {
    return [
      ...history,
      { role: "user", parts: [{ text: "<continue>" }] },
    ];
  }
  return history;
};

const mergeConsecutiveRoles = (contents: Content[]): Content[] => {
  if (empty(contents)) return [];
  const result: Content[] = [{ ...contents[0] }];
  for (let i = 1; i < contents.length; i++) {
    const last = result[result.length - 1];
    const current = contents[i];
    if (last.role === current.role) {
      last.parts = [...(last.parts ?? []), ...(current.parts ?? [])];
    } else {
      result.push({ ...current });
    }
  }
  return result;
};

// With no function declarations, force `NONE` so the model cannot emit a
// (hallucinated) function call by mimicking function_call parts already in
// history — it must answer with text. With declarations present, leave the
// config empty (AUTO) so the model decides.
const toolingConfig = (tools: Tool<ZodType>[]) =>
  empty(tools)
    ? {
      toolConfig: {
        functionCallingConfig: { mode: FunctionCallingConfigMode.NONE },
      },
    }
    : {
      tools: [{ functionDeclarations: tools.map(actionToTool) }],
      toolConfig: { functionCallingConfig: {} },
    };

const textSafetyCategories = [
  HarmCategory.HARM_CATEGORY_HARASSMENT,
  HarmCategory.HARM_CATEGORY_HATE_SPEECH,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
  HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
];

export const buildReq = (
  lightModel: boolean | undefined,
  prompt: string,
  tools: Tool<ZodType>[],
  timezoneIANA: string,
  maxOutputTokens: number | undefined,
) =>
(events: GeminiHistoryEvent[]): GenerateContentParameters => ({
  model: geminiModelVersion(lightModel),
  config: {
    systemInstruction: prompt,
    safetySettings: textSafetyCategories.map((category) => ({
      category,
      threshold: HarmBlockThreshold.OFF,
    })),
    ...toolingConfig(tools),
    thinkingConfig: geminiThinkingConfig(lightModel),
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
    mergeConsecutiveRoles,
    fixStart,
    fixEnd,
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

export type GeminiMetadata = {
  type: "gemini";
  thoughtSignature?: string;
  responseId?: string;
  isSafetyBlock?: boolean;
};

export type GeminiHistoryEvent = HistoryEventWithMetadata<GeminiMetadata>;

export type GeminiPartOfInterest =
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
      p.text.replace(/['"\s\u200B\u200C\u200D\uFEFF\u200E\u200F]/g, "")) ||
    p.type === "inline_data" ||
    p.type === "file_data"
  );

// MIME types Gemini rejects with "Unsupported MIME type". Keeping this list
// explicit (rather than waiting for the API to 400 us) lets us strip the
// attachment once, up front, and persist the rewrite via `rewriteHistory` on
// the first call — which matters for cached test runs where the reactive
// error path in `stripAllUnsupportedMimeTypes` never fires.
const knownUnsupportedGeminiMimeTypes = new Set<string>([
  "application/octet-stream",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "application/vnd.ms-excel",
  "application/vnd.ms-powerpoint",
  "application/msword",
]);

const isUnsupportedGeminiMimeType = (mimeType: string | undefined): boolean => {
  if (!mimeType) return true;
  const normalized = mimeType.trim().toLowerCase();
  if (!normalized) return true;
  return knownUnsupportedGeminiMimeTypes.has(normalized);
};

const filterDoNothing = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => history.filter((e) => e.type !== "do_nothing");

const stripUnsupportedAttachmentsFromEvent = (
  event: GeminiHistoryEvent,
): { event: GeminiHistoryEvent; changed: boolean } => {
  if (!("attachments" in event)) return { event, changed: false };
  const attachments = event.attachments ?? [];
  if (empty(attachments)) return { event, changed: false };
  const kept = attachments.filter((att) =>
    !isUnsupportedGeminiMimeType(att.mimeType)
  );
  if (kept.length === attachments.length) return { event, changed: false };
  const removed = attachments.filter((att) =>
    isUnsupportedGeminiMimeType(att.mimeType)
  );
  console.warn(
    `Warning: Filtering out unsupported Gemini attachment mime types on event ${event.id}: ${
      removed.map((att) => att.mimeType).join(", ")
    }`,
  );
  return {
    event: { ...event, attachments: empty(kept) ? undefined : kept },
    changed: true,
  };
};

const filterUnsupportedGeminiAttachments = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] =>
  history.map((event) => stripUnsupportedAttachmentsFromEvent(event).event);

// Same stripping as `filterUnsupportedGeminiAttachments` but records
// replacements so callers can persist them via `rewriteHistory`. Runs outside
// the cached `callModel` boundary so side effects fire even on cache hits.
export const filterAndRewriteUnsupportedGeminiAttachments =
  (rewriteHistory?: AgentSpec["rewriteHistory"]) =>
  async (history: GeminiHistoryEvent[]): Promise<GeminiHistoryEvent[]> => {
    const replacements: Record<string, GeminiHistoryEvent> = {};
    const result = history.map((event) => {
      const { event: stripped, changed } = stripUnsupportedAttachmentsFromEvent(
        event,
      );
      if (changed) replacements[event.id] = stripped;
      return stripped;
    });
    if (rewriteHistory && !empty(Object.keys(replacements))) {
      await rewriteHistory(replacements);
    }
    return result;
  };

export const filterOrphanedToolResults = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => {
  const unconsumedCallIds = new Set(
    history.filter((e) => e.type === "tool_call").map((e) => e.id),
  );
  return history.filter((e) => {
    if (e.type !== "tool_result") return true;
    if (!e.toolCallId || !unconsumedCallIds.has(e.toolCallId)) {
      console.warn(
        `Warning: Filtering out orphaned tool_result (id: ${e.id}, toolCallId: ${e.toolCallId}). ` +
          `No unclaimed matching tool_call found with that ID.`,
      );
      return false;
    }
    // Each tool_call claims exactly one result; extra results for the same
    // call are orphans.
    unconsumedCallIds.delete(e.toolCallId);
    return true;
  });
};

const eventHasThoughtSignature = (e: GeminiHistoryEvent): boolean =>
  !!("modelMetadata" in e && e.modelMetadata?.thoughtSignature?.trim());

export const filterInvalidToolCalls = (
  history: GeminiHistoryEvent[],
): GeminiHistoryEvent[] =>
  history.filter((e) => {
    if (e.type === "tool_call" && !eventHasThoughtSignature(e)) {
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
  text: `I previously called the tool ${
    "name" in e ? e.name : "unknown"
  } with parameters: ${JSON.stringify("parameters" in e ? e.parameters : {})}`,
});

const toolResultToOwnThought = (e: GeminiHistoryEvent): GeminiHistoryEvent => ({
  type: "own_thought",
  isOwn: true,
  id: e.id,
  timestamp: e.timestamp,
  text: `The tool returned: ${"result" in e ? e.result : ""}`,
});

const textToOwnThought = (e: GeminiHistoryEvent): GeminiHistoryEvent => ({
  type: "own_thought",
  isOwn: true,
  id: e.id,
  timestamp: e.timestamp,
  text: "text" in e ? (e.text as string) : "",
});

const computeInvalidToolCallReplacements = (
  history: GeminiHistoryEvent[],
): {
  filtered: GeminiHistoryEvent[];
  replacements: Record<string, GeminiHistoryEvent>;
} => {
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
    const hasSignature = toolCalls.some(eventHasThoughtSignature);
    if (!hasSignature) taintedResponseIds.add(responseId);
  }

  if (taintedResponseIds.size === 0) {
    return { filtered: history, replacements: {} };
  }

  const isTainted = (e: GeminiHistoryEvent) =>
    taintedResponseIds.has(getOriginalId(e));

  const replacements: Record<string, GeminiHistoryEvent> = {};
  const filtered = history.filter((e) => {
    if (e.type === "tool_call" && isTainted(e)) {
      replacements[e.id] = toolCallToOwnThought(e);
      return false;
    }
    if (e.type === "tool_result" && "toolCallId" in e && e.toolCallId) {
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

  return { filtered, replacements };
};

// Synchronous filter used inside the cached provider caller so the filter is
// applied deterministically on every run (including on cache hits inside the
// inner call, though in practice the pre-filter runs first and this is a
// no-op). The `rewriteHistory` side-effect is fire-and-forget here because it
// has already been awaited outside the cache boundary in
// `prepareGeminiHistory`.
export const filterAndRewriteInvalidToolCalls =
  (rewriteHistory?: AgentSpec["rewriteHistory"]) =>
  (history: GeminiHistoryEvent[]): GeminiHistoryEvent[] => {
    const { filtered, replacements } = computeInvalidToolCallReplacements(
      history,
    );
    if (rewriteHistory && !empty(Object.keys(replacements))) {
      rewriteHistory(replacements).catch((err) =>
        console.warn("Failed to rewrite history for invalid tool calls:", err)
      );
    }
    return filtered;
  };

// Async variant invoked OUTSIDE the cached `callModel` boundary so the
// `rewriteHistory` side effect runs even when the inner call is served from
// the rmmbr cache. Production flows through here too; making it await means
// downstream code can rely on the persisted history being up to date.
export const filterAndRewriteInvalidToolCallsAsync =
  (rewriteHistory?: AgentSpec["rewriteHistory"]) =>
  async (history: GeminiHistoryEvent[]): Promise<GeminiHistoryEvent[]> => {
    const { filtered, replacements } = computeInvalidToolCallReplacements(
      history,
    );
    if (rewriteHistory && !empty(Object.keys(replacements))) {
      await rewriteHistory(replacements);
    }
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
  rewriteHistory?: AgentSpec["rewriteHistory"],
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
      if (rewriteHistory) await rewriteHistory(allReplacements);
      return result;
    } catch (error) {
      const err = normalizeError(error);
      if (!isUnsupportedMimeTypeError(err)) throw err;
      currentError = err;
    }
  }
  return callGemini(eventsToRequest(currentEvents), disableStreaming);
};

const getCorruptedMediaText = (attachments: MediaAttachment[]) =>
  !empty(attachments)
    ? ` <media file corrupted or unsupported: ${
      attachments.map((a: MediaAttachment) => a.caption || a.mimeType)
        .join(", ")
    }>`
    : "";

const stripAllCorruptedFileAttachments = (
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
      const placeholder = getCorruptedMediaText(fileAttachments);
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

const stripAllCorruptedFileAttachmentsAndRetry = async (
  originalError: Error,
  events: GeminiHistoryEvent[],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters,
  rewriteHistory?: AgentSpec["rewriteHistory"],
  disableStreaming?: boolean,
): Promise<GeminiOutput> => {
  console.warn(
    "Stripping all file attachments due to image processing or internal error:",
    originalError.message,
  );
  const nuclear = stripAllCorruptedFileAttachments(events);
  if (empty(Object.keys(nuclear.replacements))) {
    throw originalError;
  }
  const result = await callGemini(
    eventsToRequest(nuclear.updatedHistory),
    disableStreaming,
  );
  if (rewriteHistory) await rewriteHistory(nuclear.replacements);
  return result;
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
  rewriteHistory?: AgentSpec["rewriteHistory"],
  disableStreaming?: boolean,
): Promise<GeminiOutput> => {
  let currentEvents = events;
  let currentError = initialError;
  const allReplacements: Record<string, GeminiHistoryEvent> = {};
  for (let attempt = 0; attempt < 20; attempt++) {
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
        if (rewriteHistory) await rewriteHistory(allReplacements);
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
      if (rewriteHistory) await rewriteHistory(allReplacements);
      return result;
    } catch (error) {
      const err = normalizeError(error);
      if (!isFileNotActiveError(err)) throw err;
      currentError = err;
    }
  }
  return callGemini(eventsToRequest(currentEvents), disableStreaming);
};

export const callGeminiWithFixHistory = (
  rewriteHistory?: AgentSpec["rewriteHistory"],
  eventsToRequest: (events: GeminiHistoryEvent[]) => GenerateContentParameters =
    buildReq(false, "", [], "UTC", undefined),
  disableStreaming?: boolean,
) =>
async (events: GeminiHistoryEvent[]): Promise<GeminiOutput> => {
  try {
    const req = eventsToRequest(events);
    try {
      return await callGemini(req, disableStreaming);
    } catch (error) {
      const err = normalizeError(error);
      if (isTokenLimitExceeded(err)) {
        const totalTokens = sum(await map(estimateTokens)(events));
        throw new Error(
          `Token limit exceeded (estimated ${totalTokens} tokens, ${events.length} events). This should never happen due to history compaction.`,
        );
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
      if (isImageProcessingOrInternalError(err)) {
        return stripAllCorruptedFileAttachmentsAndRetry(
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

export const safetyWarningText =
  "I am sorry, but I cannot fulfill this request as it violates content safety guidelines.";

const maxHistoryTokens = 800_000;

const noResponseInstruction =
  `\n\nWhen you have nothing to say (e.g. the message is irrelevant), respond with exactly ${noResponseTag} and nothing else.`;

const enhancePrompt = (
  prompt: string,
  toolOutputScratchPad?: AgentSpec["toolOutputScratchPad"],
) => `${prompt}\n\n${systemInstructionTail(toolOutputScratchPad)}`;

// Side-effectful history normalization that MUST run outside the cached
// `callModel` boundary. Without this, tests replay a populated rmmbr cache
// and never see the underlying provider call — meaning the `rewriteHistory`
// calls buried inside the Gemini caller silently skip. The same logic is
// still applied inside `geminiAgentCaller` for correctness during cache
// misses / production; the pre-filter here makes those paths idempotent
// no-ops while guaranteeing the rewrite is persisted on every call.
// Rehydrates `modelMetadata` on events that lack it (e.g. re-read from Deno
// KV in prompt2bot, where we strip metadata before write to stay under the
// 64KB value cap). Events that already carry inline `modelMetadata` are left
// untouched — inline data is authoritative for the current run and may be
// fresher than whatever rmmbr has persisted.
const enrichGeminiEventsWithMetadata = async (
  events: HistoryEventWithMetadata<GeminiMetadata>[],
): Promise<HistoryEventWithMetadata<GeminiMetadata>[]> => {
  const eventIds = events.map((e) => e.id);
  const metadataList = await accessMetadataStore().mget(eventIds);
  return events.map((event, i) => {
    if ("modelMetadata" in event && event.modelMetadata) return event;
    const metadata = metadataList[i] as GeminiMetadata | null;
    if (!metadata) return event;
    return { ...event, modelMetadata: metadata };
  });
};

type ResolvedAttachment =
  | { resolved: MediaAttachment }
  | { unprocessable: MediaAttachment };

const isResolved = (
  r: ResolvedAttachment,
): r is { resolved: MediaAttachment } => "resolved" in r;

const isUnprocessable = (
  r: ResolvedAttachment,
): r is { unprocessable: MediaAttachment } => "unprocessable" in r;

const resolveSingleAttachment =
  (event: GeminiHistoryEvent, events: GeminiHistoryEvent[]) =>
  async (att: MediaAttachment): Promise<ResolvedAttachment> => {
    const needsUpload = (att.kind === "file" && !isGeminiFileUri(att.fileUri) &&
      !(event.type === "tool_result" &&
        !isInspectMediaToolResult(indexById(events))(event))) ||
      att.kind === "inline";
    if (!needsUpload) return { resolved: att };
    try {
      return { resolved: await ensureGeminiAttachmentIsLink(att) };
    } catch (error) {
      console.warn(
        `Degrading un-uploadable attachment on event ${event.id} to placeholder: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
      return { unprocessable: att };
    }
  };

const degradeEventWithUnprocessableAttachments = (
  event: GeminiHistoryEvent & { attachments?: MediaAttachment[] },
  resolved: ResolvedAttachment[],
): GeminiHistoryEvent => {
  const placeholder = resolved.filter(isUnprocessable)
    .map((r) => getUnprocessableMediaText(r.unprocessable))
    .join("");
  const attachments = resolved.filter(isResolved).map((r) => r.resolved);
  return {
    ...event,
    ...event.type === "tool_result"
      ? { result: event.result + placeholder }
      : { text: ((event as { text?: string }).text ?? "") + placeholder },
    attachments: empty(attachments) ? undefined : attachments,
  } as GeminiHistoryEvent;
};

const resolveAttachments = async (
  events: GeminiHistoryEvent[],
): Promise<
  {
    updatedHistory: GeminiHistoryEvent[];
    replacements: Record<string, GeminiHistoryEvent>;
  }
> => {
  const replacements: Record<string, GeminiHistoryEvent> = {};
  const updatedHistory = await Promise.all(
    events.map(async (event) => {
      if (!("attachments" in event) || !event.attachments) return event;
      const resolved = await Promise.all(
        event.attachments.map(resolveSingleAttachment(event, events)),
      );
      if (resolved.every(isResolved)) {
        return { ...event, attachments: resolved.map((r) => r.resolved) };
      }
      const degraded = degradeEventWithUnprocessableAttachments(
        event,
        resolved,
      );
      replacements[event.id] = degraded;
      return degraded;
    }),
  );
  return { updatedHistory, replacements };
};

export const prepareGeminiHistory =
  (rewriteHistory?: AgentSpec["rewriteHistory"]) =>
  async (
    events: HistoryEventWithMetadata<GeminiMetadata>[],
  ): Promise<HistoryEventWithMetadata<GeminiMetadata>[]> => {
    const safeRewrite = rewriteHistory ?? (() => Promise.resolve());
    const enriched = await enrichGeminiEventsWithMetadata(events);
    const filtered = await pipe(
      filterAndRewriteInvalidToolCallsAsync(safeRewrite),
      filterAndRewriteUnsupportedGeminiAttachments(safeRewrite),
    )(enriched);
    const { updatedHistory, replacements } = await resolveAttachments(filtered);
    if (!empty(Object.keys(replacements))) await safeRewrite(replacements);
    return updatedHistory;
  };

const geminiMaxTokensReason = "MAX_TOKENS";

const markTruncatedUtterances = (
  events: GeminiHistoryEvent[],
): GeminiHistoryEvent[] => {
  const hasOwnUtterance = events.some((e) => e.type === "own_utterance");
  if (!hasOwnUtterance) {
    const responseId = generateId();
    const warningEvent: OwnUtterance<GeminiMetadata> =
      ownUtteranceTurnWithMetadata(
        thinkingTokenExhaustionWarningText,
        {
          type: "gemini",
          responseId,
        },
      );
    return [
      ...events.filter((e) => e.type !== "do_nothing"),
      { ...warningEvent, truncated: true },
    ];
  }
  return events.map((e) =>
    e.type === "own_utterance" ? { ...e, truncated: true } : e
  );
};

// Marks a reason that came from Gemini's `promptFeedback.blockReason` (the whole
// prompt was rejected, zero candidates) as opposed to a candidate
// `finishReason`. Any prompt-level block means the model refused to produce
// output, so it must always surface as a user-facing safety message rather than
// a silent `do_nothing`.
export const promptBlockReasonPrefix = "PROMPT_BLOCK:";

export const isSafetyBlockReason = (reason: string | undefined): boolean => {
  if (!reason) return false;
  if (reason.startsWith(promptBlockReasonPrefix)) return true;
  const upper = reason.toUpperCase();
  return (
    upper === "SAFETY" ||
    upper === "RECITATION" ||
    upper === "BLOCKLIST" ||
    upper === "PROHIBITED_CONTENT" ||
    upper === "SPII"
  );
};

const eventText = (event: GeminiHistoryEvent): string =>
  "text" in event && typeof event.text === "string"
    ? event.text
    : "result" in event && typeof event.result === "string"
    ? event.result
    : "";

// Gemini-specific: reject a model turn that rewrote the conversation's language
// into a different writing system (homoglyph corruption). The reference is the
// system prompt plus all prior turn text; the checked output is the text this
// turn produced.
const guardResultScriptDrift = async (
  prompt: string,
  inputEvents: GeminiHistoryEvent[],
  result: GeminiHistoryEvent[],
): Promise<void> => {
  const producedText = result.map(eventText).join("\n").trim();
  if (!producedText) return;
  const reference = [prompt, ...inputEvents.map(eventText)].join("\n");
  await assertNoScriptDrift(reference, producedText);
};

const isScriptDriftError = (e: unknown): boolean =>
  e instanceof Error && "scriptDrift" in e;

// Homoglyph corruption is a transient, low-frequency model glitch: the same
// prompt almost always produces clean text on the next attempt. So re-roll the
// model a bounded number of times when the guard flags drift, and only surface
// the error if it persists — otherwise a one-off glitch becomes a user-facing
// failure even though a plain retry would have fixed it.
const maxScriptDriftRerolls = 2;

const callInnerWithDriftReroll = async (
  spec: AgentSpec,
  events: GeminiHistoryEvent[],
  box: { reason?: string },
): Promise<GeminiHistoryEvent[]> => {
  for (let attempt = 0; attempt <= maxScriptDriftRerolls; attempt++) {
    const result = await finishReasonSink.inject((r: string) => {
      box.reason = r;
    })(() => geminiAgentCallerInner(spec)(events))();
    try {
      await guardResultScriptDrift(spec.prompt, events, result);
      return result;
    } catch (e) {
      if (!isScriptDriftError(e) || attempt === maxScriptDriftRerolls) throw e;
      console.warn(
        `[script-drift] detected drift in model response (attempt ${
          attempt + 1
        }/${maxScriptDriftRerolls + 1}); re-rolling`,
      );
    }
  }
  throw new Error("unreachable: script drift re-roll loop exited");
};

export const geminiAgentCaller =
  (spec: AgentSpec) =>
  async (events: GeminiHistoryEvent[]): Promise<GeminiHistoryEvent[]> => {
    const totalTokens = await estimateAgentInputTokens(spec, events);
    if (totalTokens > 1040000) {
      throw new Error(
        `Token budget exceeded! Estimated ${totalTokens} tokens, limit is 1040000. This should never happen due to history compaction.`,
      );
    }

    const box: { reason?: string } = {};
    const result = await callInnerWithDriftReroll(spec, events, box);
    if (isSafetyBlockReason(box.reason)) {
      const responseId = generateId();
      return [
        ownUtteranceTurnWithMetadata(
          safetyWarningText,
          {
            type: "gemini",
            responseId,
            isSafetyBlock: true,
          },
        ) as GeminiHistoryEvent,
      ];
    }
    return box.reason === geminiMaxTokensReason
      ? markTruncatedUtterances(result)
      : result;
  };

const geminiAgentCallerInner = ({
  lightModel,
  prompt,
  tools,
  skills,
  allSkills,
  rewriteHistory = () => Promise.resolve(),
  timezoneIANA,
  maxOutputTokens,
  disableStreaming,
  isConsult,
  toolOutputScratchPad,
}: AgentSpec) =>
(
  events: GeminiHistoryEvent[],
): Promise<GeminiHistoryEvent[]> => {
  // Suppress the `[no response]` silence license when a pending deferred
  // tool_call has the user waiting on a reply. Otherwise the light model emits
  // the no-response tag it was taught here even though a higher-authority system
  // notification (injected by normalizeHistoryForModel) tells it to answer. The
  // notification alone is not enough — the competing license must be removed too.
  const silenceLicense = isConsult ||
      historyHasPendingDeferredUserWaitingNudge(events)
    ? ""
    : noResponseInstruction;
  return pipe(
    filterAndRewriteInvalidToolCalls(rewriteHistory),
    filterOrphanedToolResults,
    filterDoNothing,
    filterUnsupportedGeminiAttachments,
    capEventsToTokenBudget(maxHistoryTokens),
    callGeminiWithFixHistory(
      rewriteHistory,
      buildReq(
        lightModel,
        `${enhancePrompt(prompt, toolOutputScratchPad)}${silenceLicense}`,
        [
          ...tools,
          ...((allSkills ?? skills ?? []).length > 0
            ? createSkillTools(allSkills ?? skills ?? [])
            : []),
        ],
        timezoneIANA,
        maxOutputTokens,
      ),
      disableStreaming,
    ),
    (geminiOutput: GeminiOutput): GeminiHistoryEvent[] =>
      geminiOutputToHistoryEvents(geminiOutput),
  )(events);
};

const embeddedThoughtPattern = new RegExp(internalThoughtMarker, "g");

export const stripEmbeddedThoughtPatterns = (text: string): string =>
  stripJsonThought(text.replace(embeddedThoughtPattern, "")).trim();

const extractEmbeddedThoughts = (text: string): string => {
  const bracketThoughts = [...text.matchAll(embeddedThoughtPattern)]
    .map((m) => m[1])
    .join("\n")
    .trim();
  const jsonThoughts = extractJsonThought(text);
  return [bracketThoughts, jsonThoughts].filter(Boolean).join("\n").trim();
};

const storeGeminiMetadata = (eventId: string, metadata: GeminiMetadata) =>
  accessMetadataStore().set(eventId, metadata).catch((e) => {
    console.error("Failed to store Gemini metadata:", e);
  });

// Returns the event with inline `modelMetadata` for same-run use, and also
// fire-and-forgets a persistent store so the data survives a round-trip
// through storage layers that strip `modelMetadata` to stay under size caps
// (see `prompt2bot` fitEventToKv). The inline copy is what downstream
// filters like `filterInvalidToolCalls` read.
const withPersistedMetadata = <E extends GeminiHistoryEvent>(
  event: E,
  metadata: GeminiMetadata,
): E => {
  storeGeminiMetadata(event.id, metadata);
  return { ...event, modelMetadata: metadata };
};

const geminiOutputPartToHistoryEvent =
  (responseId: string) =>
  (p: GeminiPartOfInterest): GeminiHistoryEvent | null => {
    if (p.type === "text") {
      const metadata: GeminiMetadata = {
        type: "gemini",
        responseId,
        thoughtSignature: p.thoughtSignature ?? "",
      };
      const text = collapseDuplicatedText(
        typeof p.text === "string" ? p.text : "",
      );

      const stripped = stripInternalSentTimestampSuffix(text);
      const anchoredThoughtRegex = new RegExp(
        `^${embeddedThoughtPattern.source}$`,
      );
      const match = stripped.match(anchoredThoughtRegex);

      if (match) {
        return withPersistedMetadata(
          ownThoughtTurnWithMetadata(match[1], metadata) as GeminiHistoryEvent,
          metadata,
        );
      }

      const cleanedText = stripEmbeddedThoughtPatterns(stripped);
      if (!cleanedText) {
        const embedded = extractEmbeddedThoughts(stripped);
        if (!embedded) return null;
        return withPersistedMetadata(
          ownThoughtTurnWithMetadata(embedded, metadata) as GeminiHistoryEvent,
          metadata,
        );
      }

      return withPersistedMetadata(
        (p.thought
          ? ownThoughtTurnWithMetadata(cleanedText, metadata)
          : ownUtteranceTurnWithMetadata(
            cleanedText,
            metadata,
          )) as GeminiHistoryEvent,
        metadata,
      );
    }
    if (p.type === "function_call") {
      const metadata: GeminiMetadata = {
        type: "gemini",
        responseId,
        thoughtSignature: p.thoughtSignature ?? "",
      };
      return withPersistedMetadata(
        toolUseTurnWithMetadata(
          p.functionCall,
          metadata,
        ) as GeminiHistoryEvent,
        metadata,
      );
    }
    if (p.type === "inline_data") {
      const { data, mimeType } = p.inlineData;
      const metadata: GeminiMetadata = {
        type: "gemini",
        responseId,
        thoughtSignature: data ? p.thoughtSignature ?? "" : "",
      };
      if (!data) {
        return withPersistedMetadata(
          ownUtteranceTurnWithMetadata("", metadata) as GeminiHistoryEvent,
          metadata,
        );
      }
      return withPersistedMetadata(
        ownUtteranceTurnWithMetadata("", metadata, [{
          kind: "inline",
          mimeType: mimeType ?? "application/octet-stream",
          dataBase64: data,
        }]) as GeminiHistoryEvent,
        metadata,
      );
    }
    if (p.type === "file_data") {
      const { fileUri, mimeType } = p.fileData;
      const metadata: GeminiMetadata = {
        type: "gemini",
        responseId,
        thoughtSignature: fileUri ? p.thoughtSignature ?? "" : "",
      };
      if (fileUri) {
        return withPersistedMetadata(
          ownUtteranceTurnWithMetadata("", metadata, [{
            kind: "file",
            mimeType: mimeType ?? "application/octet-stream",
            fileUri,
          }]) as GeminiHistoryEvent,
          metadata,
        );
      }
      return withPersistedMetadata(
        ownUtteranceTurnWithMetadata("", metadata) as GeminiHistoryEvent,
        metadata,
      );
    }
    throw new Error(`Unknown part type: ${JSON.stringify(p)}`);
  };

const doNothingResultEvents = (
  responseId: string,
  geminiOutput: GeminiOutput,
): GeminiHistoryEvent[] => {
  const thoughtPart = geminiOutput.find((p) => p.type === "text" && p.thought);
  const textPart = geminiOutput.find((p) =>
    p.type === "text" && p.thoughtSignature
  );
  const text =
    (thoughtPart && "text" in thoughtPart ? thoughtPart.text : undefined) ||
    (textPart && "text" in textPart ? textPart.text : undefined);
  return [doNothingEventWithMetadata(
    textPart?.thoughtSignature
      ? {
        type: "gemini",
        responseId,
        thoughtSignature: textPart.thoughtSignature,
      }
      : undefined,
    text,
  )];
};

export const geminiOutputToHistoryEvents = (
  geminiOutput: GeminiOutput,
): GeminiHistoryEvent[] => {
  const responseId = generateId();
  if (didNothing(geminiOutput)) {
    return doNothingResultEvents(responseId, geminiOutput);
  }
  return geminiOutput.flatMap((part) => {
    const event = geminiOutputPartToHistoryEvent(responseId)(part);
    return event ? [event] : [];
  });
};
