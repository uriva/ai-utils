import { context, type Injection, type Injector } from "@uri/inject";
import { conditionalRetry, empty, map, sum } from "gamla";
import { OpenAI } from "openai";
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources/index.mjs";
import type { ZodType } from "zod/v4";
import { zodToGeminiParameters } from "./gemini.ts";
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
  ownThoughtTurnWithMetadata,
  ownUtteranceTurnWithMetadata,
  type Tool,
  toolUseTurnWithMetadata,
} from "./agent.ts";
import {
  appendInternalSentTimestamp,
  stripInternalSentTimestampSuffix,
} from "./internalMessageMetadata.ts";

import { encodeBase64 } from "@std/encoding/base64";

// Fetch file attachment and convert to base64
const fetchFileAttachment = async (
  attachment: MediaAttachment,
): Promise<string | null> => {
  if (attachment.kind !== "file" || !attachment.fileUri) return null;

  try {
    const response = await fetch(attachment.fileUri);
    if (!response.ok) {
      console.error(`Failed to fetch attachment: ${response.status}`);
      return null;
    }
    const arrayBuffer = await response.arrayBuffer();
    const base64 = encodeBase64(new Uint8Array(arrayBuffer));
    return base64;
  } catch (e) {
    console.error("Error fetching file attachment:", e);
    return null;
  }
};

const kimiApiKeyInjection: Injection<() => string> = context((): string => {
  throw new Error("no kimi API key injected");
});

export const injectKimiToken = (token: string): Injector =>
  kimiApiKeyInjection.inject(() => token);

const kimiModelVersion = "kimi-k2.5";

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

const isTokenLimitExceeded = (error: Error) =>
  "status" in error && (error as { status: number }).status === 400 &&
  error.message.includes("token count exceeds");

const isRateLimitError = (error: Error) =>
  "status" in error && (error as { status: number }).status === 429;

const dropOldestHalf = <T extends { type: string }>(events: T[]): T[] => {
  if (events.length <= 2) return events;
  const half = Math.floor(events.length / 2);
  return events.slice(half);
};

type KimiMetadata = {
  type: "kimi";
  responseId: string;
  reasoningContent?: string | null;
};

type KimiHistoryEvent = HistoryEventWithMetadata<KimiMetadata>;

type KimiOutputPart =
  | { type: "text"; text: string; reasoningContent?: string }
  | {
    type: "function_call";
    name: string;
    arguments: Record<string, unknown>;
    id?: string;
    reasoningContent?: string;
  };

type KimiRequestParams = {
  model: string;
  messages: ChatCompletionMessageParam[];
  tools?: ChatCompletionTool[];
  max_tokens?: number;
  temperature?: number;
  stream: boolean;
};

const attachmentsToContentParts = async (
  attachments?: MediaAttachment[],
): Promise<OpenAI.Chat.Completions.ChatCompletionContentPart[] | undefined> => {
  if (!attachments || empty(attachments)) return undefined;

  const parts: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [];

  for (const att of attachments) {
    if (att.kind === "inline") {
      if (att.mimeType.startsWith("image/")) {
        parts.push({
          type: "image_url",
          image_url: { url: `data:${att.mimeType};base64,${att.dataBase64}` },
        });
      } else if (att.mimeType.startsWith("audio/")) {
        // Send audio as base64 data URL for multimodal models
        parts.push({
          type: "text",
          text: `data:${att.mimeType};base64,${att.dataBase64}`,
        });
      } else {
        parts.push({
          type: "text",
          text: `[Attachment: ${att.caption || att.mimeType}]`,
        });
      }
    } else if (att.kind === "file" && att.fileUri) {
      const base64 = await fetchFileAttachment(att);
      if (base64) {
        if (att.mimeType.startsWith("image/")) {
          parts.push({
            type: "image_url",
            image_url: { url: `data:${att.mimeType};base64,${base64}` },
          });
        } else if (att.mimeType.startsWith("audio/")) {
          parts.push({
            type: "text",
            text: `data:${att.mimeType};base64,${base64}`,
          });
        } else {
          parts.push({
            type: "text",
            text: `[Attachment: ${att.caption || att.mimeType}]`,
          });
        }
      } else {
        parts.push({
          type: "text",
          text: `[File: ${att.caption || att.fileUri}]`,
        });
      }
    }
  }

  return parts;
};

const historyEventToMessage = (
  eventById: (id: string) => KimiHistoryEvent | undefined,
  timezoneIANA: string,
) =>
async (e: KimiHistoryEvent): Promise<ChatCompletionMessageParam[]> => {
  const getRefText = (onMessage: MessageId): string => {
    const msg = eventById(onMessage);
    return typeof msg === "object" && "text" in msg ? msg.text : "";
  };

  const stampText = (text: string) =>
    appendInternalSentTimestamp(text, e.timestamp, timezoneIANA);

  if (
    e.type === "participant_utterance" || e.type === "participant_edit_message"
  ) {
    const text = e.type === "participant_edit_message"
      ? `${e.name} edited message "${
        getRefText(e.onMessage).slice(0, 100)
      }" to: ${e.text}`
      : e.text
      ? `${e.name}: ${e.text}`
      : "";

    const contentParts = await attachmentsToContentParts(e.attachments);

    if (contentParts && contentParts.length > 0) {
      const textPart: OpenAI.Chat.Completions.ChatCompletionContentPart = {
        type: "text",
        text: stampText(text) || "<message with attachments>",
      };
      return [{ role: "user", content: [textPart, ...contentParts] }];
    }

    return [{ role: "user", content: stampText(text) }];
  }

  if (e.type === "own_utterance" || e.type === "own_edit_message") {
    const text = e.type === "own_edit_message"
      ? `You edited message "${
        getRefText(e.onMessage).slice(0, 100)
      }" to: ${e.text}`
      : e.text;

    // For assistant messages with attachments, include as text descriptions
    // (Kimi/OpenAI doesn't support images in assistant content parts)
    const contentParts = await attachmentsToContentParts(e.attachments);
    const attachmentDescriptions = contentParts
      ? contentParts
        .filter((
          p,
        ): p is OpenAI.Chat.Completions.ChatCompletionContentPartText =>
          p.type === "text"
        )
        .map((p) => p.text)
        .join("\n")
      : "";

    const fullContent = attachmentDescriptions
      ? `${text || ""}\n${attachmentDescriptions}`.trim()
      : (text || "");

    return [{ role: "assistant", content: fullContent }];
  }

  if (e.type === "tool_call") {
    const metadata = e.modelMetadata as KimiMetadata | undefined;
    return [{
      role: "assistant",
      content: null,
      tool_calls: [{
        id: e.id,
        type: "function",
        function: {
          name: e.name,
          arguments: JSON.stringify(e.parameters),
        },
      }],
      // Kimi-specific: include reasoning_content for thinking mode
      ...(metadata?.reasoningContent !== undefined
        ? { reasoning_content: metadata.reasoningContent }
        : {}),
    } as ChatCompletionMessageParam];
  }

  if (e.type === "tool_result") {
    return [{
      role: "tool",
      tool_call_id: e.toolCallId || e.id,
      content: stampText(e.result),
    }];
  }

  if (e.type === "own_thought") {
    return e.modelMetadata
      ? [{ role: "assistant", content: "" }]
      : [{ role: "user", content: `[System notification: ${e.text}]` }];
  }

  if (e.type === "own_reaction") {
    return [{
      role: "assistant",
      content: `You reacted ${e.reaction} to message: ${
        getRefText(e.onMessage).slice(0, 100)
      }`,
    }];
  }

  if (e.type === "participant_reaction") {
    return [{
      role: "user",
      content: `${e.name} reacted ${e.reaction} to message: ${
        getRefText(e.onMessage).slice(0, 100)
      }`,
    }];
  }

  if (e.type === "do_nothing") {
    return [{
      role: "assistant",
      content: "",
    }];
  }

  throw new Error(
    `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
  );
};

const actionToTool = (
  { name, description, parameters }: Tool<ZodType>,
): ChatCompletionTool => {
  const schema = zodToGeminiParameters(parameters);
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: schema as unknown as Record<string, unknown>,
    },
  };
};

type BuildReqFn = (events: KimiHistoryEvent[]) => Promise<KimiRequestParams>;

const buildReq = (
  prompt: string,
  tools: Tool<ZodType>[],
  skills: AgentSpec["skills"],
  timezoneIANA: string,
  maxOutputTokens: number | undefined,
): BuildReqFn =>
async (events: KimiHistoryEvent[]): Promise<KimiRequestParams> => {
  const eventById = (id: MessageId) => events.find((e) => e.id === id);

  // Filter out do_nothing events as they create empty assistant messages
  const filteredEvents = events.filter((e) => e.type !== "do_nothing");

  const messages: ChatCompletionMessageParam[] = [
    { role: "system", content: prompt },
    ...(await Promise.all(
      filteredEvents.map(historyEventToMessage(eventById, timezoneIANA)),
    )).flat(),
  ];

  const finalMessages: ChatCompletionMessageParam[] = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    finalMessages.push(msg);
    if (msg.role === "assistant" && msg.tool_calls) {
      const missingToolCalls = msg.tool_calls.map((tc) => tc.id).filter(
        (id) => {
          for (let j = i + 1; j < messages.length; j++) {
            const nextMsg = messages[j];
            if (nextMsg.role === "tool" && nextMsg.tool_call_id === id) {
              return false;
            }
          }
          return true;
        },
      );
      for (const id of missingToolCalls) {
        finalMessages.push({
          role: "tool",
          tool_call_id: id,
          content: "[Tool result unavailable]",
        });
      }
    }
  }
  messages.length = 0;
  messages.push(...finalMessages);

  const firstNonSystem = messages.find((m) => m.role !== "system");
  if (firstNonSystem && firstNonSystem.role !== "user") {
    messages.splice(1, 0, { role: "user", content: "<conversation started>" });
  }

  const allTools = skills && skills.length > 0
    ? [...tools, ...createSkillTools(skills)]
    : tools;

  const req: KimiRequestParams = {
    model: kimiModelVersion,
    messages,
    stream: false,
    ...(allTools.length > 0 ? { tools: allTools.map(actionToTool) } : {}),
    ...(maxOutputTokens ? { max_tokens: maxOutputTokens } : {}),
  };

  return req;
};

const rawCallKimi = async ({
  req,
  disableStreaming,
}: {
  req: KimiRequestParams;
  disableStreaming?: boolean;
}): Promise<KimiOutputPart[]> => {
  const handleStreamChunk = getStreamChunk();

  const client = new OpenAI({
    apiKey: kimiApiKeyInjection.access(),
    baseURL: "https://api.moonshot.ai/v1",
  });

  if (disableStreaming) {
    const response = await client.chat.completions.create({
      ...req,
      stream: false,
    });

    const choice = response.choices[0];
    if (!choice) return [];

    const reasoning = (choice.message as unknown as Record<string, unknown>)
      .reasoning_content as
        | string
        | undefined;

    if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
      return choice.message.tool_calls.map((tc, i): KimiOutputPart => ({
        type: "function_call",
        id: tc.id,
        name: tc.function.name,
        arguments: JSON.parse(tc.function.arguments || "{}"),
        ...(i === 0 && reasoning ? { reasoningContent: reasoning } : {}),
      }));
    }

    const content = choice.message.content || "";
    if (content) {
      await handleStreamChunk(content);
    }

    return [{
      type: "text",
      text: content,
      ...(reasoning ? { reasoningContent: reasoning } : {}),
    }];
  }

  // Streaming mode
  const stream = await client.chat.completions.create({
    ...req,
    stream: true,
  });

  const accumulatedContent: string[] = [];
  const accumulatedReasoning: string[] = [];
  const toolCalls = new Map<
    string,
    { id: string; name: string; args: string }
  >();

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta;
    if (!delta) continue;

    const deltaRecord = delta as unknown as Record<string, unknown>;
    if (typeof deltaRecord.reasoning_content === "string") {
      accumulatedReasoning.push(deltaRecord.reasoning_content);
    }

    if (delta.content) {
      await handleStreamChunk(delta.content);
      accumulatedContent.push(delta.content);
    }

    if (delta.tool_calls && delta.tool_calls.length > 0) {
      for (const tc of delta.tool_calls) {
        const idx = tc.index?.toString() || "0";
        const existing = toolCalls.get(idx);
        if (existing) {
          existing.args += tc.function?.arguments || "";
        } else {
          toolCalls.set(idx, {
            id: tc.id || generateId(),
            name: tc.function?.name || "",
            args: tc.function?.arguments || "",
          });
        }
      }
    }
  }

  const reasoning = accumulatedReasoning.join("") || undefined;

  if (toolCalls.size > 0) {
    const parts = Array.from(toolCalls.values());
    return parts.map((tc, i): KimiOutputPart => ({
      type: "function_call",
      id: tc.id,
      name: tc.name,
      arguments: JSON.parse(tc.args || "{}"),
      ...(i === 0 && reasoning ? { reasoningContent: reasoning } : {}),
    }));
  }

  return [{
    type: "text",
    text: accumulatedContent.join(""),
    ...(reasoning ? { reasoningContent: reasoning } : {}),
  }];
};

const callKimiWithRetry = conditionalRetry(isServerError)(
  1000,
  4,
  rawCallKimi,
);

const callKimi = async (
  req: KimiRequestParams,
  disableStreaming?: boolean,
): Promise<KimiOutputPart[]> => {
  try {
    return await callKimiWithRetry({ req, disableStreaming });
  } catch (error) {
    const err = normalizeError(error);
    if (isServerError(err) || isRateLimitError(err)) {
      return rawCallKimi({ req, disableStreaming });
    }
    throw err;
  }
};

const callKimiWithFixHistory = (
  _rewriteHistory: AgentSpec["rewriteHistory"],
  eventsToRequest: BuildReqFn,
  disableStreaming?: boolean,
) =>
async (events: KimiHistoryEvent[]): Promise<KimiOutputPart[]> => {
  try {
    return await callKimi(await eventsToRequest(events), disableStreaming);
  } catch (error) {
    const err = normalizeError(error);

    if (isTokenLimitExceeded(err)) {
      const totalTokens = sum(map(estimateTokens)(events));
      console.warn(
        `Token limit exceeded (estimated ${totalTokens} tokens, ${events.length} events). Dropping oldest half.`,
      );
      const truncated = dropOldestHalf(events);
      if (truncated.length === events.length) throw err;
      return callKimi(await eventsToRequest(truncated), disableStreaming);
    }

    throw err;
  }
};

const didNothing = (output: KimiOutputPart[]) =>
  output.length === 0 ||
  (output.length === 1 &&
    output[0].type === "text" &&
    !output[0].text.replace(/[\s\u200B\u200C\u200D\uFEFF]/g, ""));

const kimiOutputPartToHistoryEvents =
  (responseId: string) => (p: KimiOutputPart): KimiHistoryEvent[] => {
    const metadata: KimiMetadata = {
      type: "kimi",
      responseId,
      reasoningContent: p.reasoningContent,
    };

    const thoughtEvent = p.reasoningContent
      ? [ownThoughtTurnWithMetadata<KimiMetadata>(p.reasoningContent, metadata)]
      : [];

    if (p.type === "text") {
      const text = p.text || "";
      const stripped = stripInternalSentTimestampSuffix(text);

      const thoughtRegex =
        /^\[Internal thought, visible only to you: ([\s\S]*?)\]$/;
      const match = stripped.match(thoughtRegex);

      if (match) {
        return [
          ...thoughtEvent,
          ownThoughtTurnWithMetadata<KimiMetadata>(match[1], metadata),
        ];
      }

      const notificationRegex = /^\[System notification: ([\s\S]*?)\]$/;
      const notificationMatch = stripped.match(notificationRegex);

      if (notificationMatch) {
        return [
          ...thoughtEvent,
          ownThoughtTurnWithMetadata<KimiMetadata>(
            notificationMatch[1],
            metadata,
          ),
        ];
      }

      return [
        ...thoughtEvent,
        ownUtteranceTurnWithMetadata<KimiMetadata>(stripped, metadata),
      ];
    }

    if (p.type === "function_call") {
      return [
        ...thoughtEvent,
        toolUseTurnWithMetadata(
          { name: p.name, args: p.arguments, id: p.id },
          metadata,
        ),
      ];
    }

    throw new Error(`Unknown Kimi output part type: ${JSON.stringify(p)}`);
  };

export const kimiAgentCaller = ({
  lightModel,
  prompt,
  tools,
  skills,
  rewriteHistory,
  timezoneIANA,
  maxOutputTokens,
  disableStreaming,
}: AgentSpec) =>
async (events: KimiHistoryEvent[]): Promise<KimiHistoryEvent[]> => {
  void lightModel;

  const enhancedPrompt = skills && skills.length > 0
    ? `${prompt}\n\nAvailable skills:\n${
      skills.map((skill) => `- ${skill.name}: ${skill.description}`).join("\n")
    }`
    : prompt;

  const kimiOutput = await callKimiWithFixHistory(
    rewriteHistory,
    buildReq(enhancedPrompt, tools, skills, timezoneIANA, maxOutputTokens),
    disableStreaming,
  )(events);

  const responseId = generateId();

  if (didNothing(kimiOutput)) {
    return [doNothingEvent({ type: "kimi", responseId })];
  }

  return kimiOutput.flatMap(kimiOutputPartToHistoryEvents(responseId));
};
