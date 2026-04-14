import Anthropic from "@anthropic-ai/sdk";
import { context, type Injection, type Injector } from "@uri/inject";
import { conditionalRetry, empty, map, sum } from "gamla";
import type { ZodType } from "zod/v4";
import { zodToGeminiParameters } from "./gemini.ts";
import {
  type AgentSpec,
  createSkillTools,
  doNothingEvent,
  estimateTokens,
  generateId,
  getStreamChunk,
  getStreamThinkingChunk,
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
import { isRetryableError, normalizeError } from "./utils.ts";

import { encodeBase64 } from "@std/encoding/base64";

const fetchFileAttachment = async (
  attachment: MediaAttachment,
): Promise<string | null> => {
  if (attachment.kind !== "file" || !attachment.fileUri) return null;

  try {
    const response = await fetch(attachment.fileUri);
    if (!response.ok) {
      console.error(`Failed to fetch attachment: ${response.status}`);
      await response.body?.cancel();
      return null;
    }
    const arrayBuffer = await response.arrayBuffer();
    return encodeBase64(new Uint8Array(arrayBuffer));
  } catch (e) {
    console.error("Error fetching file attachment:", e);
    return null;
  }
};

const anthropicApiKeyInjection: Injection<() => string> = context(
  (): string => {
    throw new Error("no anthropic API key injected");
  },
);

export const injectAnthropicToken = (token: string): Injector =>
  anthropicApiKeyInjection.inject(() => token);

const anthropicModel = (lightModel?: boolean) =>
  lightModel ? "claude-sonnet-4-6" : "claude-opus-4-6";

const isTokenLimitExceeded = (error: Error) =>
  "status" in error && (error as { status: number }).status === 400 &&
  (error.message.includes("too long") ||
    error.message.includes("token") ||
    error.message.includes("max_tokens"));

const dropOldestHalf = <T extends { type: string }>(events: T[]): T[] => {
  if (events.length <= 2) return events;
  const half = Math.floor(events.length / 2);
  return events.slice(half);
};

type AnthropicMetadata = {
  type: "anthropic";
  responseId: string;
  thinkingContent?: string | null;
};

type AnthropicHistoryEvent = HistoryEventWithMetadata<AnthropicMetadata>;

type AnthropicOutputPart =
  | { type: "text"; text: string; thinkingContent?: string }
  | {
    type: "function_call";
    name: string;
    arguments: Record<string, unknown>;
    id?: string;
    thinkingContent?: string;
  };

type StreamingOutputPart =
  | { type: "text"; text: string; index: number }
  | {
    type: "function_call";
    id: string;
    name: string;
    input: string;
    index: number;
  };

type AnthropicRequestParams = {
  model: string;
  system: string;
  messages: Anthropic.Messages.MessageParam[];
  tools?: Anthropic.Messages.Tool[];
  max_tokens: number;
  stream: boolean;
  thinking?: { type: "enabled"; budget_tokens: number };
};

const anthropicThinkingBudget = (maxTokens: number) =>
  Math.max(1024, Math.min(10000, maxTokens - 1));

type AnthropicMediaType =
  | "image/jpeg"
  | "image/png"
  | "image/gif"
  | "image/webp";

const isImageMimeType = (
  mimeType: string,
): mimeType is AnthropicMediaType =>
  ["image/jpeg", "image/png", "image/gif", "image/webp"].includes(mimeType);

const sanitizeToolId = (id: string): string =>
  id.replace(/[^a-zA-Z0-9_-]/g, "_");

const attachmentsToContentBlocks = async (
  attachments?: MediaAttachment[],
): Promise<Anthropic.Messages.ContentBlockParam[] | undefined> => {
  if (!attachments || empty(attachments)) return undefined;

  const blocks: Anthropic.Messages.ContentBlockParam[] = [];

  for (const att of attachments) {
    if (att.kind === "inline") {
      if (isImageMimeType(att.mimeType)) {
        blocks.push({
          type: "image",
          source: {
            type: "base64",
            media_type: att.mimeType,
            data: att.dataBase64,
          },
        });
      } else {
        blocks.push({
          type: "text",
          text: `[Attachment: ${att.caption || att.mimeType}]`,
        });
      }
    } else if (att.kind === "file" && att.fileUri) {
      const base64 = await fetchFileAttachment(att);
      if (base64 && isImageMimeType(att.mimeType)) {
        blocks.push({
          type: "image",
          source: {
            type: "base64",
            media_type: att.mimeType,
            data: base64,
          },
        });
      } else {
        blocks.push({
          type: "text",
          text: `[File: ${att.caption || att.fileUri}]`,
        });
      }
    }
  }

  return blocks;
};

const historyEventToMessage = (
  eventById: (id: string) => AnthropicHistoryEvent | undefined,
  timezoneIANA: string,
) =>
async (
  e: AnthropicHistoryEvent,
): Promise<Anthropic.Messages.MessageParam[]> => {
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

    const contentBlocks = await attachmentsToContentBlocks(e.attachments);

    if (contentBlocks && contentBlocks.length > 0) {
      const textBlock: Anthropic.Messages.TextBlockParam = {
        type: "text",
        text: stampText(text) || "<message with attachments>",
      };
      return [{
        role: "user",
        content: [textBlock, ...contentBlocks],
      }];
    }

    return [{ role: "user", content: stampText(text) }];
  }

  if (e.type === "own_utterance" || e.type === "own_edit_message") {
    const text = e.type === "own_edit_message"
      ? `You edited message "${
        getRefText(e.onMessage).slice(0, 100)
      }" to: ${e.text}`
      : e.text;

    const contentBlocks = await attachmentsToContentBlocks(e.attachments);
    const attachmentDescriptions = contentBlocks
      ? contentBlocks
        .filter((p): p is Anthropic.Messages.TextBlockParam =>
          p.type === "text"
        )
        .map((p) => p.text)
        .join("\n")
      : "";

    const fullContent = attachmentDescriptions
      ? `${text || ""}\n${attachmentDescriptions}`.trim()
      : (text || "");

    if (!fullContent.trim()) return [];
    return [{ role: "assistant", content: fullContent }];
  }

  if (e.type === "tool_call") {
    return [{
      role: "assistant",
      content: [{
        type: "tool_use",
        id: sanitizeToolId(e.id),
        name: e.name,
        input: e.parameters as Record<string, unknown>,
      } as Anthropic.Messages.ToolUseBlockParam],
    }];
  }

  if (e.type === "tool_result") {
    return [{
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: sanitizeToolId(e.toolCallId || e.id),
        content: stampText(e.result),
      } as Anthropic.Messages.ToolResultBlockParam],
    }];
  }

  if (e.type === "own_thought") {
    return e.modelMetadata
      ? []
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
      content: " ",
    }];
  }

  throw new Error(
    `Unknown history event type: ${JSON.stringify(e, null, 2)}`,
  );
};

const actionToTool = (
  { name, description, parameters }: Tool<ZodType>,
): Anthropic.Messages.Tool => {
  const schema = zodToGeminiParameters(parameters);
  return {
    name,
    description,
    input_schema: schema as unknown as Anthropic.Messages.Tool.InputSchema,
  };
};

type BuildReqFn = (
  events: AnthropicHistoryEvent[],
) => Promise<AnthropicRequestParams>;

const mergeConsecutiveSameRole = (
  messages: Anthropic.Messages.MessageParam[],
): Anthropic.Messages.MessageParam[] => {
  if (messages.length === 0) return messages;

  const merged: Anthropic.Messages.MessageParam[] = [messages[0]];

  for (let i = 1; i < messages.length; i++) {
    const prev = merged[merged.length - 1];
    const curr = messages[i];

    if (prev.role === curr.role) {
      const prevContent = typeof prev.content === "string"
        ? [{ type: "text" as const, text: prev.content }]
        : prev.content;
      const currContent = typeof curr.content === "string"
        ? [{ type: "text" as const, text: curr.content }]
        : curr.content;
      merged[merged.length - 1] = {
        role: prev.role,
        content: [...prevContent, ...currContent],
      };
    } else {
      merged.push(curr);
    }
  }

  return merged;
};

const toolUseIdsInMessage = (
  msg: Anthropic.Messages.MessageParam,
): string[] =>
  msg.role === "assistant" && Array.isArray(msg.content)
    ? (msg.content as Anthropic.Messages.ContentBlockParam[])
      .filter((b): b is Anthropic.Messages.ToolUseBlockParam =>
        "type" in b && b.type === "tool_use"
      )
      .map((b) => b.id)
    : [];

const toolResultIdsInMessage = (
  msg: Anthropic.Messages.MessageParam,
): Set<string> =>
  new Set(
    msg.role === "user" && Array.isArray(msg.content)
      ? (msg.content as Anthropic.Messages.ContentBlockParam[])
        .filter((b): b is Anthropic.Messages.ToolResultBlockParam =>
          "type" in b && b.type === "tool_result"
        )
        .map((b) => b.tool_use_id)
      : [],
  );

const toContentBlocks = (
  content: Anthropic.Messages.MessageParam["content"],
): Anthropic.Messages.ContentBlockParam[] =>
  typeof content === "string"
    ? [{ type: "text" as const, text: content }]
    : content;

const isToolResultBlock = (
  block: Anthropic.Messages.ContentBlockParam,
): block is Anthropic.Messages.ToolResultBlockParam =>
  "type" in block && block.type === "tool_result";

const orderToolResultsFirst = (
  useIds: string[],
  content: Anthropic.Messages.MessageParam["content"],
): Anthropic.Messages.ContentBlockParam[] => {
  const blocks = toContentBlocks(content);
  const toolResults = blocks.filter(isToolResultBlock);
  const prioritized = useIds.flatMap((id) =>
    toolResults.filter((block) => block.tool_use_id === id)
  );
  const prioritizedIds = new Set(prioritized.map((block) => block.tool_use_id));
  return [
    ...prioritized,
    ...blocks.filter((block) =>
      !isToolResultBlock(block) || !prioritizedIds.has(block.tool_use_id)
    ),
  ];
};

const syntheticToolResult = (
  toolUseId: string,
): Anthropic.Messages.ToolResultBlockParam => ({
  type: "tool_result" as const,
  tool_use_id: toolUseId,
  content: "[Tool result unavailable]",
});

const allToolUseIds = (
  messages: Anthropic.Messages.MessageParam[],
): Set<string> => new Set(messages.flatMap(toolUseIdsInMessage));

const stripOrphanedToolResults = (
  messages: Anthropic.Messages.MessageParam[],
): Anthropic.Messages.MessageParam[] => {
  const validIds = allToolUseIds(messages);
  return messages.flatMap((msg) => {
    if (msg.role !== "user" || !Array.isArray(msg.content)) return [msg];
    const filtered = (msg.content as Anthropic.Messages.ContentBlockParam[])
      .filter((b) =>
        !("type" in b && b.type === "tool_result") ||
        validIds.has(
          (b as Anthropic.Messages.ToolResultBlockParam).tool_use_id,
        )
      );
    if (empty(filtered)) return [];
    return [{ role: "user" as const, content: filtered }];
  });
};

const ensureToolResultsForToolUses = (
  messages: Anthropic.Messages.MessageParam[],
): Anthropic.Messages.MessageParam[] => {
  const cleaned = stripOrphanedToolResults(messages);
  const result: Anthropic.Messages.MessageParam[] = [];
  for (let i = 0; i < cleaned.length; i++) {
    result.push(cleaned[i]);
    const useIds = toolUseIdsInMessage(cleaned[i]);
    if (empty(useIds)) continue;
    const next = cleaned[i + 1];
    const existingIds = next ? toolResultIdsInMessage(next) : new Set();
    const missingIds = useIds.filter((id) => !existingIds.has(id));
    if (next && next.role === "user") {
      const existing = orderToolResultsFirst(useIds, next.content);
      if (empty(missingIds)) {
        cleaned[i + 1] = {
          role: "user",
          content: existing,
        };
        continue;
      }
      cleaned[i + 1] = {
        role: "user",
        content: [
          ...missingIds.map(syntheticToolResult),
          ...existing,
        ] as Anthropic.Messages.ContentBlockParam[],
      };
    } else {
      if (empty(missingIds)) continue;
      result.push({
        role: "user",
        content: missingIds.map(syntheticToolResult),
      });
    }
  }
  return result;
};

const buildReq = (
  systemPrompt: string,
  tools: Tool<ZodType>[],
  skills: AgentSpec["skills"],
  timezoneIANA: string,
  maxOutputTokens: number | undefined,
  lightModel: boolean | undefined,
): BuildReqFn =>
async (events: AnthropicHistoryEvent[]): Promise<AnthropicRequestParams> => {
  const eventById = (id: MessageId) => events.find((e) => e.id === id);

  const filteredEvents = events.filter((e) => e.type !== "do_nothing");

  const rawMessages: Anthropic.Messages.MessageParam[] = (await Promise.all(
    filteredEvents.map(historyEventToMessage(eventById, timezoneIANA)),
  )).flat();

  const merged = mergeConsecutiveSameRole(rawMessages);

  const withToolResults = ensureToolResultsForToolUses(merged);

  // Anthropic requires the first message to be from the user
  const withUserFirst =
    withToolResults.length > 0 && withToolResults[0].role !== "user"
      ? [{
        role: "user" as const,
        content: "<conversation started>",
      }, ...withToolResults]
      : withToolResults;

  // Anthropic requires at least one message
  const nonEmpty = empty(withUserFirst)
    ? [{ role: "user" as const, content: "<conversation started>" }]
    : withUserFirst;

  const effectiveMaxTokens = maxOutputTokens ?? 16000;
  const thinkingEnabled = effectiveMaxTokens > 1024;
  const thinkingBudget = thinkingEnabled
    ? anthropicThinkingBudget(effectiveMaxTokens)
    : undefined;

  // When thinking is enabled, Anthropic does not allow assistant prefill
  // (last message must be from user)
  const messages = thinkingEnabled &&
      nonEmpty[nonEmpty.length - 1].role === "assistant"
    ? [...nonEmpty, { role: "user" as const, content: "<continue>" }]
    : nonEmpty;

  const allTools = skills && skills.length > 0
    ? [...tools, ...createSkillTools(skills)]
    : tools;

  return {
    model: anthropicModel(lightModel),
    system: systemPrompt,
    messages,
    stream: false,
    max_tokens: effectiveMaxTokens,
    ...(thinkingEnabled && thinkingBudget
      ? {
        thinking: { type: "enabled" as const, budget_tokens: thinkingBudget },
      }
      : {}),
    ...(allTools.length > 0 ? { tools: allTools.map(actionToTool) } : {}),
  };
};

const rawCallAnthropic = async ({
  req,
  disableStreaming,
}: {
  req: AnthropicRequestParams;
  disableStreaming?: boolean;
}): Promise<AnthropicOutputPart[]> => {
  const handleStreamChunk = getStreamChunk();
  const handleStreamThinkingChunk = getStreamThinkingChunk();

  const client = new Anthropic({
    apiKey: anthropicApiKeyInjection.access(),
  });

  if (disableStreaming) {
    const { system, stream: _stream, ...rest } = req;
    const response = await client.messages.create({
      ...rest,
      system,
      stream: false,
    } as Anthropic.Messages.MessageCreateParamsNonStreaming);

    const parts: AnthropicOutputPart[] = [];
    let lastThinkingContent: string | undefined;

    for (const block of response.content) {
      if (block.type === "thinking") {
        lastThinkingContent = block.thinking;
        await handleStreamThinkingChunk(block.thinking);
      } else if (block.type === "text") {
        if (block.text) {
          await handleStreamChunk(block.text);
        }
        parts.push({
          type: "text",
          text: block.text,
          thinkingContent: lastThinkingContent,
        });
        lastThinkingContent = undefined;
      } else if (block.type === "tool_use") {
        parts.push({
          type: "function_call",
          id: block.id,
          name: block.name,
          arguments: block.input as Record<string, unknown>,
          thinkingContent: lastThinkingContent,
        });
        lastThinkingContent = undefined;
      }
    }

    return parts.length > 0 ? parts : [{ type: "text", text: "" }];
  }

  // Streaming mode
  const { system, stream: _stream, ...rest } = req;
  const stream = client.messages.stream({
    ...rest,
    system,
  } as Anthropic.Messages.MessageCreateParamsStreaming);

  const accumulatedThinking: string[] = [];
  const parts = new Map<number, StreamingOutputPart>();
  let currentToolId: string | null = null;
  let currentToolIndex: number | null = null;
  let currentTextIndex: number | null = null;

  for await (const event of stream) {
    if (event.type === "content_block_start") {
      const block = event.content_block;
      if (block.type === "text") {
        currentTextIndex = event.index;
        parts.set(event.index, {
          type: "text",
          text: block.text,
          index: event.index,
        });
      }
      if (block.type === "tool_use") {
        currentToolId = block.id;
        currentToolIndex = event.index;
        parts.set(event.index, {
          type: "function_call",
          id: block.id,
          name: block.name,
          input: "",
          index: event.index,
        });
      }
    } else if (event.type === "content_block_delta") {
      const delta = event.delta;
      if (delta.type === "text_delta") {
        await handleStreamChunk(delta.text);
        if (currentTextIndex !== null) {
          const existing = parts.get(currentTextIndex);
          if (existing?.type === "text") {
            existing.text += delta.text;
          }
        }
      } else if (delta.type === "thinking_delta") {
        accumulatedThinking.push(delta.thinking);
        await handleStreamThinkingChunk(delta.thinking);
      } else if (delta.type === "input_json_delta" && currentToolId) {
        const existing = currentToolIndex !== null
          ? parts.get(currentToolIndex)
          : undefined;
        if (
          existing?.type === "function_call" && existing.id === currentToolId
        ) {
          existing.input += delta.partial_json;
        }
      }
    } else if (event.type === "content_block_stop") {
      currentToolId = null;
      currentToolIndex = null;
      currentTextIndex = null;
    }
  }

  const thinkingContent = accumulatedThinking.join("") || undefined;

  const output = Array.from(parts.values())
    .sort((a, b) => a.index - b.index)
    .map((part): AnthropicOutputPart =>
      part.type === "text"
        ? {
          type: "text",
          text: part.text,
          thinkingContent,
        }
        : {
          type: "function_call",
          id: part.id,
          name: part.name,
          arguments: JSON.parse(part.input || "{}"),
          thinkingContent,
        }
    )
    .filter((part) => part.type !== "text" || part.text);

  return output.length > 0 ? output : [{ type: "text", text: "" }];
};

const callAnthropicWithRetry = conditionalRetry(isRetryableError)(
  1000,
  4,
  rawCallAnthropic,
);

const callAnthropic = async (
  req: AnthropicRequestParams,
  disableStreaming?: boolean,
): Promise<AnthropicOutputPart[]> => {
  try {
    return await callAnthropicWithRetry({ req, disableStreaming });
  } catch (error) {
    const err = normalizeError(error);
    if (isRetryableError(err)) {
      return rawCallAnthropic({ req, disableStreaming });
    }
    throw err;
  }
};

const callAnthropicWithFixHistory = (
  _rewriteHistory: AgentSpec["rewriteHistory"],
  eventsToRequest: BuildReqFn,
  disableStreaming?: boolean,
) =>
async (events: AnthropicHistoryEvent[]): Promise<AnthropicOutputPart[]> => {
  try {
    return await callAnthropic(
      await eventsToRequest(events),
      disableStreaming,
    );
  } catch (error) {
    const err = normalizeError(error);

    if (isTokenLimitExceeded(err)) {
      const totalTokens = sum(map(estimateTokens)(events));
      console.warn(
        `Token limit exceeded (estimated ${totalTokens} tokens, ${events.length} events). Dropping oldest half.`,
      );
      const truncated = dropOldestHalf(events);
      if (truncated.length === events.length) throw err;
      return callAnthropic(
        await eventsToRequest(truncated),
        disableStreaming,
      );
    }

    throw err;
  }
};

export const noResponseTag = "<no response>";

const didNothing = (output: AnthropicOutputPart[]) =>
  output.length === 0 ||
  (output.length === 1 &&
    output[0].type === "text" &&
    (!output[0].text.replace(/[\s\u200B\u200C\u200D\uFEFF]/g, "") ||
      output[0].text.trim().toLowerCase() === noResponseTag));

const anthropicOutputPartToHistoryEvents =
  (responseId: string) => (p: AnthropicOutputPart): AnthropicHistoryEvent[] => {
    const metadata: AnthropicMetadata = {
      type: "anthropic",
      responseId,
      thinkingContent: p.thinkingContent,
    };

    const thoughtEvent = p.thinkingContent
      ? [
        ownThoughtTurnWithMetadata<AnthropicMetadata>(
          p.thinkingContent,
          metadata,
        ),
      ]
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
          ownThoughtTurnWithMetadata<AnthropicMetadata>(match[1], metadata),
        ];
      }

      const notificationRegex = /^\[System notification: ([\s\S]*?)\]$/;
      const notificationMatch = stripped.match(notificationRegex);

      if (notificationMatch) {
        return [
          ...thoughtEvent,
          ownThoughtTurnWithMetadata<AnthropicMetadata>(
            notificationMatch[1],
            metadata,
          ),
        ];
      }

      return [
        ...thoughtEvent,
        ownUtteranceTurnWithMetadata<AnthropicMetadata>(stripped, metadata),
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

    throw new Error(
      `Unknown Anthropic output part type: ${JSON.stringify(p)}`,
    );
  };

export const anthropicAgentCaller = ({
  lightModel,
  prompt,
  tools,
  skills,
  rewriteHistory,
  timezoneIANA,
  maxOutputTokens,
  disableStreaming,
}: AgentSpec) =>
async (events: AnthropicHistoryEvent[]): Promise<AnthropicHistoryEvent[]> => {
  const enhancedPrompt = [
    prompt,
    ...(skills && skills.length > 0
      ? [
        `Available skills:\n${
          skills.map((skill) => `- ${skill.name}: ${skill.description}`).join(
            "\n",
          )
        }`,
      ]
      : []),
    `If you have nothing to say, reply with exactly ${noResponseTag} and nothing else.`,
  ].join("\n\n");

  const anthropicOutput = await callAnthropicWithFixHistory(
    rewriteHistory,
    buildReq(
      enhancedPrompt,
      tools,
      skills,
      timezoneIANA,
      maxOutputTokens,
      lightModel,
    ),
    disableStreaming,
  )(events);

  const responseId = generateId();

  if (didNothing(anthropicOutput)) {
    return [doNothingEvent({ type: "anthropic", responseId })];
  }

  return anthropicOutput.flatMap(
    anthropicOutputPartToHistoryEvents(responseId),
  );
};
