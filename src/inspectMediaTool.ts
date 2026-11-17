import { encodeBase64 } from "@std/encoding/base64";
import { context, type Injection } from "@uri/inject";
import { z } from "zod/v4";
import {
  getHistory,
  type HistoryEvent,
  type MediaAttachment,
  type Tool,
  type ToolReturn,
} from "./agent.ts";
import { applyCleanActiveMemoryDirectives } from "./compaction.ts";

export const inspectMediaUrlToolName = "inspect_media_url";

export const defaultInspectMediaTimeoutMs = 30_000;

const inspectMediaTimeoutMsInjection: Injection<() => number> = context(
  () => defaultInspectMediaTimeoutMs,
);

export const injectInspectMediaTimeoutMs =
  inspectMediaTimeoutMsInjection.inject;
export const accessInspectMediaTimeoutMs =
  inspectMediaTimeoutMsInjection.access;

export const alreadyVisibleMediaMessage = (url: string): string =>
  `Media from ${url} is already visible in your active conversation context — inspect it directly without calling ${inspectMediaUrlToolName} again.`;

const attachmentMatchesImageUrl = (url: string) => (att: MediaAttachment) =>
  att.mimeType.startsWith("image/") &&
  ((att.kind === "inline" && att.caption === url) ||
    (att.kind === "file" && (att.fileUri === url || att.caption === url)));

const isRenderedToolAttachmentForUrl = (
  url: string,
  isInspectCall: boolean,
) =>
(att: MediaAttachment): boolean =>
  (att.kind === "inline" && att.caption === url) ||
  (att.kind === "file" &&
    (Boolean(att.renderInline) || isInspectCall) &&
    (att.fileUri === url || att.caption === url));

export const isMediaUrlAlreadyVisibleInHistory = (
  history: HistoryEvent[],
  url: string,
): boolean => {
  const activeEvents = applyCleanActiveMemoryDirectives(history);
  const inspectCallIds = new Set(
    activeEvents
      .filter((e) =>
        e.type === "tool_call" && e.name === inspectMediaUrlToolName
      )
      .map((e) => e.id),
  );
  return activeEvents.some((e) => {
    if (e.type === "participant_utterance") {
      return (e.attachments ?? []).some(attachmentMatchesImageUrl(url));
    }
    if (e.type === "tool_result") {
      const isInspectCall = Boolean(
        e.toolCallId && inspectCallIds.has(e.toolCallId),
      );
      return (e.attachments ?? []).some(
        isRenderedToolAttachmentForUrl(url, isInspectCall),
      );
    }
    return false;
  });
};

const contentType = (response: Response, fallback: string) =>
  response.headers.get("content-type")?.split(";")[0]?.trim() || fallback;

const isPng = (bytes: Uint8Array) =>
  bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e &&
  bytes[3] === 0x47;

const isJpeg = (bytes: Uint8Array) => bytes[0] === 0xff && bytes[1] === 0xd8;

const isGif = (bytes: Uint8Array) =>
  bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46;

const isWebp = (bytes: Uint8Array) =>
  bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 &&
  bytes[3] === 0x46 && bytes[8] === 0x57 && bytes[9] === 0x45 &&
  bytes[10] === 0x42 && bytes[11] === 0x50;

const isValidMedia = (mimeType: string, bytes: Uint8Array) => {
  if (mimeType === "image/png") return isPng(bytes);
  if (mimeType === "image/jpeg") return isJpeg(bytes);
  if (mimeType === "image/gif") return isGif(bytes);
  if (mimeType === "image/webp") return isWebp(bytes);
  return true;
};

const isTimeoutOrAbortError = (error: unknown): boolean =>
  error instanceof Error &&
  (error.name === "AbortError" || error.name === "TimeoutError");

const fetchMedia = async (url: string, mimeType: string | undefined) => {
  const timeoutMs = accessInspectMediaTimeoutMs();
  const response = await fetch(url, {
    signal: AbortSignal.timeout(timeoutMs),
  }).catch((error: unknown) => {
    if (isTimeoutOrAbortError(error)) {
      throw new Error(
        `Failed to fetch media from ${url}: request timed out after ${timeoutMs}ms`,
      );
    }
    throw error;
  });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`Failed to fetch media: ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const resolvedMimeType = contentType(
    response,
    mimeType ?? "application/octet-stream",
  );
  if (!isValidMedia(resolvedMimeType, bytes)) {
    throw new Error(
      `Fetched media does not match declared MIME type ${resolvedMimeType}`,
    );
  }
  return { dataBase64: encodeBase64(bytes), mimeType: resolvedMimeType };
};

export const inspectMediaUrlTool: Tool<
  z.ZodObject<{
    url: z.ZodString;
    mimeType: z.ZodOptional<z.ZodString>;
  }>
> = {
  name: inspectMediaUrlToolName,
  description:
    "Fetch an image or video URL and make its visual content available to the model for inspection. Use this when a previous tool returned a media URL and you need to look at it before responding.",
  parameters: z.object({
    url: z.string().url().describe("The image or video URL to inspect"),
    mimeType: z.string().optional().describe(
      "Optional MIME type, e.g. image/png or video/mp4",
    ),
  }),
  handler: async ({ url, mimeType }) => {
    const history = await getHistory().catch((): HistoryEvent[] => []);
    if (isMediaUrlAlreadyVisibleInHistory(history, url)) {
      return alreadyVisibleMediaMessage(url);
    }
    const media = await fetchMedia(url, mimeType).catch((error: unknown) =>
      error instanceof Error ? error.message : String(error)
    );
    if (typeof media === "string") return media;
    const result: ToolReturn = {
      result: `Fetched media from ${url} (${media.mimeType}).`,
      attachments: [{
        kind: "inline",
        mimeType: media.mimeType,
        dataBase64: media.dataBase64,
        caption: url,
      }],
    };
    return result;
  },
};
