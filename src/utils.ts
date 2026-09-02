import { type EitherOutput, type Func, throwerCatcher } from "gamla";
import type { z, ZodType } from "zod/v4";

export type ModelTier = "lite" | "flash" | "pro";

export type ModelOpts = {
  tier?: ModelTier;
  maxOutputTokens?: number;
  provider?: "google" | "openai" | "gemini";
  disableThinking?: boolean;
};

const typeAdherenceError = throwerCatcher("AI refused to adhere to typing");

export const aiRefusesToAdhereTyping = typeAdherenceError.thrower;
export const catchAiRefusesToAdhereToTyping: <G extends Func>(
  fallback: G,
) => <F extends Func>(f: F) => (...xs: Parameters<F>) => EitherOutput<F, G> =
  typeAdherenceError.catcher;

export const normalizeError = (error: unknown): Error => {
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

const errorStatus = (error: unknown) =>
  error instanceof Error && "status" in error
    ? (error as { status: number }).status
    : undefined;

export const isServerError = (error: unknown) =>
  (errorStatus(error) ?? 0) >= 500;

export const isRateLimitError = (error: unknown) => errorStatus(error) === 429;

export const syntheticTimeoutMarker = "syntheticTimeout";

export const isSyntheticTimeoutError = (error: unknown) =>
  error instanceof Error && syntheticTimeoutMarker in error &&
  (error as { syntheticTimeout: unknown }).syntheticTimeout === true;

export const isTransientFetchError = (error: unknown) => {
  const norm = normalizeError(error);
  return (
    norm instanceof TypeError &&
    (/reading a body|network|connection|fetch/i.test(norm.message) ||
      norm.message.length === 0)
  );
};

export const isInvalidArgumentError = (error: unknown) =>
  normalizeError(error).message.includes(
    "Request contains an invalid argument",
  );

export const emptyGeminiCandidateMessage =
  "Gemini returned an empty candidate (no text)";

export const isEmptyGeminiCandidateError = (error: unknown) =>
  normalizeError(error).message.includes(emptyGeminiCandidateMessage);

const geminiBlockedPrefix = "Gemini request blocked with reason:";

export const geminiBlockedMessage = (blockReason: string) =>
  `${geminiBlockedPrefix} ${blockReason}`;

export const isGeminiBlockedError = (error: unknown) =>
  normalizeError(error).message.startsWith(geminiBlockedPrefix);

export const invalidGenJsonMessage =
  "genJson result did not match the requested schema";

export const validateAgainstSchema = <T extends ZodType>(
  zodType: T,
  result: unknown,
): z.infer<T> => {
  const parsed = zodType.safeParse(result);
  if (!parsed.success) {
    throw new Error(`${invalidGenJsonMessage}: ${parsed.error.message}`);
  }
  return parsed.data;
};

export const is403PermissionError = (error: unknown) => {
  const norm = normalizeError(error);
  if ("status" in norm && (norm as { status: number }).status === 403) {
    return true;
  }
  return norm.message.includes("403") &&
    norm.message.includes("PERMISSION_DENIED");
};

export const isMaxTokensError = (error: unknown) =>
  normalizeError(error).message.includes("MAX_TOKENS");

export const isRetryableError = (error: unknown) => {
  const norm = normalizeError(error);
  return (
    !isSyntheticTimeoutError(norm) &&
    (isServerError(norm) ||
      isRateLimitError(norm) ||
      isTransientFetchError(norm) ||
      norm instanceof SyntaxError ||
      isEmptyGeminiCandidateError(norm))
  );
};

export const geminiUploadJsonParseErrorMessage = "Unexpected end of JSON input";

export const isRetryableUploadError = (error: unknown) =>
  isRetryableError(error) || isTransientFetchError(error) ||
  error instanceof SyntaxError;

const emojiPattern =
  /\p{Emoji_Presentation}|\p{Extended_Pictographic}|\p{Regional_Indicator}/gu;

const maxEmojis = 100;

export const isEmojiFlood = (text: string) => {
  const emojiCount = text.match(emojiPattern)?.length ?? 0;
  if (emojiCount <= maxEmojis) return false;
  const nonWhitespaceCount = text.replace(/\s/g, "").length;
  return emojiCount / (nonWhitespaceCount || 1) > 0.25;
};

export const isRepetitionFlood = (text: string) => {
  const matches = text.matchAll(/(.{1,15}?)\1{29,}/gs);
  for (const match of matches) {
    const repeatedSeq = match[1];
    if (!/^[ \t\n\r\-=_*.~#|:;()[\]{}]+$/.test(repeatedSeq)) {
      return true;
    }
  }
  return false;
};

export const stripAnsi = (text: string): string => {
  const esc = "\\u" + "001b";
  const c1 = "\\u" + "009b";
  return text.replace(
    new RegExp(
      "[" + esc + c1 +
        "][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]",
      "g",
    ),
    "",
  );
};

// Models sometimes emit the same message body several times back to back in a
// single response, with arbitrary separators, occasional markdown emphasis
// variation between repetitions (e.g. ** vs *), and sometimes a final
// repetition truncated mid-way. Detect a whole-message repetition and collapse
// it to the first repetition, formatting intact.
const repetitionIgnoredChars = /[\s*_~`]/;

const buildContentWithMap = (text: string) => {
  const chars: string[] = [];
  const map: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (repetitionIgnoredChars.test(ch)) continue;
    chars.push(ch);
    map.push(i);
  }
  return { content: chars.join(""), map };
};

const minRepetitionUnitContentLength = 10;
const repetitionHeadLength = 30;

const countLeadingRepetitions = (content: string, unit: string): number =>
  content.startsWith(unit)
    ? 1 + countLeadingRepetitions(content.slice(unit.length), unit)
    : 0;

const headOccurrences = (
  content: string,
  head: string,
  from = 1,
): number[] => {
  const at = content.indexOf(head, from);
  return at === -1 ? [] : [at, ...headOccurrences(content, head, at + 1)];
};

export const collapseDuplicatedText = (text: string): string => {
  const trimmed = text.trim();
  if (trimmed.length < 80) return text;
  const { content, map } = buildContentWithMap(trimmed);
  if (content.length < 2 * minRepetitionUnitContentLength) return text;
  const head = content.slice(0, repetitionHeadLength);
  const tryUnitEnd = (unitEnd: number): string | undefined => {
    if (unitEnd < minRepetitionUnitContentLength) return undefined;
    const unit = content.slice(0, unitEnd);
    if (!/[\p{L}\p{N}]/u.test(unit)) return undefined;
    const repetitions = countLeadingRepetitions(content, unit);
    if (repetitions < 2) return undefined;
    const remainder = content.slice(repetitions * unitEnd);
    if (remainder.length > 0 && !unit.startsWith(remainder)) return undefined;
    return trimmed.slice(0, map[unitEnd - 1] + 1);
  };
  for (const unitEnd of headOccurrences(content, head)) {
    const collapsed = tryUnitEnd(unitEnd);
    if (collapsed !== undefined) return collapsed;
  }
  return text;
};

export const cleanActiveMemoryToolName = "clean_active_memory";
