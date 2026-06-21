import { type EitherOutput, type Func, throwerCatcher } from "gamla";

export type ModelOpts = {
  mini: boolean;
  maxOutputTokens?: number;
  provider?: "google" | "openai" | "gemini";
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

export const isRetryableError = (error: unknown) =>
  !isSyntheticTimeoutError(error) &&
  (isServerError(error) || isRateLimitError(error));

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

export const collapseDuplicatedText = (text: string): string => {
  const trimmed = text.trim();
  const len = trimmed.length;
  if (len < 80) return text;
  const halfLen = Math.floor(len / 2);
  const firstHalf = trimmed.slice(0, halfLen).trim();
  const secondHalf = trimmed.slice(halfLen).trim();
  if (firstHalf === secondHalf) {
    const origLen = text.length;
    const origHalfLen = Math.floor(origLen / 2);
    const origFirst = text.slice(0, origHalfLen);
    const origSecond = text.slice(origHalfLen);
    if (origFirst.trim() === origSecond.trim()) {
      return origFirst;
    }
    return firstHalf;
  }
  return text;
};

export const cleanActiveMemoryToolName = "clean_active_memory";
