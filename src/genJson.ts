import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import type { z, ZodType } from "zod/v4";
import type { MediaAttachment } from "./agent.ts";
import { geminiGenJsonFromConvo } from "./gemini.ts";
import { openAiGenJsonFromConvo, structuredMsgs } from "./openai.ts";
import type { ModelOpts } from "./utils.ts";
import { assertNoScriptDrift } from "./scriptDriftGuard.ts";

const messagesToText = (messages: ChatCompletionMessageParam[]): string =>
  messages
    .map(({ content }) => typeof content === "string" ? content : "")
    .join("\n");

// Gemini-specific: guard against the model rewriting the input's language into
// a different writing system (homoglyph corruption) inside structured output.
// Throws so the caller's retry produces a clean generation instead of caching
// or surfacing corrupted text.
const guardGeminiScriptDrift = async <R>(
  inputText: string,
  result: R,
): Promise<R> => {
  await assertNoScriptDrift(inputText, JSON.stringify(result));
  return result;
};

export const genJsonFromConvo = async <T extends ZodType>(
  opts: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
  attachments?: MediaAttachment[],
): Promise<z.infer<T>> => {
  const provider = opts.provider || "google";
  if (provider === "openai") {
    return await openAiGenJsonFromConvo(opts, messages, zodType);
  }
  const result = await geminiGenJsonFromConvo(
    opts,
    messages,
    zodType,
    attachments,
  );
  if (opts.disableScriptDriftGuard) {
    return result;
  }
  return await guardGeminiScriptDrift(
    messagesToText(messages),
    result,
  );
};

import { context, type Injection } from "@uri/inject";

// deno-lint-ignore no-explicit-any
export const genJsonOverride: Injection<any> = context(() => null);

export const genJson =
  <T extends ZodType>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string, attachments?: MediaAttachment[]): Promise<z.infer<T>> => {
    const override = genJsonOverride.access();
    if (override) {
      return override(opts, systemMsg, zodType)(userMsg, attachments);
    }
    return genJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
      attachments,
    );
  };
