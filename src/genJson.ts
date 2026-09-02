import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { empty } from "gamla";
import type { z, ZodType } from "zod/v4";
import type { MediaAttachment } from "./agent.ts";
import { geminiGenJsonFromConvo } from "./gemini.ts";
import { kimiGenJsonFromConvo } from "./kimiJson.ts";
import { openAiGenJsonFromConvo, structuredMsgs } from "./openai.ts";
import {
  invalidGenJsonMessage,
  isGeminiBlockedError,
  type ModelOpts,
  validateAgainstSchema,
} from "./utils.ts";

export { invalidGenJsonMessage };

const routeGeminiWithKimiBlockedFallback = <T extends ZodType>(
  opts: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
  attachments?: MediaAttachment[],
): Promise<z.infer<T>> =>
  geminiGenJsonFromConvo(opts, messages, zodType, attachments)
    .catch((e: unknown): Promise<z.infer<T>> =>
      empty(attachments ?? []) && isGeminiBlockedError(e)
        ? kimiGenJsonFromConvo(opts, messages, zodType)
        : Promise.reject(e)
    );

export const genJsonFromConvo = async <T extends ZodType>(
  opts: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
  attachments?: MediaAttachment[],
): Promise<z.infer<T>> => {
  const provider = opts.provider || "google";
  if (provider === "openai") {
    return validateAgainstSchema(
      zodType,
      await openAiGenJsonFromConvo(opts, messages, zodType),
    );
  }
  const result = await routeGeminiWithKimiBlockedFallback(
    opts,
    messages,
    zodType,
    attachments,
  );
  return validateAgainstSchema(zodType, result);
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

export const geminiGenJson = genJson;
