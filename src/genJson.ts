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

const maxScriptDriftRerolls = 2;

export const invalidGenJsonMessage =
  "genJson result did not match the requested schema";

const validateAgainstSchema = <T extends ZodType>(
  zodType: T,
  result: unknown,
): z.infer<T> => {
  const parsed = zodType.safeParse(result);
  if (!parsed.success) {
    throw new Error(`${invalidGenJsonMessage}: ${parsed.error.message}`);
  }
  return parsed.data;
};

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
  for (let attempt = 0; attempt <= maxScriptDriftRerolls; attempt++) {
    try {
      const result = await geminiGenJsonFromConvo(
        opts,
        messages,
        zodType,
        attachments,
      );
      return await guardGeminiScriptDrift(
        messagesToText(messages),
        validateAgainstSchema(zodType, result),
      );
    } catch (e) {
      if (
        e instanceof Error &&
        "scriptDrift" in e &&
        attempt < maxScriptDriftRerolls
      ) {
        console.warn(
          `Script drift detected in genJson (homoglyph corruption) on attempt ${attempt}. Retrying...`,
        );
        continue;
      }
      throw e;
    }
  }
  throw new Error("Unreachable");
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
