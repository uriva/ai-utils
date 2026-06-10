import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import type { z, ZodType } from "zod/v4";
import { geminiGenJsonFromConvo } from "./gemini.ts";
import { openAiGenJsonFromConvo, structuredMsgs } from "./openai.ts";
import type { ModelOpts } from "./utils.ts";

export const genJsonFromConvo = async <T extends ZodType>(
  opts: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const provider = opts.provider || "google";
  if (provider === "openai") {
    return await openAiGenJsonFromConvo(opts, messages, zodType);
  }
  return await geminiGenJsonFromConvo(opts, messages, zodType);
};

export const genJson =
  <T extends ZodType>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string): Promise<z.infer<T>> =>
    genJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
    );
