import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { empty } from "gamla";
import { z, type ZodType } from "zod/v4";
import type { MediaAttachment } from "./agent.ts";
import { decide, isDecisionField, isStringField } from "./decisionModel.ts";
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

const isRecord = (val: unknown): val is Record<string, unknown> =>
  typeof val === "object" && val !== null;

export const genJson =
  <T extends ZodType>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  async (
    userMsg: string,
    attachments?: MediaAttachment[],
  ): Promise<z.infer<T>> => {
    const override = genJsonOverride.access();
    if (override) {
      return override(opts, systemMsg, zodType)(userMsg, attachments);
    }

    if (attachments && !empty(attachments)) {
      return genJsonFromConvo(
        opts,
        structuredMsgs(systemMsg, userMsg),
        zodType,
        attachments,
      );
    }

    if (isDecisionField(zodType)) {
      try {
        return await decide(systemMsg, zodType)(userMsg);
      } catch {
        return genJsonFromConvo(
          opts,
          structuredMsgs(systemMsg, userMsg),
          zodType,
          attachments,
        );
      }
    }

    const isObj = isRecord(zodType) && isRecord(zodType.shape);
    if (isObj) {
      const shape = zodType.shape as Record<string, ZodType>;
      const stringKeys = Object.keys(shape).filter((k) =>
        isStringField(shape[k])
      );
      const decisionKeys = Object.keys(shape).filter((k) =>
        isDecisionField(shape[k])
      );
      const otherKeys = Object.keys(shape).filter((k) =>
        !isStringField(shape[k]) && !isDecisionField(shape[k])
      );

      if (empty(stringKeys) && empty(otherKeys) && !empty(decisionKeys)) {
        try {
          return await decide(systemMsg, zodType)(userMsg);
        } catch {
          return genJsonFromConvo(
            opts,
            structuredMsgs(systemMsg, userMsg),
            zodType,
            attachments,
          );
        }
      }

      if ((!empty(stringKeys) || !empty(otherKeys)) && !empty(decisionKeys)) {
        try {
          const decisionShape = Object.fromEntries(
            decisionKeys.map((k) => [k, shape[k]]),
          );
          const generativeShape = Object.fromEntries(
            [...stringKeys, ...otherKeys].map((k) => [k, shape[k]]),
          );
          const decisionSchema = z.object(decisionShape);
          const generativeSchema = z.object(generativeShape);

          const [decisionRes, generativeRes] = await Promise.all([
            decide(systemMsg, decisionSchema)(userMsg),
            genJsonFromConvo(
              opts,
              structuredMsgs(systemMsg, userMsg),
              generativeSchema,
              attachments,
            ),
          ]);

          return validateAgainstSchema(zodType, {
            ...decisionRes,
            ...generativeRes,
          });
        } catch {
          return genJsonFromConvo(
            opts,
            structuredMsgs(systemMsg, userMsg),
            zodType,
            attachments,
          );
        }
      }
    }

    return genJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
      attachments,
    );
  };

export const geminiGenJson = genJson;
