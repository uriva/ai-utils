import { pipe, throwerCatcher, append } from "npm:gamla@122.0.0";
import type { ChatCompletionMessageParam } from "npm:openai@4.71.1/resources/index.mjs";
import { type ZodSchema } from "npm:zod@3.24.2";
import { zodToTs } from "./zodToTsStr.ts";

export const appendTypingInstruction = pipe(
  <T extends ZodSchema>(
    zodType: T,
    role: "system" | "user" | "assistant",
  ) => ({
    role,
    content:
      `The output should be a valid json, as short as possible, no redundant whitespace, adhering to this typing: ${(zodToTs(
        { schema: zodType, name: "Schema" },
      ))}`,
  }),
  append<ChatCompletionMessageParam>,
);

export type ModelOpts = { thinking: boolean; mini: boolean };

export const {
  thrower: aiRefusesToAdhereTyping,
  catcher: catchAiRefusesToAdhereToTyping,
} = throwerCatcher();
