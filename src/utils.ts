import { ZodToTypescript } from "npm:@duplojs/zod-to-typescript@0.4.0";
import {
  append,
  type EitherOutput,
  type Func,
  pipe,
  throwerCatcher,
} from "npm:gamla@124.0.0";
import type { ChatCompletionMessageParam } from "npm:openai@4.71.1/resources/index.mjs";
import { type ZodSchema } from "npm:zod@3.24.2";

export const appendTypingInstruction: <T extends ZodSchema>(
  zodType: T,
  role: "system" | "user" | "assistant",
) => (arr: ChatCompletionMessageParam[]) => ChatCompletionMessageParam[] = pipe(
  <T extends ZodSchema>(
    zodType: T,
    role: "system" | "user" | "assistant",
  ) => ({
    role,
    content:
      `The output should be a valid json, as short as possible, no redundant whitespace, adhering to this typing: ${
        ZodToTypescript.convert(zodType, { name: "Schema" })
      }`,
  }),
  append<ChatCompletionMessageParam>,
);

export type ModelOpts = { thinking: boolean; mini: boolean };

const typeAdherenceError = throwerCatcher();

export const aiRefusesToAdhereTyping = typeAdherenceError.thrower;
export const catchAiRefusesToAdhereToTyping: <G extends Func>(
  fallback: G,
) => <F extends Func>(f: F) => (...xs: Parameters<F>) => EitherOutput<F, G> =
  typeAdherenceError.catcher;

export type TokenInjection = {
  inject: (fn: () => string) => FnToSameFn;
  access: () => string;
};

export type FnToSameFn = <F extends Func>(f: F) => F;
