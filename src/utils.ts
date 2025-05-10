import { ZodToTypescript } from "@duplojs/zod-to-typescript";
import {
  append,
  type EitherOutput,
  type Func,
  pipe,
  throwerCatcher,
} from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { type ZodSchema } from "zod";

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

export type SomethingInjection<T extends Func> = {
  inject: (fn: T) => FnToSameFn;
  access: T;
};

export type TokenInjection = SomethingInjection<() => string>;

export type FnToSameFn = <F extends Func>(f: F) => F;
