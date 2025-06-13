import { context } from "context-inject";
import { map, pipe, prop } from "gamla";
import { OpenAI } from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/index.mjs";
import z, { type ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import {
  aiRefusesToAdhereTyping,
  type FnToSameFn,
  type ModelOpts,
  type TokenInjection,
} from "./utils.ts";

const tokenInjection: TokenInjection = context((): string => {
  throw new Error("no openai token injected");
});

export const injectOpenAiToken = (token: string): FnToSameFn =>
  tokenInjection.inject(() => token);

export const replaceSystem =
  (replacement: string) =>
  (message: ChatCompletionMessageParam): ChatCompletionMessageParam => ({
    ...message,
    // @ts-expect-error not sure why this is not working
    role: message.role === "system" ? replacement : message.role,
  });

const extractLastMarkdownJsonBlock = (text: string): string | null => {
  const regex = /```json([\s\S]*?)```/g;
  const matches = [...text.matchAll(regex)];
  return matches.length > 0 ? matches[matches.length - 1][1].trim() : null;
};

type StructuredInference = <T extends ZodType>(
  opts: ModelOpts,
  msgs: ChatCompletionMessageParam[],
  zodType: T,
) => Promise<z.infer<T>>;

export const extractJson =
  <T extends ZodType>(genMethod: StructuredInference) =>
  (typing: T) =>
  async (text: string): Promise<z.infer<T>> => {
    const jsonText = extractLastMarkdownJsonBlock(text) ?? text;
    try {
      return JSON.parse(jsonText) as z.infer<T>;
    } catch (_) {
      return await genMethod(
        { thinking: false, mini: true },
        structuredMsgs("Fix this json.", text),
        typing,
      );
    }
  };

export const openAiGenJsonFromConvo = async <
  // deno-lint-ignore no-explicit-any
  T extends ZodType<any, any, any>,
>(
  { mini, thinking }: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const cachedCall = makeCache(
    "openAiTypedCompletion",
  )((opts: ChatCompletionCreateParamsNonStreaming) =>
    new OpenAI({ apiKey: tokenInjection.access() }).chat.completions
      .parse(opts)
  );
  const { choices } = await cachedCall({
    model: thinking
      ? (mini ? "o4-mini" : "o3")
      : (mini ? "gpt-4.1-mini" : "gpt-4.1"),
    messages,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "structured-response",
        strict: true,
        schema: z.toJSONSchema(zodType),
      },
    },
  });
  if (!choices[0].message.content) {
    aiRefusesToAdhereTyping();
    throw new Error("AI refused to return content");
  }
  return (thinking ? extractJson(openAiGenJsonFromConvo)(zodType) : JSON.parse)(
    choices[0].message.content,
  );
};

export const structuredMsgs = (
  systemMsg: string,
  userMsg: string,
): ChatCompletionMessageParam[] => [
  { role: "system", content: systemMsg },
  { role: "user", content: userMsg },
];

export const openAiGenJson =
  <T extends ZodType>(opts: ModelOpts, systemMsg: string, zodType: T) =>
  (userMsg: string): Promise<z.infer<T>> =>
    openAiGenJsonFromConvo(
      opts,
      structuredMsgs(systemMsg, userMsg),
      zodType,
    );

const numberedItem = (x: string, i: number) => `(${i}): ${x}`;

const makePromptForMatching = (
  setA: string[],
  setB: string[],
  reasoningNotes: string,
) =>
  `You need to match between the following groups.

Group A:

${setA.map(numberedItem).join("\n\n")}


Group B:

${setB.map(numberedItem).join("\n\n")}


Notes for reasoning: ${reasoningNotes}

The output should be an index from group A, then index from group B then reasoning (why you think this is a good match). please double check the indices.`;

const OutputZod = z.object({
  results: z.array(z.tuple([z.number(), z.number(), z.string()])),
});

type Output = z.infer<typeof OutputZod>;

export const openAiMatching = <X, Y>(
  opts: ModelOpts,
  reasoningNotes: string,
  xToStr: (y: X) => string,
  yToStr: (y: Y) => string,
) =>
(xs: X[], ys: Y[]): Promise<[X, Y, string][]> =>
  pipe(
    makePromptForMatching,
    openAiGenJson(opts, "", OutputZod),
    prop<Output>()("results"),
    map(([indA, indB, reasoning]: Output["results"][0]) =>
      [xs[indA], ys[indB], reasoning] satisfies [X, Y, string]
    ),
  )(xs.map(xToStr), ys.map(yToStr), reasoningNotes);
