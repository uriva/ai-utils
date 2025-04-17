import { map, pipe, prop } from "npm:gamla@122.0.0";
import { OpenAI } from "npm:openai@4.71.1";
import { zodResponseFormat } from "npm:openai@4.71.1/helpers/zod";
import type {
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletionMessageParam,
} from "npm:openai@4.71.1/resources/index.mjs";
import z, { type ZodSchema } from "npm:zod@3.24.2";
import { aiRefusesToAdhereTyping, ModelOpts, TokenInjection } from "./utils.ts";

import { context } from "npm:context-inject@0.0.3";
import { makeCache } from "./cacher.ts";

const tokenInjection: TokenInjection = context((): string => {
    throw new Error("no openai token injected");
});

export const injectOpenAiToken = (token: string) =>
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

type StructuredInference = <T extends ZodSchema>(
    opts: ModelOpts,
    msgs: ChatCompletionMessageParam[],
    zodType: T,
) => Promise<z.infer<T>>;

export const extractJson =
    <T extends ZodSchema>(genMethod: StructuredInference) =>
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

// https://github.com/openai/openai-node/issues/1365
const cleanSchema = (
    schema: ReturnType<typeof zodResponseFormat>,
): ReturnType<typeof zodResponseFormat> => {
    if (Array.isArray(schema)) {
        // @ts-expect-error ignore
        return schema
            .map(cleanSchema)
            .filter((item) =>
                JSON.stringify(item) !== JSON.stringify({ not: {} })
            );
    }

    if (typeof schema === "object" && schema !== null) {
        return Object.fromEntries(
            Object.entries(schema)
                .map((
                    [key, value],
                ) => [
                    key,
                    key === "anyOf" ? cleanSchema(value) : cleanSchema(value),
                ])
                .filter(([, value]) => value !== undefined),
        );
    }

    return schema;
};

export const openAiGenJsonFromConvo = async <T extends ZodSchema>(
    { mini, thinking }: ModelOpts,
    messages: ChatCompletionMessageParam[],
    zodType: T,
): Promise<z.infer<T>> => {
    const cachedCall = makeCache(
        "openAiTypedCompletion",
    )((opts: ChatCompletionCreateParamsNonStreaming) =>
        new OpenAI({ apiKey: tokenInjection.access() }).beta.chat.completions
            .parse(opts)
    );
    const { choices } = await cachedCall({
        model: thinking
            ? (mini ? "o3-mini" : "o1")
            : (mini ? "gpt-4o-mini" : "gpt-4o"),
        messages: messages,
        response_format: pipe(zodResponseFormat, cleanSchema)(
            zodType,
            "event-bot-response",
            {},
        ),
    });
    if (!choices[0].message.content) {
        aiRefusesToAdhereTyping();
        throw new Error("AI refused to return content");
    }
    return (thinking
        ? extractJson(openAiGenJsonFromConvo)(zodType)
        : JSON.parse)(
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
    <T extends ZodSchema>(opts: ModelOpts, systemMsg: string, zodType: T) =>
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
) => `You need to match between the following groups.

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
