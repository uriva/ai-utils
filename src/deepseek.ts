import { context } from "npm:context-inject@0.0.3";
import { coerce } from "npm:gamla@122.0.0";
import { OpenAI } from "npm:openai@4.71.1";
import type {
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletionMessageParam,
} from "npm:openai@4.71.1/resources/index.mjs";
import type z from "npm:zod@3.24.3";
import type { ZodSchema } from "npm:zod@3.24.3";
import { makeCache } from "./cacher.ts";
import { extractJson } from "./openai.ts";
import {
    appendTypingInstruction,
    FnToSameFn,
    ModelOpts,
    TokenInjection,
} from "./utils.ts";

const tokenInjection: TokenInjection = context((): string => {
    throw new Error("deepseek token not injected");
});

export const injectDeepSeekToken = (token: string): FnToSameFn =>
    tokenInjection.inject(() => token);

const combineConsecutiveSystemMessages = (
    messages: ChatCompletionMessageParam[],
) => {
    const newMessages = [];
    for (const message of messages) {
        if (
            newMessages.length > 0 &&
            newMessages[newMessages.length - 1].role === "system" &&
            message.role === "system"
        ) {
            newMessages[newMessages.length - 1].content +=
                `\n\n${message.content}`;
        } else {
            newMessages.push({ ...message });
        }
    }
    return newMessages;
};

const makeSureLastMessageIsUser = (
    messages: ChatCompletionMessageParam[],
): ChatCompletionMessageParam[] => {
    if (messages[messages.length - 1].role === "user") {
        return messages;
    }
    return [...messages, { role: "user", content: "" }];
};

export const deepSeekGenJsonFromConvo = async <T extends ZodSchema>(
    { thinking }: ModelOpts,
    messages: ChatCompletionMessageParam[],
    zodType: T,
): Promise<z.infer<T>> => {
    const deepSeekCachedCall = makeCache(
        "deepSeekTypedCompletion-feb-2-2025",
    )((opts: ChatCompletionCreateParamsNonStreaming) =>
        new OpenAI({
            apiKey: tokenInjection.access(),
            baseURL: "https://api.deepseek.com/",
        })
            .beta.chat.completions.parse(opts)
    );
    const { choices } = await deepSeekCachedCall({
        model: thinking ? "deepseek-reasoner" : "deepseek-chat",
        messages: pipe(
            makeSureLastMessageIsUser,
            combineConsecutiveSystemMessages,
            appendTypingInstruction(zodType, "system"),
        )(messages),
    });
    return extractJson(deepSeekGenJsonFromConvo)(zodType)(
        coerce(choices[0].message.content),
    );
};
