import { coerce, empty, map, pipe, remove, retry } from "npm:gamla@122.0.0";
import {
    type Content,
    type GenerateContentRequest,
    GoogleGenerativeAI,
    type ModelParams,
} from "npm:@google/generative-ai@0.21.0";
import type { ChatCompletionMessageParam } from "npm:openai@4.71.1/resources/index.mjs";
import type { z, ZodSchema } from "npm:zod@3.24.2";
import { appendTypingInstruction, ModelOpts } from "./utils.ts";
import { makeCache } from "./cacher.ts";
import { zodToGeminiParameters } from "./mcpFramework.ts";
import { extractJson, replaceSystem, structuredMsgs } from "./openai.ts";
import { context } from "npm:context-inject@0.0.3";

const token = context((): string => {
    throw new Error("no gemini token injected");
});

export const accessGeminiToken = token.access;
export const injectGeminiToken = token.inject;

const openAiToGeminiMessage = pipe(
    map((
        { role, content }: ChatCompletionMessageParam,
    ): Content => ({
        role: role === "user" ? role : "model",
        parts: [{
            text: typeof content === "string"
                ? content
                : coerce(content?.toString()),
        }]
            .filter((x) => x.text),
    })),
    remove(({ parts }: Content) => empty(parts)),
);

const cachedCall = makeCache(
    "geminiCompletionResponseText",
)((modelParams: ModelParams, req: GenerateContentRequest) =>
    new GoogleGenerativeAI(token.access()).getGenerativeModel(modelParams)
        .generateContent(req).then((x) => x.response.text())
);

export const geminiGenJsonFromConvo: <T extends ZodSchema>(
    { thinking, mini }: ModelOpts,
    messages: ChatCompletionMessageParam[],
    zodType: T,
) => Promise<z.infer<T>> = retry(
    10000,
    2,
    async <T extends ZodSchema>(
        { thinking, mini }: ModelOpts,
        messages: ChatCompletionMessageParam[],
        zodType: T,
    ): Promise<z.infer<T>> => {
        if (!thinking) {
            return (JSON.parse(
                await cachedCall(
                    {
                        model: mini
                            ? "gemini-2.5-pro-preview-03-25"
                            : "gemini-2.5-pro-preview-03-25",
                        generationConfig: {
                            responseMimeType: "application/json",
                            responseSchema: zodToGeminiParameters(zodType),
                        },
                    },
                    {
                        contents: pipe(
                            map(replaceSystem("assistant")),
                            openAiToGeminiMessage,
                        )(
                            messages,
                        ),
                    },
                ),
            )) as z.infer<T>;
        }
        return extractJson(geminiGenJsonFromConvo)(zodType)(
            await cachedCall(
                { model: "gemini-2.5-pro-preview-03-25" },
                {
                    contents: pipe(
                        map(replaceSystem("assistant")),
                        appendTypingInstruction(zodType, "assistant"),
                        openAiToGeminiMessage,
                    )(messages),
                },
            ),
        );
    },
);

export const geminiAiGenJson =
    <T extends ZodSchema>(opts: ModelOpts, systemMsg: string, zodType: T) =>
    (userMsg: string): Promise<z.TypeOf<T>> =>
        geminiGenJsonFromConvo(
            opts,
            structuredMsgs(systemMsg, userMsg),
            zodType,
        );
