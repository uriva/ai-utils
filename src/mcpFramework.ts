import { context } from "https://deno.land/x/context_inject@0.0.3/src/index.ts";
import {
    coerce,
    empty,
    init,
    last,
    logAfter,
    map,
    nonempty,
    pipe,
} from "npm:gamla@122.0.0";
import {
    type Content,
    FunctionCall,
    FunctionDeclarationSchema,
    FunctionResponsePart,
    type GenerateContentRequest,
    GenerateContentResult,
    GoogleGenerativeAI,
    type ModelParams,
} from "npm:@google/generative-ai@0.21.0";
import { makeCache } from "./cacher.ts";
import { z, ZodSchema } from "npm:zod";
import { zodToJsonSchema } from "npm:zod-to-json-schema";
import { accessGeminiToken } from "./gemini.ts";

// deno-lint-ignore no-explicit-any
const isRedundantAnyMember = (x: any) =>
    Object.keys(x).length === 1 && typeof (x.not) === "object" &&
    Object.keys(x.not).length === 0;

// deno-lint-ignore no-explicit-any
const removeAdditionalProperties = <T>(obj: Record<string, any>) => {
    if (typeof obj === "object" && obj !== null) {
        let newObj = { ...obj };
        if (obj.anyOf) {
            // deno-lint-ignore no-explicit-any
            newObj.anyOf = obj.anyOf.filter((x: any) =>
                x.type !== "null" && !isRedundantAnyMember(x)
            ).map(removeAdditionalProperties);
        }
        if (newObj.anyOf?.length === 1) newObj = newObj.anyOf[0];
        if (Array.isArray(obj.type)) {
            if (obj.type.find((x: string) => x === "null")) {
                newObj.nullable = true;
            }
            newObj.type = obj.type.find((x) => x !== "null");
        }
        if ("additionalProperties" in newObj) {
            newObj.additionalProperties = undefined;
        }
        for (const key in newObj) {
            if (key in newObj) {
                if (Array.isArray(newObj[key])) {
                    newObj[key] = newObj[key].map(removeAdditionalProperties);
                } else if (
                    typeof newObj[key] === "object" && newObj[key] !== null
                ) {
                    newObj[key] = removeAdditionalProperties(newObj[key]);
                }
            }
        }
        return newObj;
    }
    return obj;
};

// deno-lint-ignore no-explicit-any
export const zodToGeminiParameters = (zodObj: any) => {
    const jsonSchema = removeAdditionalProperties(zodToJsonSchema(zodObj));
    // deno-lint-ignore no-unused-vars
    const { $schema, ...rest } = jsonSchema;
    return rest as FunctionDeclarationSchema;
};

export const systemUser = "system";

export type Action<T extends ZodSchema, O> = {
    description: string;
    name: string;
    parameters: T;
    handler: (params: z.infer<T>) => Promise<O>;
};

const callGemini = (modelParams: ModelParams) =>
    makeCache("gemini response with function calls")((
        req: GenerateContentRequest,
    ) => new GoogleGenerativeAI(accessGeminiToken())
        .getGenerativeModel(modelParams).generateContent(req).then((
            { response }: GenerateContentResult,
        ) => ({
            text: response.text(),
            functionCalls: response.functionCalls() ?? [],
        }))
    );

export type BotSpec = {
    // deno-lint-ignore no-explicit-any
    actions: Action<any, any>[];
    prompt: string;
    botNameInHistory: string;
};

const combineConsecutiveModelMessages = (acc: Content[], curr: Content) =>
    nonempty(acc) && last(acc).role === curr.role
        ? [...init(acc), {
            ...last(acc),
            parts: [...last(acc).parts, ...curr.parts],
        }]
        : [...acc, curr];

// deno-lint-ignore no-explicit-any
const actionToTool = ({ name, description, parameters }: Action<any, any>) => ({
    name,
    description,
    parameters: zodToGeminiParameters(parameters),
});

const geminiInput = (
    systemInstruction: string,
    // deno-lint-ignore no-explicit-any
    actions: Action<any, any>[],
    contents: Content[],
): GenerateContentRequest => ({
    systemInstruction,
    tools: [{ functionDeclarations: actions.map(actionToTool) }],
    contents,
});

const callToResult =
    // deno-lint-ignore no-explicit-any
    (actions: Action<any, any>[]) =>
    async <T extends ZodSchema, O>(
        { name, args }: FunctionCall,
    ): Promise<FunctionResponsePart> => {
        const { handler, parameters }: Action<T, O> = coerce(
            actions.find(({ name: n }) => n === name),
        );
        return {
            functionResponse: {
                name,
                response: {
                    result: await handler(parameters.parse(args) as T),
                },
            },
        };
    };

export const makeBot = async (
    { actions, prompt, botNameInHistory }: BotSpec,
) => {
    let c = 0;
    const contents = (await getHistory()).map(({ from, text }) => ({
        role: from !== botNameInHistory && from !== systemUser
            ? "user"
            : "model",
        parts: [{ text }],
    })).reduce(combineConsecutiveModelMessages, []);
    const thoughts: Content[] = [];
    while (true) {
        c++;
        if (c > 5) throw new Error("Too many iterations");
        const { text, functionCalls } = await pipe(
            geminiInput,
            logAfter(callGemini({ model: "gemini-2.5-pro-preview-03-25" })),
        )(prompt, actions, [...contents, ...thoughts]);
        console.log("model replied", text);
        console.log("model called", JSON.stringify(functionCalls, null, 2));
        if (text) await reply(text);
        const calls = functionCalls ?? [];
        const results = await map(callToResult(actions))(calls);
        for (let i = 0; i < results.length; i++) {
            thoughts.push({
                role: "model",
                parts: [{ functionCall: calls[i] }],
            });
            thoughts.push({ role: "user", parts: [results[i]] });
        }
        if (empty(functionCalls)) return;
    }
};

export const { access: getHistory, inject: injectAccessHistory } = context(
    (): Promise<HistoryEvent[]> => {
        throw new Error("History not injected");
    },
);

export const { access: reply, inject: injectReply } = context(
    (_text: string): Promise<void> => {
        throw new Error("Reply not injected");
    },
);

export type HistoryEvent = { text: string; from: string; time: number };

export type MessageDraft = { from: string; text: string };
