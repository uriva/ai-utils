import { context, type Injection, type Injector } from "@uri/inject";
import { OpenAI } from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/index.mjs";
import { z, type ZodType } from "zod/v4";
import { makeCache } from "./cacher.ts";
import { type ModelOpts, validateAgainstSchema } from "./utils.ts";

export const kimiModelVersion = "kimi-k2.7-code";

const kimiApiKeyInjection: Injection<() => string> = context((): string => {
  throw new Error("no kimi API key injected");
});

export const injectKimiToken = (token: string): Injector =>
  kimiApiKeyInjection.inject(() => token);

export const kimiClient = () =>
  new OpenAI({
    apiKey: kimiApiKeyInjection.access(),
    baseURL: "https://api.moonshot.ai/v1",
  });

const createKimiCompletion = (
  req: ChatCompletionCreateParamsNonStreaming,
): Promise<OpenAI.Chat.Completions.ChatCompletion> =>
  kimiClient().chat.completions.create(req);

const kimiChatCompletionInjection: Injection<typeof createKimiCompletion> =
  context(createKimiCompletion);

export const injectKimiChatCompletion = kimiChatCompletionInjection.inject;

export const emptyKimiCompletionMessage = "Kimi returned an empty completion";

const schemaAdherenceInstruction = (zodType: ZodType) =>
  `Reply with a single JSON value and nothing else. It must conform to this JSON Schema:\n${
    JSON.stringify(z.toJSONSchema(zodType))
  }`;

export const kimiGenJsonFromConvo = async <T extends ZodType>(
  _opts: ModelOpts,
  messages: ChatCompletionMessageParam[],
  zodType: T,
): Promise<z.infer<T>> => {
  const cachedCall = makeCache("kimiTypedCompletion-v1")(
    (req: ChatCompletionCreateParamsNonStreaming) =>
      kimiChatCompletionInjection.access(req),
  );
  const { choices } = await cachedCall({
    model: kimiModelVersion,
    messages: [
      { role: "system", content: schemaAdherenceInstruction(zodType) },
      ...messages,
    ],
    response_format: { type: "json_object" },
  });
  const content = choices[0]?.message?.content;
  if (!content) throw new Error(emptyKimiCompletionMessage);
  return validateAgainstSchema(zodType, JSON.parse(content));
};
