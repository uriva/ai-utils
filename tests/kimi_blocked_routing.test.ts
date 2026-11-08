import { assertEquals, assertRejects } from "@std/assert";
import type { GenerateContentResponse } from "@google/genai";
import type { Injector } from "@uri/inject";
import {
  genJson,
  type HistoryEvent,
  injectAccessHistory,
  injectCacher,
  injectGeminiToken,
  injectKimiToken,
  injectOutputEvent,
  invalidGenJsonMessage,
  participantUtteranceTurn,
  runAgent,
  z,
} from "../mod.ts";
import { injectGeminiGenerateContent } from "../src/gemini.ts";
import { injectGeminiSdkExchange } from "../src/geminiAgent.ts";
import { injectKimiChatCompletion, kimiModelVersion } from "../src/kimiJson.ts";

// When Gemini blocks a request outright (e.g. PROHIBITED_CONTENT on the prompt),
// retrying Gemini or switching to its fallback model can never succeed: every
// variant of the same payload is rejected. genJson must instead detect the
// block reason at parse time and route the structured-output call to a
// different provider (Kimi) exactly once, so callers get a schema-valid result
// instead of a misleading INVALID_ARGUMENT error after a pointless retry spiral.

const memoryCacher =
  (store: Record<string, unknown>) => (_cacheId: string): Injector =>
    ((f: (...args: unknown[]) => Promise<unknown>) =>
    async (...args: unknown[]) => {
      const key = JSON.stringify(args);
      if (key in store) return store[key];
      const result = await f(...args);
      store[key] = result;
      return result;
    }) as Injector;

const blockedResponse = {
  promptFeedback: { blockReason: "PROHIBITED_CONTENT" },
  // deno-lint-ignore no-explicit-any
} as any as GenerateContentResponse;

const stopResponse = (text: string) =>
  ({
    candidates: [{ finishReason: "STOP" }],
    text,
    // deno-lint-ignore no-explicit-any
  }) as any as GenerateContentResponse;

const kimiCompletion = (content: string | null) => ({
  id: "kimi-fake",
  object: "chat.completion" as const,
  created: 0,
  model: kimiModelVersion,
  choices: [{
    index: 0,
    message: {
      role: "assistant" as const,
      content,
      refusal: null,
    },
    finish_reason: "stop" as const,
    logprobs: null,
  }],
});

const schema = z.object({ answer: z.string() });

const runGenJson = (
  // deno-lint-ignore no-explicit-any
  attachments?: any[],
): Promise<z.infer<typeof schema>> =>
  genJson({ provider: "google", tier: "pro" }, "sys", schema)(
    "user",
    attachments,
  );

Deno.test("a blocked Gemini prompt routes to Kimi instead of retrying", async () => {
  let geminiCalls = 0;
  let kimiCalls = 0;
  const run = async () => {
    const result = await runGenJson();
    assertEquals(result, { answer: "kimi-ok" });
    assertEquals(geminiCalls, 1);
    assertEquals(kimiCalls, 1);
  };
  await injectGeminiToken("unused-gemini-faked")(
    injectKimiToken("unused-kimi-faked")(
      injectCacher(memoryCacher({}))(
        injectGeminiGenerateContent(() => {
          geminiCalls++;
          return Promise.resolve(blockedResponse);
        })(
          injectKimiChatCompletion((_req) => {
            kimiCalls++;
            return Promise.resolve(
              // deno-lint-ignore no-explicit-any
              kimiCompletion('{"answer":"kimi-ok"}') as any,
            );
          })(run),
        ),
      ),
    ),
  )();
});

Deno.test("an unblocked Gemini response never touches Kimi", async () => {
  let kimiCalls = 0;
  const run = async () => {
    const result = await runGenJson();
    assertEquals(result, { answer: "gemini-ok" });
    assertEquals(kimiCalls, 0);
  };
  await injectGeminiToken("unused-gemini-faked")(
    injectKimiToken("unused-kimi-faked")(
      injectCacher(memoryCacher({}))(
        injectGeminiGenerateContent(() =>
          Promise.resolve(stopResponse('{"answer":"gemini-ok"}'))
        )(
          injectKimiChatCompletion((_req) => {
            kimiCalls++;
            return Promise.resolve(kimiCompletion(null));
          })(run),
        ),
      ),
    ),
  )();
});

Deno.test("a blocked prompt with media attachments rethrows instead of losing the attachments", async () => {
  let kimiCalls = 0;
  const attachments = [{
    kind: "file",
    fileUri: "https://example.com/picture.png",
    mimeType: "image/png",
  }];
  const run = async () => {
    const error = await assertRejects(
      () => runGenJson(attachments),
      Error,
      "PROHIBITED_CONTENT",
    );
    assertEquals((error as Error).message.includes("blocked"), true);
    assertEquals(kimiCalls, 0);
  };
  await injectGeminiToken("unused-gemini-faked")(
    injectKimiToken("unused-kimi-faked")(
      injectCacher(memoryCacher({}))(
        injectGeminiGenerateContent(() => Promise.resolve(blockedResponse))(
          injectKimiChatCompletion((_req) => {
            kimiCalls++;
            return Promise.resolve(kimiCompletion('{"answer":"x"}'));
          })(run),
        ),
      ),
    ),
  )();
});

Deno.test("a schema-violating Kimi fallback result surfaces a validation error", async () => {
  const run = async () => {
    await assertRejects(
      () => runGenJson(),
      Error,
      invalidGenJsonMessage,
    );
  };
  await injectGeminiToken("unused-gemini-faked")(
    injectKimiToken("unused-kimi-faked")(
      injectCacher(memoryCacher({}))(
        injectGeminiGenerateContent(() => Promise.resolve(blockedResponse))(
          injectKimiChatCompletion((_req) =>
            Promise.resolve(
              // deno-lint-ignore no-explicit-any
              kimiCompletion('{"wrongField":1}') as any,
            )
          )(run),
        ),
      ),
    ),
  )();
});

Deno.test("a Gemini safety-blocked runAgent turn routes to Kimi instead of emitting English safetyWarningText", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "benign conversational turn that triggers false positive block",
    }),
  ];
  let geminiCalls = 0;
  let kimiCalls = 0;
  await injectGeminiToken("unused-gemini-faked")(
    injectKimiToken("unused-kimi-faked")(
      injectGeminiSdkExchange(() => {
        geminiCalls++;
        return Promise.resolve({
          parts: [],
          finishReason: undefined,
          usageMetadata: undefined,
          promptBlockReason: "PROHIBITED_CONTENT",
        });
      })(
        injectKimiChatCompletion(() => {
          kimiCalls++;
          return Promise.resolve(
            // deno-lint-ignore no-explicit-any
            kimiCompletion("Hola, estoy aquí para ayudarte.") as any,
          );
        })(
          injectAccessHistory(() => Promise.resolve(history))(
            injectOutputEvent((e) => {
              history.push(e);
              return Promise.resolve();
            })(() =>
              runAgent({
                prompt: "You are a helpful Spanish coach.",
                tools: [],
                timezoneIANA: "UTC",
                maxIterations: 2,
              })
            ),
          ),
        ),
      ),
    ),
  )();
  assertEquals(geminiCalls, 1);
  assertEquals(kimiCalls, 1);
  const utterances = history.filter((e) => e.type === "own_utterance");
  assertEquals(utterances.length, 1);
  assertEquals(
    utterances[0].type === "own_utterance" && utterances[0].text,
    "Hola, estoy aquí para ayudarte.",
  );
});
