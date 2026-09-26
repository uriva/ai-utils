import { assert, assertEquals } from "@std/assert";
import {
  type GenerateContentParameters,
  type GenerateContentResponse,
  ThinkingLevel,
} from "@google/genai";
import { runAgent } from "../mod.ts";
import {
  getStreamThinkingChunk,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  geminiGenJsonFromConvo,
  geminiThinkingConfig,
  injectGeminiGenerateContent,
} from "../src/gemini.ts";
import { pipe } from "gamla";
import { z } from "zod/v4";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

runForAllProviders(
  "agent returns own_thought events when thinking is enabled",
  async (runAgent) => {
    if (
      Deno.env.get("TEST_PROVIDER") === "google" ||
      Deno.env.get("TEST_PROVIDER") === "gemini"
    ) return;
    const mockHistory = [
      participantUtteranceTurn({
        name: "user",
        text: "What is 137 * 248? Think step by step.",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "You are a helpful assistant. Think carefully before answering.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const thoughts = mockHistory.filter((e) => e.type === "own_thought");
    assert(
      thoughts.length > 0,
      `Expected at least one own_thought event from thinking, but got none. Events: ${
        mockHistory.map((e) => e.type).join(", ")
      }`,
    );
  },
);

// Streaming is a contract between runAgent and the injected callModel,
// independent of provider SDK. Injecting a fake callModel that fires chunks
// during the call lets us test the contract deterministically and provider-
// agnostically.
Deno.test(
  "onStreamThinkingChunk receives thinking chunks fired during callModel",
  async () => {
    let thinkingText = "";
    let thinkingChunkCount = 0;

    const fakeCallModel = async () => {
      const emit = getStreamThinkingChunk();
      await emit("The answer is ");
      await emit("42 because ");
      await emit("math says so.");
      return [
        ownThoughtTurn("The answer is 42 because math says so."),
        ownUtteranceTurn("42"),
      ];
    };

    await injectCallModel(fakeCallModel)(async () => {
      await agentDeps([
        participantUtteranceTurn({ name: "user", text: "what is 6*7?" }),
      ])(runAgent)({
        maxIterations: 1,
        tools: [],
        prompt: "unused in fake",
        onStreamThinkingChunk: (chunk) => {
          thinkingText += chunk;
          thinkingChunkCount++;
        },
        rewriteHistory: noopRewriteHistory,
        timezoneIANA: "UTC",
      });
    })();

    assert(
      thinkingChunkCount === 3,
      `expected 3 thinking chunks, got ${thinkingChunkCount}`,
    );
    assert(
      thinkingText === "The answer is 42 because math says so.",
      `expected assembled thinking text, got: ${thinkingText}`,
    );
  },
);

Deno.test(
  "geminiThinkingConfig uses ThinkingLevel constants and never thinkingBudget numbers",
  () => {
    const highConfig = geminiThinkingConfig(ThinkingLevel.HIGH);
    assert(highConfig.includeThoughts === true);
    assertEquals(highConfig.thinkingLevel, ThinkingLevel.HIGH);
    assert(!("thinkingBudget" in highConfig));

    const lowConfig = geminiThinkingConfig(ThinkingLevel.LOW);
    assert(lowConfig.includeThoughts === true);
    assertEquals(lowConfig.thinkingLevel, ThinkingLevel.LOW);
    assert(!("thinkingBudget" in lowConfig));

    const disabledConfig = geminiThinkingConfig(ThinkingLevel.LOW, false);
    assertEquals(disabledConfig.includeThoughts, false);
    assertEquals(disabledConfig.thinkingLevel, ThinkingLevel.LOW);
    assert(!("thinkingBudget" in disabledConfig));
  },
);

Deno.test(
  "geminiGenJsonFromConvo sets includeThoughts false when disableThinking is true",
  async () => {
    let capturedReq: GenerateContentParameters | undefined;
    const fakeGenerateContent = (req: GenerateContentParameters) => {
      capturedReq = req;
      return Promise.resolve({
        candidates: [{ finishReason: "STOP" }],
        text: '{"ok": true}',
      } as unknown as GenerateContentResponse);
    };

    const result = await pipe(
      injectGeminiGenerateContent(fakeGenerateContent),
    )(() =>
      geminiGenJsonFromConvo(
        { tier: "flash", disableThinking: true },
        [{ role: "user", content: "hello" }],
        z.object({ ok: z.boolean() }),
      )
    )();

    assertEquals(result, { ok: true });
    assertEquals(capturedReq?.config?.thinkingConfig?.includeThoughts, false);
    assertEquals(
      capturedReq?.config?.thinkingConfig?.thinkingLevel,
      ThinkingLevel.LOW,
    );
  },
);

Deno.test(
  "geminiGenJsonFromConvo sets includeThoughts false when tier is lite",
  async () => {
    let capturedReq: GenerateContentParameters | undefined;
    const fakeGenerateContent = (req: GenerateContentParameters) => {
      capturedReq = req;
      return Promise.resolve({
        candidates: [{ finishReason: "STOP" }],
        text: '{"ok": true}',
      } as unknown as GenerateContentResponse);
    };

    const result = await pipe(
      injectGeminiGenerateContent(fakeGenerateContent),
    )(() =>
      geminiGenJsonFromConvo(
        { tier: "lite" },
        [{ role: "user", content: "hello" }],
        z.object({ ok: z.boolean() }),
      )
    )();

    assertEquals(result, { ok: true });
    assertEquals(capturedReq?.config?.thinkingConfig?.includeThoughts, false);
    assertEquals(
      capturedReq?.config?.thinkingConfig?.thinkingLevel,
      ThinkingLevel.LOW,
    );
  },
);
