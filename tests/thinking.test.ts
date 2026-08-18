import { assert } from "@std/assert";
import { ThinkingLevel } from "@google/genai";
import { runAgent } from "../mod.ts";
import {
  getStreamThinkingChunk,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { geminiThinkingConfig } from "../src/gemini.ts";
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
      lightModel: true,
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
  "geminiThinkingConfig bounds thinkingBudget to prevent thinking from starving output space",
  () => {
    const fullConfig = geminiThinkingConfig(false, 16000);
    assert(fullConfig.includeThoughts === true);
    assert(fullConfig.thinkingBudget === 8000);
    assert(!("thinkingLevel" in fullConfig));

    const mediumConfig = geminiThinkingConfig(false, 8192);
    assert(mediumConfig.thinkingBudget === 4096);

    const miniConfig = geminiThinkingConfig(true, 16000);
    assert(
      miniConfig.thinkingLevel === ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
    );
    assert(!("thinkingBudget" in miniConfig));

    const defaultMini = geminiThinkingConfig(true);
    assert(
      defaultMini.thinkingLevel === ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
    );
    assert(!("thinkingBudget" in defaultMini));

    const defaultFull = geminiThinkingConfig(false);
    assert(defaultFull.thinkingBudget === 8192);
  },
);
