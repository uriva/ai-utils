import { assert } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  getStreamThinkingChunk,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

runForAllProviders(
  "agent returns own_thought events when thinking is enabled",
  async (runAgent) => {
    const mockHistory = [
      participantUtteranceTurn({
        name: "user",
        text: "What is 137 * 248? Think step by step.",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
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
        onMaxIterationsReached: () => {},
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
