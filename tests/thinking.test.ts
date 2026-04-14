import { assert } from "@std/assert";
import { participantUtteranceTurn } from "../src/agent.ts";
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

runForAllProviders(
  "onStreamThinkingChunk receives thinking content during streaming",
  async (runAgent) => {
    let thinkingText = "";
    let thinkingChunkCount = 0;

    await agentDeps([
      participantUtteranceTurn({
        name: "user",
        text: "What is 137 * 248? Think step by step.",
      }),
    ])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helpful assistant. Think carefully before answering.",
      lightModel: true,
      onStreamThinkingChunk: (chunk) => {
        thinkingText += chunk;
        thinkingChunkCount++;
      },
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    assert(
      thinkingChunkCount > 0,
      `Expected at least one thinking stream chunk, got ${thinkingChunkCount}`,
    );
    assert(
      thinkingText.length > 10,
      `Expected substantial thinking text, got ${thinkingText.length} chars`,
    );
  },
);
