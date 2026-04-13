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
