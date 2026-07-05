import { assertEquals } from "@std/assert";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";
import {
  type HistoryEvent,
  participantUtteranceTurn,
  stopThoughtPrefix,
} from "../src/agent.ts";

runForAllProviders(
  "runAgent - stop thought in history causes silent failure on next user prompt",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "do something that gets stuck",
      }),
      {
        type: "own_thought",
        isOwn: true,
        id: "stop-thought-id",
        timestamp: Date.now() - 2000,
        text:
          `${stopThoughtPrefix} I should instead stop and ask the user for help.`,
      },
      participantUtteranceTurn({
        name: "user",
        text: "Are you done?", // new user message
      }),
    ];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 30,
      tools: [],
      prompt:
        "You are a specialized movie scene finder bot. If the user asks if you are done or ready, and you are still active, reply saying you are active and ready to help. But if you have already stopped, output exactly [no response] and nothing else.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    // Verification of fixed behavior:
    // With the fix applied, the previous stop thought is filtered out of the model's history view,
    // so the real Gemini model responds normally with a helpful own_utterance instead of staying silent.
    const lastEvent = mockHistory[mockHistory.length - 1];
    assertEquals(
      lastEvent.type,
      "own_utterance",
      "Expected Gemini to respond with a real utterance to the user prompt.",
    );
  },
  3,
  true, // geminiOnly = true
);
