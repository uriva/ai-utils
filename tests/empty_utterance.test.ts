import { assert } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
  someTool,
} from "../test_helpers.ts";

runForAllProviders(
  "agent does not emit own_utterance with empty text",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          `Think carefully about what you need to do, then call the doSomethingUnique tool. Make sure to reason through your approach first.`,
      }),
    ];
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt:
        "You are an AI assistant. Always think through your approach before taking action. Use [Internal thought, visible only to you: ...] to think.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const emptyUtterances = mockHistory.filter(
      (e) => e.type === "own_utterance" && !e.text.trim(),
    );
    assert(
      emptyUtterances.length === 0,
      `Expected no empty own_utterance events, but found ${emptyUtterances.length}. History types: ${
        mockHistory.map((e) =>
          e.type === "own_utterance"
            ? `own_utterance("${e.text.slice(0, 40)}")`
            : e.type
        ).join(", ")
      }`,
    );
  },
);
