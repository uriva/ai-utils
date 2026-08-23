import {
  type HistoryEvent,
  participantUtteranceTurn,
  toolResultTurn,
  toolUseTurn,
} from "../mod.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
  someTool,
} from "../test_helpers.ts";

// Duplicate delivery of a background tool result can place two tool_results
// with the same toolCallId into persisted history. Anthropic hard-rejects such
// requests ("each tool_use must have a single result"), killing the whole
// agent turn. The harness must keep the provider request valid.
const seedHistory = (): HistoryEvent[] => {
  const call = toolUseTurn({ name: someTool.name, args: {} });
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "Please call the doSomethingUnique tool.",
    }),
    call,
    toolResultTurn({ result: "43212e8e", toolCallId: call.id }),
    toolResultTurn({ result: "43212e8e", toolCallId: call.id }),
    participantUtteranceTurn({
      name: "user",
      text: "Thanks! Acknowledge briefly in one short sentence.",
    }),
  ];
  history.forEach((event, index) => {
    event.timestamp = index + 1;
  });
  return history;
};

runForAllProviders(
  "duplicate tool_result delivery keeps the provider request valid",
  async (runAgentWithProvider) => {
    const history = seedHistory();
    await agentDeps(history)(runAgentWithProvider)({
      maxIterations: 3,
      tools: [someTool],
      prompt: "You are a helpful assistant. Be brief.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const reply = history.find((event) => event.type === "own_utterance");
    if (!reply) throw new Error("expected the agent to acknowledge");
  },
);
