import { assert } from "@std/assert";
import { z } from "zod/v4";
import {
  type HistoryEvent,
  participantUtteranceTurn,
  tool,
} from "../../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../../test_helpers.ts";
import {
  appendEvalRecord,
  currentProvider,
  summarizeHistory,
} from "./eval_logger.ts";

const fetchPriceTool = tool({
  name: "fetch_price",
  description: "Fetch the current price of a specified grocery item.",
  parameters: z.object({
    item: z.string().describe("The item to check (e.g. apples, bananas)"),
  }),
  handler: ({ item }) => Promise.resolve(`Price for ${item}: $2.50`),
});

runForAllProviders(
  "eval tool reliability scores parallel fetch",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What are the prices of apples and bananas?",
      }),
    ];
    const start = Date.now();
    let pass = false;
    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      tools: [fetchPriceTool],
      prompt: [
        "You are a grocery shopping assistant.",
        "When a user asks about multiple items, fetch all of them concurrently in the same turn.",
      ].join(" "),
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });
    const toolCalls = mockHistory.filter((e) => e.type === "tool_call");
    const replies = mockHistory.filter((e) => e.type === "own_utterance");
    pass = toolCalls.length >= 2 &&
      replies.some((r) =>
        r.type === "own_utterance" && r.text.includes("2.50")
      );
    assert(pass, `Expected 2 tool calls plus priced reply`);
    const summary = summarizeHistory(mockHistory);
    await appendEvalRecord({
      suite: "tool",
      task: "parallel_fetch_price",
      provider: currentProvider(),
      pass,
      turns: summary.turns,
      toolCalls: summary.toolCalls,
      historyTokens: summary.historyTokens,
      timeMs: Date.now() - start,
    });
  },
  1,
);
