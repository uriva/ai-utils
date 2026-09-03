import { assert } from "@std/assert";
import { z } from "zod/v4";
import {
  type HistoryEvent,
  participantUtteranceTurn,
  tool,
  type ToolUse,
} from "../src/agent.ts";
import {
  agentDeps,
  noopRewriteHistory,
  runForAllProviders,
} from "../test_helpers.ts";

const fetchPriceTool = tool({
  name: "fetch_price",
  description: "Fetch the current price of a specified grocery item.",
  parameters: z.object({
    item: z.string().describe("The item to check (e.g. apples, bananas)"),
  }),
  handler: ({ item }) => Promise.resolve(`Price for ${item}: $2.50`),
});

runForAllProviders(
  "agent emits parallel tool calls in a single turn when instructed",
  async (runAgentWithProvider) => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What are the prices of apples and bananas?",
      }),
    ];

    await agentDeps(mockHistory)(runAgentWithProvider)({
      maxIterations: 5,
      tools: [fetchPriceTool],
      prompt: [
        "You are a grocery shopping assistant.",
        "CRITICAL EFFICIENCY REQUIREMENT: When a user asks about multiple items, you MUST fetch information for all requested items concurrently in the same turn by calling fetch_price for each item in parallel in your very first response.",
        "Never look up items one by one in sequential turns.",
      ].join(" "),
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const toolCalls = mockHistory.filter(
      (e): e is ToolUse<{ item: string }> => e.type === "tool_call",
    );

    assert(
      toolCalls.length >= 2,
      `Expected at least 2 tool calls, got: ${toolCalls.length}`,
    );

    const items = toolCalls.map((tc) =>
      String(tc.parameters?.item ?? "").toLowerCase()
    );
    assert(
      items.some((it) => it.includes("apple")),
      `Expected apple lookup in ${JSON.stringify(items)}`,
    );
    assert(
      items.some((it) => it.includes("banana")),
      `Expected banana lookup in ${JSON.stringify(items)}`,
    );

    // Verify that the assistant ultimately replied to the user
    const replies = mockHistory.filter((e) => e.type === "own_utterance");
    assert(
      replies.some((r) =>
        r.text?.includes("2.50") ||
        (r.text?.toLowerCase().includes("apple") &&
          r.text?.toLowerCase().includes("banana"))
      ),
      `Expected reply mentioning items/prices, got: ${JSON.stringify(replies)}`,
    );
  },
);
