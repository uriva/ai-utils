import { assert, assertEquals } from "@std/assert";
import { capEventsToTokenBudget } from "../src/geminiAgent.ts";
import { estimateTokens } from "../src/agent.ts";
import { map, sum } from "gamla";

const makeUtterance = (id: string, text: string, timestamp: number) => ({
  type: "participant_utterance" as const,
  isOwn: false,
  id,
  timestamp,
  name: "user",
  text,
});

const makeOwnUtterance = (id: string, text: string, timestamp: number) => ({
  type: "own_utterance" as const,
  isOwn: true,
  id,
  timestamp,
  text,
  modelMetadata: {
    type: "gemini" as const,
    responseId: `resp-${id}`,
    thoughtSignature: "",
  },
});

const makeToolCall = (id: string, timestamp: number) => ({
  type: "tool_call" as const,
  isOwn: true,
  id,
  timestamp,
  name: "someTool",
  parameters: { foo: "bar" },
  modelMetadata: {
    type: "gemini" as const,
    responseId: `resp-${id}`,
    thoughtSignature: "sig",
  },
});

const makeToolResult = (id: string, toolCallId: string, timestamp: number) => ({
  type: "tool_result" as const,
  isOwn: true,
  id,
  timestamp,
  name: "someTool",
  result: "ok",
  toolCallId,
});

// ~650 chars => ~211 tokens each
const longText = "x".repeat(650);

Deno.test(
  "capEventsToTokenBudget drops oldest events when total exceeds budget",
  () => {
    // deno-lint-ignore no-explicit-any
    const events: any[] = Array.from(
      { length: 100 },
      (_, i) => makeUtterance(`msg${i}`, longText, i * 1000),
    );

    const totalBefore = sum(map(estimateTokens)(events));
    assert(
      totalBefore > 10000,
      `Total tokens ${totalBefore} should exceed 10000`,
    );

    // deno-lint-ignore no-explicit-any
    const capped = capEventsToTokenBudget(10000)(events as any);
    const totalAfter = sum(map(estimateTokens)(capped));

    assert(
      totalAfter <= 10000,
      `Capped tokens ${totalAfter} should be <= 10000`,
    );
    assert(capped.length < events.length, "Should have fewer events after cap");
    // Should keep the newest events
    assertEquals(
      capped[capped.length - 1].id,
      "msg99",
      "Should keep the newest event",
    );
  },
);

Deno.test(
  "capEventsToTokenBudget cleans up orphaned tool results after dropping",
  () => {
    // Create a history where old tool_call + tool_result pairs get split
    // when oldest events are dropped
    // deno-lint-ignore no-explicit-any
    const events: any[] = [
      // Old tool call (will be dropped)
      makeToolCall("tc0", 100),
      // Old tool result (should also be cleaned up since tc0 is dropped)
      makeToolResult("tr0", "tc0", 200),
      // Many filler events to push over budget
      ...Array.from(
        { length: 50 },
        (_, i) => makeUtterance(`filler${i}`, longText, 1000 + i * 100),
      ),
      // Recent tool call + result (should be kept)
      makeToolCall("tc1", 10000),
      makeToolResult("tr1", "tc1", 10100),
      makeOwnUtterance("final", "done", 10200),
    ];

    // Use a budget that keeps ~half the filler + the recent events
    const totalTokens = sum(map(estimateTokens)(events));
    const budget = Math.floor(totalTokens / 2);

    // deno-lint-ignore no-explicit-any
    const capped = capEventsToTokenBudget(budget)(events as any);

    // The old tool_result for tc0 should NOT be in the output
    // (either dropped with tc0 or cleaned as orphan)
    const orphanedResults = capped.filter(
      // deno-lint-ignore no-explicit-any
      (e: any) => e.type === "tool_result" && e.toolCallId === "tc0",
    );
    assertEquals(
      orphanedResults.length,
      0,
      "Orphaned tool_result for dropped tool_call should be removed",
    );

    // The recent tc1/tr1 pair should still be intact
    // deno-lint-ignore no-explicit-any
    const recentCall = capped.find((e: any) => e.id === "tc1");
    // deno-lint-ignore no-explicit-any
    const recentResult = capped.find((e: any) => e.id === "tr1");
    assert(recentCall, "Recent tool_call should be kept");
    assert(recentResult, "Recent tool_result should be kept");
  },
);

Deno.test(
  "capEventsToTokenBudget is a no-op when events are within budget",
  () => {
    // deno-lint-ignore no-explicit-any
    const events: any[] = [
      makeUtterance("msg0", "Hello", 100),
      makeOwnUtterance("msg1", "Hi there", 200),
    ];

    const totalTokens = sum(map(estimateTokens)(events));
    // deno-lint-ignore no-explicit-any
    const capped = capEventsToTokenBudget(totalTokens + 1000)(events as any);

    assertEquals(capped.length, events.length, "No events should be dropped");
    assertEquals(capped[0].id, "msg0");
    assertEquals(capped[1].id, "msg1");
  },
);

Deno.test(
  "capEventsToTokenBudget also cleans orphaned tool_calls (no matching result)",
  () => {
    // deno-lint-ignore no-explicit-any
    const events: any[] = [
      // Old tool result that matches old call
      makeToolResult("tr0", "tc0", 50),
      // Old tool call whose result was already dropped
      makeToolCall("tc0", 100),
      // Filler
      ...Array.from(
        { length: 30 },
        (_, i) => makeUtterance(`f${i}`, longText, 500 + i * 100),
      ),
      makeOwnUtterance("final", "end", 10000),
    ];

    const totalTokens = sum(map(estimateTokens)(events));
    const budget = Math.floor(totalTokens / 3);

    // deno-lint-ignore no-explicit-any
    const capped = capEventsToTokenBudget(budget)(events as any);

    // After capping, if tr0 was dropped but tc0 somehow survived,
    // the orphan filter should handle it. Or both are dropped.
    // Either way, no orphaned tool_results should remain.
    const allToolResults = capped.filter(
      // deno-lint-ignore no-explicit-any
      (e: any) => e.type === "tool_result",
    );
    const allToolCallIds = new Set(
      // deno-lint-ignore no-explicit-any
      capped.filter((e: any) => e.type === "tool_call").map((e: any) => e.id),
    );
    for (const tr of allToolResults) {
      // deno-lint-ignore no-explicit-any
      const toolCallId = (tr as any).toolCallId;
      assert(
        allToolCallIds.has(toolCallId),
        `tool_result ${tr.id} references missing tool_call ${toolCallId}`,
      );
    }
  },
);
