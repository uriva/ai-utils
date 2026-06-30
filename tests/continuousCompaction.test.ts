import {
  assertEquals,
  assertNotEquals,
  assertStringIncludes,
} from "@std/assert";
import {
  getSpillThreshold,
  type HistoryEvent,
  runToolResultCompaction,
} from "../mod.ts";

Deno.test("continuousCompaction - getSpillThreshold behaves as expected over time", () => {
  const now = Date.now();

  // At 0 minutes, threshold should be exactly maxThreshold (15,000)
  const threshold0 = getSpillThreshold(now);
  assertEquals(threshold0, 15000);

  // At 15 minutes, threshold should be close to 5,000
  const threshold15 = getSpillThreshold(now - 15 * 60 * 1000);
  assertEquals(threshold15 >= 4500 && threshold15 <= 5500, true);

  // At 60 minutes, threshold should be very close to minThreshold (1,500)
  const threshold60 = getSpillThreshold(now - 60 * 60 * 1000);
  assertEquals(threshold60 >= 1500 && threshold60 <= 1650, true);

  // At 24 hours, threshold should be exactly minThreshold (1,500)
  const threshold24h = getSpillThreshold(now - 24 * 60 * 60 * 1000);
  assertEquals(threshold24h, 1500);
});

Deno.test("continuousCompaction - runToolResultCompaction retroactively compacts old large tool results", async () => {
  const now = Date.now();
  const scratchStore = new Map<string, string>();
  const replacements: Record<string, HistoryEvent> = {};

  const setScratch = (id: string, content: string): Promise<void> => {
    scratchStore.set(id, content);
    return Promise.resolve();
  };

  const rewriteHistory = (
    reps: Record<string, HistoryEvent>,
  ): Promise<void> => {
    Object.assign(replacements, reps);
    return Promise.resolve();
  };

  const mockGenerateTLDR = (
    _toolCall: HistoryEvent,
    resultText: string,
  ): Promise<string> => {
    if (resultText.includes("revert command")) {
      return Promise.resolve("Reverted buggy code changes successfully.");
    }
    return Promise.resolve("Successfully executed command.");
  };

  // Construct history
  const recentToolCallId = "recent-call-id";
  const oldToolCallId = "old-call-id";

  const history: HistoryEvent[] = [
    // 1. A recent tool run (10 seconds ago)
    {
      id: recentToolCallId,
      type: "tool_call",
      name: "run_command",
      parameters: { command: "deno cache main.ts" },
      timestamp: now - 10 * 1000,
      isOwn: true,
    },
    {
      id: "recent-result-id",
      type: "tool_result",
      toolCallId: recentToolCallId,
      result: "Success: cached main.ts\nExit code: 0\n" + "a".repeat(8000), // 8000 chars, under 15k threshold
      timestamp: now - 8 * 1000,
      isOwn: true,
    },
    // 2. An old tool run (20 minutes ago)
    {
      id: oldToolCallId,
      type: "tool_call",
      name: "run_command",
      parameters: { command: "git revert acee253" },
      timestamp: now - 20 * 60 * 1000,
      isOwn: true,
    },
    {
      id: "old-result-id",
      type: "tool_result",
      toolCallId: oldToolCallId,
      result: "Success: revert command executed cleanly\n" + "b".repeat(8000), // 8000 chars, over ~4000 threshold at 20 min
      timestamp: now - 19 * 60 * 1000,
      isOwn: true,
    },
  ];

  await runToolResultCompaction(
    history,
    { setScratch, generateTLDR: mockGenerateTLDR },
    rewriteHistory,
  );

  // Assertions
  // Recent result should NOT have been modified
  assertEquals(replacements["recent-result-id"], undefined);

  // Old result SHOULD have been modified
  assertNotEquals(replacements["old-result-id"], undefined);
  const updatedOldResult = replacements["old-result-id"] as Extract<
    HistoryEvent,
    { type: "tool_result" }
  >;

  assertStringIncludes(
    updatedOldResult.result,
    "[Because time has passed, this tool result has been compacted to save space.",
  );
  assertStringIncludes(
    updatedOldResult.result,
    "Memory TLDR: Reverted buggy code changes successfully.",
  );
  assertStringIncludes(
    updatedOldResult.result,
    'read_scratch_file` with the ID: "old-result-id"',
  );

  // Verify full content was saved to scratchpad store
  assertEquals(
    scratchStore.get("old-result-id"),
    (history[3] as Extract<HistoryEvent, { type: "tool_result" }>).result,
  );
});
