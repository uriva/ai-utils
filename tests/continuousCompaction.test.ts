import {
  assertEquals,
  assertNotEquals,
  assertStringIncludes,
} from "@std/assert";
import { compactToolResultsInMemory, getSpillThreshold } from "../mod.ts";
import type { HistoryEvent } from "../src/agent.ts";

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

Deno.test("continuousCompaction - compactToolResultsInMemory compacts old large tool results in memory", async () => {
  const now = Date.now();
  const scratchStore = new Map<string, string>();

  const setScratch = (id: string, content: string): Promise<void> => {
    scratchStore.set(id, content);
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

  const compacted = await compactToolResultsInMemory(
    history,
    { setScratch, generateTLDR: mockGenerateTLDR },
  );

  // Assertions
  // Recent result should NOT have been modified
  const updatedRecentResult = compacted.find((e: HistoryEvent) =>
    e.id === "recent-result-id"
  ) as Extract<
    HistoryEvent,
    { type: "tool_result" }
  >;
  assertEquals(
    updatedRecentResult.result.startsWith("Success: cached main.ts"),
    true,
  );

  // Old result SHOULD have been modified
  const updatedOldResult = compacted.find((e: HistoryEvent) =>
    e.id === "old-result-id"
  ) as Extract<
    HistoryEvent,
    { type: "tool_result" }
  >;
  assertNotEquals(updatedOldResult, undefined);

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

  // Scratchpad must have stored the full original output
  const spilledContent = scratchStore.get("old-result-id");
  assertNotEquals(spilledContent, undefined);
  assertEquals(
    spilledContent?.startsWith("Success: revert command executed cleanly\n"),
    true,
  );
  assertEquals(spilledContent?.length, 8041);

  // Recent tool result must NOT have been spilled to scratchpad
  assertEquals(scratchStore.get("recent-result-id"), undefined);
});

Deno.test("continuousCompaction - deterministic rich TLDR generation without custom model", async () => {
  const now = Date.now();
  const scratchStore = new Map<string, string>();

  const setScratch = (id: string, content: string): Promise<void> => {
    scratchStore.set(id, content);
    return Promise.resolve();
  };

  const oldToolCallId = "old-inspect-id";
  const history: HistoryEvent[] = [
    {
      id: oldToolCallId,
      type: "tool_call",
      name: "inspect_module",
      parameters: { module: "Database", depth: 2 },
      timestamp: now - 30 * 60 * 1000,
      isOwn: true,
    },
    {
      id: "old-inspect-res",
      type: "tool_result",
      toolCallId: oldToolCallId,
      result:
        "Found 24 schema definitions and 3 migrations.\nDetails: all migrations valid.\n" +
        "x".repeat(8000),
      timestamp: now - 29 * 60 * 1000,
      isOwn: true,
    },
  ];

  // Run compaction WITHOUT passing generateTLDR
  const compacted = await compactToolResultsInMemory(
    history,
    { setScratch },
  );

  const updatedResult = compacted.find((e: HistoryEvent) =>
    e.id === "old-inspect-res"
  ) as Extract<
    HistoryEvent,
    { type: "tool_result" }
  >;
  assertNotEquals(updatedResult, undefined);

  // Must include rich deterministic info (tool name, parameters, first line of output)
  assertStringIncludes(
    updatedResult.result,
    'Memory TLDR: Command "inspect_module"',
  );
  assertStringIncludes(
    updatedResult.result,
    "module: Database",
  );
  assertStringIncludes(
    updatedResult.result,
    "Found 24 schema definitions and 3 migrations.",
  );
  // Must NOT fall back to dummy uninformative "Command completed."
  assertEquals(
    updatedResult.result.includes("Memory TLDR: Command completed."),
    false,
  );
});

Deno.test("continuousCompaction - deterministic rich TLDR unwraps run_command to the inner command", async () => {
  const now = Date.now();
  const scratchStore = new Map<string, string>();
  const setScratch = (id: string, content: string): Promise<void> => {
    scratchStore.set(id, content);
    return Promise.resolve();
  };

  const oldToolCallId = "old-run-cmd-id";
  const history: HistoryEvent[] = [
    {
      id: oldToolCallId,
      type: "tool_call",
      name: "run_command",
      parameters: {
        command: "search/query",
        params: { q: "foo" },
        spinnerText: "Searching...",
      },
      timestamp: now - 30 * 60 * 1000,
      isOwn: true,
    },
    {
      id: "old-run-cmd-res",
      type: "tool_result",
      toolCallId: oldToolCallId,
      result: "Found 10 search results.\nDetails: result details here.\n" +
        "x".repeat(8000),
      timestamp: now - 29 * 60 * 1000,
      isOwn: true,
    },
  ];

  const compacted = await compactToolResultsInMemory(history, { setScratch });
  const updatedResult = compacted.find((e: HistoryEvent) =>
    e.id === "old-run-cmd-res"
  ) as Extract<HistoryEvent, { type: "tool_result" }>;
  assertNotEquals(updatedResult, undefined);

  assertStringIncludes(
    updatedResult.result,
    'Memory TLDR: Command "search/query"',
  );
  assertStringIncludes(
    updatedResult.result,
    "q: foo",
  );
});
