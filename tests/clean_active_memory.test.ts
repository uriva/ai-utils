import { assertEquals, assertStringIncludes } from "@std/assert";
import {
  applyCleanActiveMemoryDirectives,
  cleanActiveMemoryToolRaw,
} from "../src/compaction.ts";
import { cleanActiveMemoryToolName } from "../src/utils.ts";
import {
  callToResult,
  type HistoryEvent,
  projectHistoryToModelContext,
  skillLoadedResultText,
} from "../src/agent.ts";

Deno.test("clean_active_memory tool - deletes target events in projected context without mutating raw history", async () => {
  const history: HistoryEvent[] = [
    {
      id: "msg-1",
      type: "participant_utterance",
      isOwn: false,
      text: "user message 1",
      timestamp: 1000,
      name: "user",
    },
    {
      id: "call-2",
      type: "tool_call",
      name: "toolA",
      parameters: {},
      timestamp: 2000,
      isOwn: true,
    },
    {
      id: "res-3",
      type: "tool_result",
      result: "resultA",
      timestamp: 3000,
      isOwn: true,
      toolCallId: "call-2",
    },
    {
      id: "msg-4",
      type: "own_utterance",
      isOwn: true,
      text: "bot reply 1",
      timestamp: 4000,
    },
  ];

  const rawHistoryCopy = JSON.parse(JSON.stringify(history));
  const mockGetHistory = () => Promise.resolve(history);

  const cleanTool = {
    ...cleanActiveMemoryToolRaw(mockGetHistory),
  };

  const resolver = callToResult([cleanTool]);
  const res = await resolver({
    name: cleanActiveMemoryToolName,
    args: {
      start_time: "1970-01-01T00:00:02.000Z", // timestamp 2000
      end_time: "1970-01-01T00:00:03.000Z", // timestamp 3000
    },
    id: "call-cleanup",
  });

  assertEquals(res?.toolCallId, "call-cleanup");
  assertEquals(
    res?.result,
    "Successfully deleted 2 events from 1970-01-01T00:00:02.000Z to 1970-01-01T00:00:03.000Z.",
  );

  // Raw history must remain completely untouched!
  assertEquals(history, rawHistoryCopy);

  // Now verify that the projection function applies the deletion dynamically
  const fullHistoryWithClean: HistoryEvent[] = [
    ...history,
    {
      id: "call-cleanup",
      type: "tool_call",
      name: cleanActiveMemoryToolName,
      parameters: {
        start_time: "1970-01-01T00:00:02.000Z",
        end_time: "1970-01-01T00:00:03.000Z",
      },
      timestamp: 5000,
      isOwn: true,
    },
    {
      id: "res-cleanup",
      type: "tool_result",
      result: res?.result ?? "",
      timestamp: 5001,
      isOwn: true,
      toolCallId: "call-cleanup",
    },
  ];

  const projected = applyCleanActiveMemoryDirectives(fullHistoryWithClean);
  assertEquals(projected.some((e) => e.id === "call-2"), false);
  assertEquals(projected.some((e) => e.id === "res-3"), false);
  assertEquals(projected.some((e) => e.id === "msg-1"), true);
  assertEquals(projected.some((e) => e.id === "msg-4"), true);
});

Deno.test("clean_active_memory tool - summarizes target events in projected context without mutating raw history", async () => {
  const history: HistoryEvent[] = [
    {
      id: "msg-1",
      type: "participant_utterance",
      isOwn: false,
      text: "user message 1",
      timestamp: 1000,
      name: "user",
    },
    {
      id: "call-2",
      type: "tool_call",
      name: "toolA",
      parameters: {},
      timestamp: 2000,
      isOwn: true,
    },
    {
      id: "res-3",
      type: "tool_result",
      result: "resultA",
      timestamp: 3000,
      isOwn: true,
      toolCallId: "call-2",
    },
    {
      id: "msg-4",
      type: "own_utterance",
      isOwn: true,
      text: "bot reply 1",
      timestamp: 4000,
    },
  ];

  const rawHistoryCopy = JSON.parse(JSON.stringify(history));
  const mockGetHistory = () => Promise.resolve(history);

  const cleanTool = {
    ...cleanActiveMemoryToolRaw(mockGetHistory),
  };

  const resolver = callToResult([cleanTool]);
  const res = await resolver({
    name: cleanActiveMemoryToolName,
    args: {
      start_time: "1970-01-01T00:00:02.000Z",
      end_time: "1970-01-01T00:00:03.000Z",
      summary: "I completed task A.",
    },
    id: "call-cleanup",
  });

  assertEquals(res?.toolCallId, "call-cleanup");
  assertEquals(
    res?.result,
    'Successfully summarized 2 events from 1970-01-01T00:00:02.000Z to 1970-01-01T00:00:03.000Z with summary: "I completed task A."',
  );

  // Raw history must remain completely untouched!
  assertEquals(history, rawHistoryCopy);

  const fullHistoryWithClean: HistoryEvent[] = [
    ...history,
    {
      id: "call-cleanup",
      type: "tool_call",
      name: cleanActiveMemoryToolName,
      parameters: {
        start_time: "1970-01-01T00:00:02.000Z",
        end_time: "1970-01-01T00:00:03.000Z",
        summary: "I completed task A.",
      },
      timestamp: 5000,
      isOwn: true,
    },
    {
      id: "res-cleanup",
      type: "tool_result",
      result: res?.result ?? "",
      timestamp: 5001,
      isOwn: true,
      toolCallId: "call-cleanup",
    },
  ];

  const projected = applyCleanActiveMemoryDirectives(fullHistoryWithClean);
  assertEquals(projected.some((e) => e.id === "call-2"), false);
  assertEquals(projected.some((e) => e.id === "res-3"), false);
  assertEquals(projected.some((e) => e.id === "msg-1"), true);
  assertEquals(projected.some((e) => e.id === "msg-4"), true);

  const summaryThought = projected.find((e) =>
    e.type === "own_thought" &&
    typeof e.text === "string" &&
    e.text.includes(
      "[SYSTEM SUMMARY of events from 1970-01-01T00:00:02.000Z to 1970-01-01T00:00:03.000Z]: I completed task A.",
    )
  );
  assertEquals(summaryThought !== undefined, true);
});

Deno.test("clean_active_memory tool - blocks deleting user messages", async () => {
  const history: HistoryEvent[] = [
    {
      id: "msg-1",
      type: "participant_utterance",
      isOwn: false,
      text: "user message 1",
      timestamp: 1000,
      name: "user",
    },
    {
      id: "msg-4",
      type: "own_utterance",
      isOwn: true,
      text: "bot reply 1",
      timestamp: 4000,
    },
  ];

  const mockGetHistory = () => Promise.resolve(history);
  const cleanTool = {
    ...cleanActiveMemoryToolRaw(mockGetHistory),
  };

  const resolver = callToResult([cleanTool]);
  const res = await resolver({
    name: cleanActiveMemoryToolName,
    args: {
      start_time: "1970-01-01T00:00:01.000Z",
      end_time: "1970-01-01T00:00:04.000Z",
    },
    id: "call-cleanup",
  });

  assertEquals(res?.toolCallId, "call-cleanup");
  assertEquals(
    res?.result.includes("Memory cleanup aborted"),
    true,
    `expected blocked notice, got: ${res?.result}`,
  );
});

Deno.test(
  "clean_active_memory tool - detects and alerts on deleted learn_skill events during cleanup and removes skill from active skills",
  async () => {
    const mockHistory: HistoryEvent[] = [
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: "learn_skill",
        parameters: { skillName: "p2b-coder" },
        timestamp: 1000,
      },
      {
        id: "result-1",
        type: "tool_result",
        isOwn: true,
        toolCallId: "call-1",
        result: skillLoadedResultText,
        timestamp: 2000,
      },
    ];

    const rawHistoryCopy = JSON.parse(JSON.stringify(mockHistory));
    const mockGetHistory = () => Promise.resolve(mockHistory);
    const cleanTool = {
      ...cleanActiveMemoryToolRaw(mockGetHistory),
    };

    const resolver = callToResult([cleanTool]);
    const res = await resolver({
      name: cleanActiveMemoryToolName,
      args: {
        start_time: "1970-01-01T00:00:01.000Z", // timestamp 1000
        end_time: "1970-01-01T00:00:02.000Z", // timestamp 2000
      },
      id: "call-cleanup",
    });

    assertEquals(res?.toolCallId, "call-cleanup");
    assertEquals(
      res?.result.includes("Successfully deleted 2 events"),
      true,
      "Should report successful deletion",
    );
    assertEquals(
      res?.result.includes("permanently removed the following active skills"),
      true,
      "Should alert that skills are unlearned",
    );
    assertStringIncludes(res?.result ?? "", "p2b-coder");

    // Raw history is NOT mutated
    assertEquals(mockHistory, rawHistoryCopy);

    const fullHistoryWithClean: HistoryEvent[] = [
      ...mockHistory,
      {
        id: "call-cleanup",
        type: "tool_call",
        name: cleanActiveMemoryToolName,
        parameters: {
          start_time: "1970-01-01T00:00:01.000Z",
          end_time: "1970-01-01T00:00:02.000Z",
        },
        timestamp: 3000,
        isOwn: true,
      },
      {
        id: "res-cleanup",
        type: "tool_result",
        result: res?.result ?? "",
        timestamp: 3001,
        isOwn: true,
        toolCallId: "call-cleanup",
      },
    ];

    const projected = applyCleanActiveMemoryDirectives(fullHistoryWithClean);
    assertEquals(projected.some((e) => e.id === "call-1"), false);
    assertEquals(projected.some((e) => e.id === "result-1"), false);
  },
);

Deno.test("projectHistoryToModelContext integrates applyCleanActiveMemoryDirectives end-to-end", async () => {
  const events: HistoryEvent[] = [
    {
      id: "user-1",
      type: "participant_utterance",
      isOwn: false,
      text: "Please analyze the logs.",
      timestamp: 10000,
      name: "user",
    },
    {
      id: "call-trial-1",
      type: "tool_call",
      name: "run_query",
      parameters: { query: "trial 1" },
      timestamp: 11000,
      isOwn: true,
    },
    {
      id: "res-trial-1",
      type: "tool_result",
      result: "log error trial 1",
      timestamp: 12000,
      isOwn: true,
      toolCallId: "call-trial-1",
    },
    {
      id: "call-clean",
      type: "tool_call",
      name: cleanActiveMemoryToolName,
      parameters: {
        start_time: new Date(11000).toISOString(),
        end_time: new Date(12000).toISOString(),
        summary: "Investigated logs: found transient trial 1 error.",
      },
      timestamp: 13000,
      isOwn: true,
    },
    {
      id: "res-clean",
      type: "tool_result",
      result: `Successfully summarized 2 events from ${
        new Date(11000).toISOString()
      } to ${
        new Date(12000).toISOString()
      } with summary: "Investigated logs: found transient trial 1 error."`,
      timestamp: 13001,
      isOwn: true,
      toolCallId: "call-clean",
    },
  ];

  const projected = await projectHistoryToModelContext({
    rawHistory: events,
  });

  // call-trial-1 and res-trial-1 must be omitted from projected context
  assertEquals(projected.some((e) => e.id === "call-trial-1"), false);
  assertEquals(projected.some((e) => e.id === "res-trial-1"), false);

  // The summary must be present in projected context
  const summaryEvent = projected.find((e) =>
    e.type === "own_thought" &&
    typeof e.text === "string" &&
    e.text.includes("Investigated logs: found transient trial 1 error.")
  );
  assertEquals(summaryEvent !== undefined, true);
  // User utterance must be intact
  assertEquals(projected.some((e) => e.id === "user-1"), true);
});
