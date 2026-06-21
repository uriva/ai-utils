import { assertEquals } from "@std/assert";
import { cleanActiveMemoryToolRaw } from "../src/compaction.ts";
import { cleanActiveMemoryToolName } from "../src/utils.ts";
import { callToResult, type HistoryEvent } from "../src/agent.ts";

Deno.test("clean_active_memory tool - deletes target events successfully", async () => {
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

  let capturedReplacements: Record<string, HistoryEvent> = {};
  const mockRewriteHistory = (replacements: Record<string, HistoryEvent>) => {
    capturedReplacements = replacements;
    return Promise.resolve();
  };
  const mockGetHistory = () => Promise.resolve(history);

  const cleanTool = {
    ...cleanActiveMemoryToolRaw(mockRewriteHistory, mockGetHistory),
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

  // Verify that both call-2 and res-3 were mapped to do_nothing
  assertEquals(capturedReplacements["call-2"]?.type, "do_nothing");
  assertEquals(capturedReplacements["res-3"]?.type, "do_nothing");
});

Deno.test("clean_active_memory tool - summarizes target events successfully", async () => {
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

  let capturedReplacements: Record<string, HistoryEvent> = {};
  const mockRewriteHistory = (replacements: Record<string, HistoryEvent>) => {
    capturedReplacements = replacements;
    return Promise.resolve();
  };
  const mockGetHistory = () => Promise.resolve(history);

  const cleanTool = {
    ...cleanActiveMemoryToolRaw(mockRewriteHistory, mockGetHistory),
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
  // Check the first event is replaced with the summary thought
  assertEquals(capturedReplacements["call-2"]?.type, "own_thought");
  const c2 = capturedReplacements["call-2"];
  assertEquals(
    c2 && "text" in c2 ? c2.text : undefined,
    "[SYSTEM SUMMARY of events from 1970-01-01T00:00:02.000Z to 1970-01-01T00:00:03.000Z]: I completed task A.",
  );
  // Rest should be do_nothing
  assertEquals(capturedReplacements["res-3"]?.type, "do_nothing");
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

  const mockRewriteHistory = () => Promise.resolve();
  const mockGetHistory = () => Promise.resolve(history);

  const cleanTool = {
    ...cleanActiveMemoryToolRaw(mockRewriteHistory, mockGetHistory),
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
