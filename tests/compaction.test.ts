import { assertEquals } from "@std/assert";
import { map, sum } from "gamla";
import { estimateTokens, type HistoryEvent } from "../src/agent.ts";
import {
  eventToPlainText,
  groupToolCallPairs,
  partitionSegments,
  segmentHistoryEvents,
} from "../src/compaction.ts";

const makeOwnUtterance = (
  text: string,
  timestamp: number,
): HistoryEvent => ({
  id: crypto.randomUUID(),
  type: "own_utterance",
  text,
  timestamp,
  isOwn: true,
});

const makeParticipantUtterance = (
  text: string,
  timestamp: number,
): HistoryEvent => ({
  id: crypto.randomUUID(),
  type: "participant_utterance",
  text,
  timestamp,
  isOwn: false,
  name: "user",
});

const makeUtterance = (text: string, timestamp: number, isOwn: boolean) =>
  isOwn
    ? makeOwnUtterance(text, timestamp)
    : makeParticipantUtterance(text, timestamp);

const makeToolCall = (
  id: string,
  name: string,
  timestamp: number,
): HistoryEvent => ({
  id,
  type: "tool_call",
  isOwn: true,
  name,
  timestamp,
  parameters: { arg: "val" },
});

const makeToolResult = (
  toolCallId: string,
  timestamp: number,
): HistoryEvent => ({
  id: crypto.randomUUID(),
  type: "tool_result",
  isOwn: true,
  toolCallId,
  result: "done",
  timestamp,
});

const segmentGapMs = 30 * 60 * 1000;
const longText = "x".repeat(2000);

Deno.test("groupToolCallPairs keeps tool_call with following tool_results", () => {
  const events: HistoryEvent[] = [
    makeParticipantUtterance("hi", 100),
    makeToolCall("c1", "toolA", 200),
    makeToolResult("c1", 201),
    makeParticipantUtterance("ok", 300),
  ];
  const groups = groupToolCallPairs(events);
  assertEquals(groups.length, 3);
  assertEquals(groups[0].length, 1);
  assertEquals(groups[1].length, 2);
  assertEquals(groups[1][0].type, "tool_call");
  assertEquals(groups[1][1].type, "tool_result");
  assertEquals(groups[2].length, 1);
});

Deno.test("groupToolCallPairs handles multiple consecutive tool_calls each with results", () => {
  const events: HistoryEvent[] = [
    makeToolCall("c1", "toolA", 100),
    makeToolResult("c1", 101),
    makeToolCall("c2", "toolB", 200),
    makeToolResult("c2", 201),
    makeToolResult("c2", 202),
  ];
  const groups = groupToolCallPairs(events);
  assertEquals(groups.length, 2);
  assertEquals(groups[0].length, 2);
  assertEquals(groups[1].length, 3);
});

Deno.test("segmentation never splits tool_call from its tool_result", () => {
  const base = Date.now();
  const events: HistoryEvent[] = [
    makeParticipantUtterance("hello", base),
    makeOwnUtterance("hi there", base + 1000),
    makeToolCall("c1", "myTool", base + segmentGapMs - 100),
    makeToolResult("c1", base + segmentGapMs + 100),
    makeParticipantUtterance("thanks", base + segmentGapMs + 200),
    makeOwnUtterance("welcome", base + segmentGapMs + 300),
  ];
  const segments = segmentHistoryEvents(events, segmentGapMs);
  for (const seg of segments) {
    const toolCallIds = new Set(
      seg.events
        .filter((e): e is Extract<HistoryEvent, { type: "tool_call" }> =>
          e.type === "tool_call"
        )
        .map((e) => e.id),
    );
    const toolResultCallIds = seg.events
      .filter((e): e is Extract<HistoryEvent, { type: "tool_result" }> =>
        e.type === "tool_result"
      )
      .map((e) => e.toolCallId)
      .filter((id): id is string => id !== undefined);
    for (const resultCallId of toolResultCallIds) {
      assertEquals(
        toolCallIds.has(resultCallId),
        true,
        `tool_result referencing ${resultCallId} is in a segment without its tool_call`,
      );
    }
  }
});

Deno.test("segmentation: tool_call/tool_result pair spanning gap stays together", () => {
  const base = Date.now();
  const events: HistoryEvent[] = [
    makeParticipantUtterance("start", base),
    makeOwnUtterance("thinking", base + 1000),
    makeToolCall("c1", "longRunning", base + segmentGapMs - 50),
    makeToolResult("c1", base + segmentGapMs + 50),
  ];
  const segments = segmentHistoryEvents(events, segmentGapMs);
  const allEvents = segments.flatMap((s) => s.events);
  assertEquals(allEvents.length, events.length, "No events should be lost");
  const segWithCall = segments.find((s) =>
    s.events.some((e) => e.type === "tool_call")
  );
  const segWithResult = segments.find((s) =>
    s.events.some((e) => e.type === "tool_result")
  );
  assertEquals(
    segWithCall,
    segWithResult,
    "tool_call and tool_result should be in same segment",
  );
});

Deno.test("partitionSegments never splits tool_call from tool_result across kept/summarized", () => {
  const base = Date.now();
  const oldEvents: HistoryEvent[] = [
    makeParticipantUtterance("old msg", base),
    makeToolCall("c1", "oldTool", base + 1000),
    makeToolResult("c1", base + 2000),
    makeOwnUtterance("old response", base + 3000),
  ];
  const recentStart = base + segmentGapMs + 1;
  const recentEvents: HistoryEvent[] = Array.from(
    { length: 40 },
    (_, i) =>
      makeUtterance(
        `Recent ${i}: ${longText}`,
        recentStart + i * 60_000,
        i % 2 === 0,
      ),
  );
  const allEvents = [...oldEvents, ...recentEvents];
  const segments = segmentHistoryEvents(allEvents, segmentGapMs);
  const { kept, toSummarize } = partitionSegments(30000, segments);

  const keptIds = new Set(kept.flatMap((s) => s.events.map((e) => e.id)));
  const summarizedIds = new Set(
    toSummarize.flatMap((s) => s.events.map((e) => e.id)),
  );

  const allToolCalls = allEvents.filter((e) => e.type === "tool_call");
  for (const tc of allToolCalls) {
    const matchingResults = allEvents.filter(
      (e) =>
        e.type === "tool_result" &&
        (e as Extract<HistoryEvent, { type: "tool_result" }>).toolCallId ===
          tc.id,
    );
    for (const tr of matchingResults) {
      const callInKept = keptIds.has(tc.id);
      const resultInKept = keptIds.has(tr.id);
      const callInSummarized = summarizedIds.has(tc.id);
      const resultInSummarized = summarizedIds.has(tr.id);
      assertEquals(
        callInKept === resultInKept,
        true,
        `tool_call ${tc.id} kept=${callInKept} but result ${tr.id} kept=${resultInKept}`,
      );
      assertEquals(
        callInSummarized === resultInSummarized,
        true,
        `tool_call ${tc.id} summarized=${callInSummarized} but result ${tr.id} summarized=${resultInSummarized}`,
      );
    }
  }
});

Deno.test("single large segment is kept, not summarized", () => {
  const baseTime = Date.now();
  const events: HistoryEvent[] = Array.from(
    { length: 40 },
    (_, i) =>
      makeUtterance(
        `Message ${i}: ${longText}`,
        baseTime + i * 60_000,
        i % 2 === 0,
      ),
  );
  const segments = segmentHistoryEvents(events, segmentGapMs);
  assertEquals(segments.length, 1, "Should be a single segment");
  const { kept, toSummarize } = partitionSegments(30000, segments);
  assertEquals(kept.length, 1, "Single segment must be kept");
  assertEquals(toSummarize.length, 0, "Nothing to summarize");
});

Deno.test("oversized newest segment gets trimmed, old segment kept if within budget", () => {
  const baseTime = Date.now();
  const events: HistoryEvent[] = Array.from(
    { length: 5 },
    (_, i) =>
      makeUtterance(`Old message ${i}`, baseTime + i * 60_000, i % 2 === 0),
  );
  const newSegmentStart = baseTime + 5 * 60_000 + segmentGapMs + 1;
  events.push(
    ...Array.from(
      { length: 80 },
      (_, i) =>
        makeUtterance(
          `Recent message ${i}: ${longText}`,
          newSegmentStart + i * 60_000,
          i % 2 === 0,
        ),
    ),
  );
  const segments = segmentHistoryEvents(events, segmentGapMs);
  assertEquals(segments.length, 2, "Should be two segments");
  const { kept, toSummarize } = partitionSegments(30000, segments);
  assertEquals(
    toSummarize.length,
    1,
    "Overflow from newest segment should be summarized",
  );
  const summarizedEvents = toSummarize.flatMap((s) => s.events);
  assertEquals(
    summarizedEvents.every((e) => e.timestamp >= newSegmentStart),
    true,
    "Summarized events should be overflow from newest segment",
  );
  assertEquals(
    kept.length,
    2,
    "Trimmed newest + old segment should both be kept",
  );
  const keptTokens = sum(map(estimateTokens)(kept.flatMap((s) => s.events)));
  assertEquals(
    keptTokens <= 30000,
    true,
    `Kept tokens (${keptTokens}) should be within budget`,
  );
});

Deno.test("single segment exceeding token budget gets older events summarized", () => {
  const baseTime = Date.now();
  const veryLongText = "x".repeat(4000);
  const events: HistoryEvent[] = Array.from(
    { length: 40 },
    (_, i) => makeUtterance(veryLongText, baseTime + i * 60_000, i % 2 === 0),
  );
  const totalTokens = sum(events.map(estimateTokens));
  assertEquals(
    totalTokens > 30000,
    true,
    `Total tokens (${totalTokens}) should exceed budget`,
  );
  const segments = segmentHistoryEvents(events, segmentGapMs);
  assertEquals(segments.length, 1, "Should be a single segment");
  const { kept, toSummarize } = partitionSegments(30000, segments);
  assertEquals(
    toSummarize.length > 0,
    true,
    "Older events should be summarized",
  );
  const keptEvents = kept.flatMap((s) => s.events);
  const summarizedEvents = toSummarize.flatMap((s) => s.events);
  assertEquals(
    keptEvents.length + summarizedEvents.length,
    events.length,
    "All events accounted for",
  );
  const oldestKeptTimestamp = Math.min(...keptEvents.map((e) => e.timestamp));
  const newestSummarizedTimestamp = Math.max(
    ...summarizedEvents.map((e) => e.timestamp),
  );
  assertEquals(
    newestSummarizedTimestamp < oldestKeptTimestamp,
    true,
    "Summarized events should be older than kept events",
  );
});

Deno.test("eventToPlainText formats tool_call correctly", () => {
  const tc = makeToolCall("c1", "myTool", 100);
  const text = eventToPlainText(tc);
  assertEquals(text.startsWith("TOOL CALL myTool"), true);
});

Deno.test("eventToPlainText formats utterance as plain text", () => {
  const u = makeOwnUtterance("hello world", 100);
  assertEquals(eventToPlainText(u), "hello world");
});

Deno.test("segmentHistoryEvents returns empty for empty input", () => {
  assertEquals(segmentHistoryEvents([], segmentGapMs), []);
});

Deno.test("groupToolCallPairs matches by toolCallId, not adjacency", () => {
  // This is the actual pattern from runAbstractAgent: all tool_calls emitted
  // first, then all tool_results after (because outputEvent emits all model
  // events before handleFunctionCalls processes them).
  const events: HistoryEvent[] = [
    makeToolCall("c1", "toolA", 100),
    makeToolCall("c2", "toolB", 101),
    makeToolResult("c1", 102),
    makeToolResult("c2", 103),
  ];
  const groups = groupToolCallPairs(events);
  // c1 and its result should be in the same group
  const c1Group = groups.find((g) =>
    g.some((e) => e.type === "tool_call" && e.id === "c1")
  )!;
  assertEquals(
    c1Group.some((e) =>
      e.type === "tool_result" &&
      (e as Extract<HistoryEvent, { type: "tool_result" }>).toolCallId === "c1"
    ),
    true,
    "tool_call c1 should be grouped with tool_result referencing c1",
  );
  // c2 and its result should be in the same group
  const c2Group = groups.find((g) =>
    g.some((e) => e.type === "tool_call" && e.id === "c2")
  )!;
  assertEquals(
    c2Group.some((e) =>
      e.type === "tool_result" &&
      (e as Extract<HistoryEvent, { type: "tool_result" }>).toolCallId === "c2"
    ),
    true,
    "tool_call c2 should be grouped with tool_result referencing c2",
  );
});

Deno.test("segmentation never splits non-adjacent tool_call from its tool_result", () => {
  const base = Date.now();
  // Pattern from runAbstractAgent with multi-call turn:
  // msg, tool_call_c1, <big gap>, tool_call_c2, tool_result_c1, tool_result_c2
  // The gap after tool_call_c1 should NOT orphan it from tool_result_c1
  const events: HistoryEvent[] = [
    makeParticipantUtterance("hello", base),
    makeToolCall("c1", "toolA", base + 1000),
    makeToolCall("c2", "toolB", base + 1000 + segmentGapMs + 1),
    makeToolResult("c1", base + 1000 + segmentGapMs + 2),
    makeToolResult("c2", base + 1000 + segmentGapMs + 3),
  ];
  const segments = segmentHistoryEvents(events, segmentGapMs);
  for (const seg of segments) {
    const callIds = new Set(
      seg.events
        .filter((e): e is Extract<HistoryEvent, { type: "tool_call" }> =>
          e.type === "tool_call"
        )
        .map((e) => e.id),
    );
    for (const e of seg.events) {
      if (e.type === "tool_result") {
        const tr = e as Extract<HistoryEvent, { type: "tool_result" }>;
        if (tr.toolCallId) {
          assertEquals(
            callIds.has(tr.toolCallId),
            true,
            `tool_result for ${tr.toolCallId} is in segment without its tool_call`,
          );
        }
      }
    }
  }
});

Deno.test("trimming within single segment respects tool_call/tool_result atomicity", () => {
  const base = Date.now();
  const events: HistoryEvent[] = [];
  for (let i = 0; i < 30; i++) {
    events.push(
      makeUtterance("x".repeat(3000), base + i * 60_000, i % 2 === 0),
    );
  }
  const callTs = base + 30 * 60_000;
  events.push(makeToolCall("c1", "bigTool", callTs));
  events.push(makeToolResult("c1", callTs + 1));
  for (let i = 0; i < 10; i++) {
    events.push(
      makeUtterance("x".repeat(3000), callTs + 2 + i * 60_000, i % 2 === 0),
    );
  }

  const segments = segmentHistoryEvents(events, segmentGapMs);
  const { kept, toSummarize } = partitionSegments(30000, segments);

  const keptEvents = kept.flatMap((s) => s.events);
  const summarizedEvents = toSummarize.flatMap((s) => s.events);

  const callInKept = keptEvents.some(
    (e) => e.type === "tool_call" && e.id === "c1",
  );
  const resultInKept = keptEvents.some(
    (e) =>
      e.type === "tool_result" &&
      (e as Extract<HistoryEvent, { type: "tool_result" }>).toolCallId === "c1",
  );
  const callInSummarized = summarizedEvents.some(
    (e) => e.type === "tool_call" && e.id === "c1",
  );
  const resultInSummarized = summarizedEvents.some(
    (e) =>
      e.type === "tool_result" &&
      (e as Extract<HistoryEvent, { type: "tool_result" }>).toolCallId === "c1",
  );

  assertEquals(
    callInKept === resultInKept,
    true,
    `tool_call in kept=${callInKept}, tool_result in kept=${resultInKept} — they must match`,
  );
  assertEquals(
    callInSummarized === resultInSummarized,
    true,
    `tool_call in summarized=${callInSummarized}, tool_result in summarized=${resultInSummarized} — they must match`,
  );
});
