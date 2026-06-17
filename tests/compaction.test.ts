import { assertEquals } from "@std/assert";
import { map, sum } from "gamla";
import { z } from "zod/v4";
import { estimateTokensLocal, type HistoryEvent } from "../src/agent.ts";
import {
  eventsToPlainText,
  eventToPlainText,
  groupToolCallPairs,
  partitionSegments,
  segmentHistoryEvents,
  summarizeEvents,
} from "../src/compaction.ts";
import { genJson } from "../src/genJson.ts";
import { injectSecrets, llmTest } from "../test_helpers.ts";

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

Deno.test("partitionSegments never splits tool_call from tool_result across kept/summarized", async () => {
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
  const { kept, toSummarize } = await partitionSegments(30000, segments);

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

Deno.test("single large segment is kept, not summarized", async () => {
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
  const { kept, toSummarize } = await partitionSegments(30000, segments);
  assertEquals(kept.length, 1, "Single segment must be kept");
  assertEquals(toSummarize.length, 0, "Nothing to summarize");
});

Deno.test("oversized newest segment gets trimmed, old segment kept if within budget", async () => {
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
  const { kept, toSummarize } = await partitionSegments(30000, segments);
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
  const keptTokens = sum(
    map(estimateTokensLocal)(kept.flatMap((s) => s.events)),
  );
  assertEquals(
    keptTokens <= 30000,
    true,
    `Kept tokens (${keptTokens}) should be within budget`,
  );
});

Deno.test("single segment exceeding token budget gets older events summarized", async () => {
  const baseTime = Date.now();
  const veryLongText = "x".repeat(4000);
  const events: HistoryEvent[] = Array.from(
    { length: 40 },
    (_, i) => makeUtterance(veryLongText, baseTime + i * 60_000, i % 2 === 0),
  );
  const totalTokens = sum(events.map(estimateTokensLocal));
  assertEquals(
    totalTokens > 30000,
    true,
    `Total tokens (${totalTokens}) should exceed budget`,
  );
  const segments = segmentHistoryEvents(events, segmentGapMs);
  assertEquals(segments.length, 1, "Should be a single segment");
  const { kept, toSummarize } = await partitionSegments(30000, segments);
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

llmTest(
  "summarizeEvents excludes implicitly rejected proposals from pending items",
  injectSecrets(async () => {
    const base = Date.now();
    const events: HistoryEvent[] = [
      makeParticipantUtterance("I need a hotel for my trip", base),
      makeOwnUtterance(
        "I recommend the Grand Plaza Hotel. It has excellent reviews and a central location.",
        base + 1000,
      ),
      makeParticipantUtterance(
        "What else is available?",
        base + 2000,
      ),
      makeOwnUtterance(
        "Another option is the Seaside Resort with ocean views, or the Artisan Inn which is a boutique hotel in the old town.",
        base + 3000,
      ),
      makeParticipantUtterance(
        "The Artisan Inn sounds perfect. Let's book that.",
        base + 4000,
      ),
      makeOwnUtterance(
        "Excellent choice! The Artisan Inn is booked for you.",
        base + 5000,
      ),
    ];

    const summary = await summarizeEvents(events);
    console.log("Summary:\n", summary);

    const pendingMatch = summary.match(/## Pending Items\n([\s\S]*?)(?=## |$)/);
    const pendingSection = pendingMatch ? pendingMatch[1] : "";

    assertEquals(
      pendingSection.toLowerCase().includes("grand plaza"),
      false,
      `Rejected proposal 'Grand Plaza Hotel' should not appear in Pending Items.\nFull summary:\n${summary}`,
    );
  }),
);

llmTest(
  "summarizeEvents excludes abandoned proposals when user complains and moves on",
  injectSecrets(async () => {
    const base = Date.now();
    const events: HistoryEvent[] = [
      makeParticipantUtterance("I need a hotel for my trip", base),
      makeOwnUtterance(
        "I recommend the Grand Plaza Hotel. It has excellent reviews and a central location.",
        base + 1000,
      ),
      makeParticipantUtterance(
        "Can you tell me about the neighborhood?",
        base + 2000,
      ),
      makeOwnUtterance(
        "The Grand Plaza is near the central station. From there you can take the blue line to the museum district in about 10 minutes. The walk from the hotel to the station is roughly 5 minutes.",
        base + 3000,
      ),
      makeParticipantUtterance(
        "Please stop repeating the same transit details, I already understood that",
        base + 4000,
      ),
      makeOwnUtterance(
        "Sorry about that! Let me focus on what you actually need.",
        base + 5000,
      ),
      makeParticipantUtterance(
        "What about boutique options?",
        base + 6000,
      ),
      makeOwnUtterance(
        "The Artisan Inn is a lovely boutique hotel in the old town with unique rooms.",
        base + 7000,
      ),
      makeParticipantUtterance(
        "The Artisan Inn sounds perfect. Let's book that.",
        base + 8000,
      ),
      makeOwnUtterance(
        "Excellent choice! The Artisan Inn is confirmed for your dates.",
        base + 9000,
      ),
    ];

    const summary = await summarizeEvents(events);
    console.log("Summary:\n", summary);

    const pendingMatch = summary.match(/## Pending Items\n([\s\S]*?)(?=## |$)/);
    const pendingSection = pendingMatch ? pendingMatch[1] : "";

    assertEquals(
      pendingSection.toLowerCase().includes("grand plaza"),
      false,
      `Abandoned proposal 'Grand Plaza Hotel' should not appear in Pending Items after user complained and chose an alternative.\nFull summary:\n${summary}`,
    );
  }),
);

llmTest(
  "summarizeEvents marks implicitly rejected items as abandoned when user moves on without explicit no",
  injectSecrets(async () => {
    const base = Date.now();
    const events: HistoryEvent[] = [
      makeParticipantUtterance(
        "I need hotels for two parts of my trip. First, somewhere for the beginning.",
        base,
      ),
      makeOwnUtterance(
        "For the first part, I recommend the Grand Plaza Hotel. It has excellent reviews and a central location.",
        base + 1000,
      ),
      makeParticipantUtterance(
        "Okay noted. Now what about the second part of the trip?",
        base + 2000,
      ),
      makeOwnUtterance(
        "For the second part, the Seaside Resort has beautiful ocean views.",
        base + 3000,
      ),
      makeParticipantUtterance(
        "The Seaside Resort looks great, let's book it.",
        base + 4000,
      ),
      makeOwnUtterance(
        "Excellent! The Seaside Resort is booked for the second part.",
        base + 5000,
      ),
    ];

    const summary = await summarizeEvents(events);
    console.log("Summary:\n", summary);

    const pendingMatch = summary.match(/## Pending Items\n([\s\S]*?)(?=## |$)/);
    const pendingSection = pendingMatch ? pendingMatch[1] : "";

    assertEquals(
      pendingSection.toLowerCase().includes("grand plaza"),
      false,
      `Implicitly rejected proposal 'Grand Plaza Hotel' should not appear in Pending Items when user moved on to alternatives without explicitly saying no.\nFull summary:\n${summary}`,
    );
  }),
);

llmTest(
  "summarizeEvents does not list a specific rejected hotel as pending when user complains and switches segments",
  injectSecrets(async () => {
    const base = Date.now();
    const events: HistoryEvent[] = [
      makeParticipantUtterance(
        "I need a hotel for the first part of my trip.",
        base,
      ),
      makeOwnUtterance(
        "I recommend the Grand Plaza Hotel. It has excellent reviews and a central location near the main station.",
        base + 1000,
      ),
      makeParticipantUtterance(
        "Tell me about getting around from there.",
        base + 2000,
      ),
      makeOwnUtterance(
        "From the Grand Plaza you can take the blue line to the museum district in about 10 minutes. The walk to the station is roughly 5 minutes through the shopping arcade.",
        base + 3000,
      ),
      makeParticipantUtterance(
        "Stop repeating the same transit info please",
        base + 4000,
      ),
      makeOwnUtterance(
        "Sorry about that! I'll stop repeating those details.",
        base + 5000,
      ),
      makeParticipantUtterance(
        "Let's talk about the second part of the trip. What do you recommend?",
        base + 6000,
      ),
      makeOwnUtterance(
        "For the second part, the Seaside Resort has beautiful ocean views and a private beach.",
        base + 7000,
      ),
      makeParticipantUtterance(
        "Perfect, book the Seaside Resort for the second part.",
        base + 8000,
      ),
      makeOwnUtterance(
        "Done! The Seaside Resort is confirmed for the second part of your trip.",
        base + 9000,
      ),
    ];

    const summary = await summarizeEvents(events);
    console.log("Summary:\n", summary);

    const pendingMatch = summary.match(/## Pending Items\n([\s\S]*?)(?=## |$)/);
    const pendingSection = pendingMatch ? pendingMatch[1] : "";

    assertEquals(
      pendingSection.toLowerCase().includes("grand plaza"),
      false,
      `Specific rejected hotel 'Grand Plaza Hotel' should not appear in Pending Items after user complained and switched to a different trip segment.\nFull summary:\n${summary}`,
    );
  }),
);

llmTest(
  "summarizeEvents marks proposal as abandoned when user moves on without confirming or rejecting it",
  injectSecrets(async () => {
    const base = Date.now();
    const events: HistoryEvent[] = [
      makeParticipantUtterance(
        "I need a hotel for the first part of my trip.",
        base,
      ),
      makeOwnUtterance(
        "I recommend the Grand Plaza Hotel. It has excellent reviews.",
        base + 1000,
      ),
      makeParticipantUtterance(
        "Okay I'll think about it. Now what about the second part?",
        base + 2000,
      ),
      makeOwnUtterance(
        "For the second part, the Seaside Resort has beautiful ocean views.",
        base + 3000,
      ),
      makeParticipantUtterance(
        "The Seaside Resort looks great, let's book it.",
        base + 4000,
      ),
      makeOwnUtterance(
        "Excellent! The Seaside Resort is booked for the second part.",
        base + 5000,
      ),
    ];

    const summary = await summarizeEvents(events);
    console.log("Summary:\n", summary);

    const pendingMatch = summary.match(/## Pending Items\n([\s\S]*?)(?=## |$)/);
    const pendingSection = pendingMatch ? pendingMatch[1] : "";

    // When a proposal was made but the user moved on to a different topic
    // without confirming or rejecting it, the specific proposal should be
    // treated as abandoned, not kept as pending.
    assertEquals(
      pendingSection.toLowerCase().includes("grand plaza"),
      false,
      `Proposal 'Grand Plaza Hotel' should be treated as abandoned when user moved on without confirming or rejecting it.\nFull summary:\n${summary}`,
    );
  }),
);

Deno.test("trimming within single segment respects tool_call/tool_result atomicity", async () => {
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
  const { kept, toSummarize } = await partitionSegments(30000, segments);

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

// Repro target: the compactor sometimes takes history that only refers to an
// entity generically (e.g. "the second hotel near X") and crystallizes it into
// a fabricated proper-noun ("X-something Hotel") in the summary. Downstream
// turns then treat that fabricated name as established fact.
//
// First failure point per AGENTS.md: a single call to `summarizeEvents`
// produces a summary containing specific proper-noun entities (hotel names,
// person names, document names) that are NOT present anywhere in the source
// events.

const buildFabricationProbeEvents = (
  neighborhood: string,
): HistoryEvent[] => {
  const base = Date.now();
  const t = (n: number) => base + n * 1000;
  return [
    makeParticipantUtterance(
      `I'm planning a Tokyo trip with my friend. We need two hotels: one downtown for the first part, and a second one near ${neighborhood} for later. I'm a professional photographer and want stylish places, not student vibes.`,
      t(0),
    ),
    makeOwnUtterance(
      `Got it — boutique/design-leaning rather than backpacker, both downtown and near ${neighborhood}. I'll keep your photography work in mind too (good light, walkable to interesting streetscapes).`,
      t(1),
    ),
    makeParticipantUtterance(
      `My friend already booked her own hotel. She sent me the details in a message.`,
      t(2),
    ),
    makeOwnUtterance(
      `OK — so we only need to coordinate yours, and we can pick somewhere near her base if you want to meet up easily. Where is she staying?`,
      t(3),
    ),
    makeParticipantUtterance(
      `She's near the convention center for work. I'll forward the message later.`,
      t(4),
    ),
    makeOwnUtterance(
      `Sounds good. For now let's focus on your two stays: downtown first, then near ${neighborhood}.`,
      t(5),
    ),
    makeParticipantUtterance(
      `How many nights does it make sense for each?`,
      t(6),
    ),
    makeOwnUtterance(
      `Plan was 3 nights at the first hotel (downtown), then 4 nights at the second hotel (${neighborhood} area). We can adjust.`,
      t(7),
    ),
    makeParticipantUtterance(
      `What about a day trip out of Tokyo somewhere coastal?`,
      t(8),
    ),
    makeOwnUtterance(
      `There are several great coastal day trips reachable by train in 1-2 hours — temples, beaches, old castle towns. We can pick one once the hotels are locked.`,
      t(9),
    ),
    makeParticipantUtterance(`I might also want a mountain shrine day.`, t(10)),
    makeOwnUtterance(
      `Doable — there's a famous mountain shrine area 2 hours north with cedar forests, very photogenic. Good for a sunrise visit.`,
      t(11),
    ),
    makeParticipantUtterance(
      `OK back to hotels. The first one — quiet downtown is fine, but I want good morning light for window shots.`,
      t(12),
    ),
    makeOwnUtterance(
      `Noted — for the downtown stay we'll prioritize east-facing higher floors. For the second hotel near ${neighborhood} we can pick something with a rooftop or a quiet side street view.`,
      t(13),
    ),
    makeParticipantUtterance(`כמה זה עולה בערך ללילה?`, t(14)),
    makeOwnUtterance(
      `בוטיק/עיצובי באזורים האלה זה בערך 25,000-45,000 ין ללילה לחדר זוגי. את המלון הראשון (downtown) אפשר לפעמים למצוא קצת יותר זול, את השני (${neighborhood}) דווקא יותר מבוקש.`,
      t(15),
    ),
    makeParticipantUtterance(
      `And what's the closest station to the second hotel area?`,
      t(16),
    ),
    makeOwnUtterance(
      `The ${neighborhood} area is served by the JR Yamanote line plus a couple of subway lines, so transit from most museum districts is 15-25 minutes.`,
      t(17),
    ),
    makeParticipantUtterance(
      `My friend texted me the name of her hotel but it's in Russian, I'll paste it later.`,
      t(18),
    ),
    makeOwnUtterance(
      `No problem, I can translate it. Just paste when you have it.`,
      t(19),
    ),
    makeParticipantUtterance(
      `Also I want to do at least one big art museum.`,
      t(20),
    ),
    makeOwnUtterance(
      `Tokyo has several major museums — modern art, photography, classical. I'd recommend at least the photography museum given your work, and one modern art museum.`,
      t(21),
    ),
    makeParticipantUtterance(`Photography museum yes definitely.`, t(22)),
    makeOwnUtterance(
      `Locked in. We'll slot it in on a day with afternoon free time near the second hotel.`,
      t(23),
    ),
    makeParticipantUtterance(
      `Let's wrap up the first hotel decision soon, I want to book this week.`,
      t(24),
    ),
    makeOwnUtterance(
      `Understood — I'll come back to you with 2-3 concrete options for the first hotel (downtown) and 2-3 for the second hotel (${neighborhood} area) so you can pick.`,
      t(25),
    ),
    makeParticipantUtterance(`Perfect.`, t(26)),
  ];
};

const fabricationJudgeSchema = z.object({
  fabricatedProperNouns: z.array(z.string()).describe(
    "List of specific proper-noun entities (hotel names, person names, document/text titles, restaurant names, brand-name landmarks) that appear in the SUMMARY but do NOT appear in the SOURCE EVENTS. Do not include generic descriptions like 'the second hotel' or neighborhood names that ARE in the source. Only include genuinely fabricated specific names. If none, return empty array.",
  ),
  reasoning: z.string().describe(
    "One short sentence explaining your decision.",
  ),
});

const judgeFabrications = (source: string, summary: string) =>
  genJson(
    { provider: "google", mini: true },
    `You are checking whether a conversation summary fabricated specific named entities (proper nouns) that were not present in the original conversation events.

You will be given:
1. SOURCE EVENTS: the raw conversation text.
2. SUMMARY: a structured summary produced from those events.

Your job: list every PROPER-NOUN entity that the SUMMARY introduces but that does NOT appear in the SOURCE EVENTS. Specifically watch for: hotel names, person names, document/text titles, specific restaurant or shop names, named tours or services. IGNORE generic phrases ("the second hotel", "the downtown hotel"), neighborhood/city names that ARE in the source, and common nouns. Only flag fabricated SPECIFIC proper-noun entities.`,
    fabricationJudgeSchema,
  )(`SOURCE EVENTS:\n${source}\n\nSUMMARY:\n${summary}`);

const probeFabrication = async (neighborhood: string) => {
  const events = buildFabricationProbeEvents(neighborhood);
  const source = eventsToPlainText(events);
  const summary = await summarizeEvents(events);
  const verdict = await judgeFabrications(source, summary);
  return { neighborhood, summary, verdict };
};

const fabricationProbeNeighborhoods = [
  "Ebisu",
  "Shibuya",
  "Asakusa",
  "Ginza",
  "Ueno",
];

llmTest(
  "summarizeEvents does not fabricate proper-noun entities absent from source events",
  injectSecrets(async () => {
    const probes = await Promise.all(
      fabricationProbeNeighborhoods.map(probeFabrication),
    );
    probes.forEach((p) =>
      console.log(
        `[${p.neighborhood}] fabricated=${
          JSON.stringify(p.verdict.fabricatedProperNouns)
        } reason=${p.verdict.reasoning}`,
      )
    );
    const totalFabricated = sum(
      probes.map((p) => p.verdict.fabricatedProperNouns.length),
    );
    const offenders = probes.filter((p) =>
      p.verdict.fabricatedProperNouns.length > 0
    );
    assertEquals(
      totalFabricated,
      0,
      `Compactor fabricated proper-noun entities in ${offenders.length}/${probes.length} probes:\n${
        offenders.map((o) =>
          `  - ${o.neighborhood}: ${
            JSON.stringify(o.verdict.fabricatedProperNouns)
          }\n      reason: ${o.verdict.reasoning}\n      summary excerpt: ${
            o.summary.slice(0, 600).replace(/\n/g, " ")
          }`
        ).join("\n")
      }`,
    );
  }),
  1,
);

llmTest(
  "summarizeEvents identifies active skills that must be re-learned after compaction",
  injectSecrets(async () => {
    const base = Date.now();
    const events: HistoryEvent[] = [
      makeParticipantUtterance("I want to integrate a service", base),
      {
        id: "call-1",
        type: "tool_call",
        isOwn: true,
        name: "learn_skill",
        parameters: { skillName: "p2b-coder" },
        timestamp: base + 1000,
      },
      {
        id: crypto.randomUUID(),
        type: "tool_result",
        isOwn: true,
        toolCallId: "call-1",
        result: JSON.stringify({
          name: "p2b-coder",
          description: "Coder skill with strict guidelines",
          instructions: "STRICT RULE: Do not ask for screenshots.",
          tools: [],
        }),
        timestamp: base + 2000,
      },
      makeOwnUtterance("I loaded the p2b-coder skill guidelines.", base + 3000),
    ];

    const summary = await summarizeEvents(events);
    console.log("Summary:\n", summary);

    const lowercaseSummary = summary.toLowerCase();
    assertEquals(
      lowercaseSummary.includes("p2b-coder"),
      true,
      `Summary should identify 'p2b-coder' as a skill to re-learn.\nFull summary:\n${summary}`,
    );
    assertEquals(
      summary.includes("Active Skills to Re-Learn"),
      true,
      `Summary should contain the active skills instruction block.\nFull summary:\n${summary}`,
    );
  }),
);
