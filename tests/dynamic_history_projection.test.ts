import { assertEquals, assertNotEquals } from "@std/assert";
import type { Injector } from "@uri/inject";
import {
  injectCacher,
  overrideTime,
  projectHistoryToModelContext,
  runAgent,
  sanitizeWindowBoundary,
  tool,
  type ToolOutputScratchPad,
} from "../mod.ts";
import { z } from "zod/v4";
import { agentDeps, runForAllProviders, someTool } from "../test_helpers.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
} from "../src/agent.ts";
import { genJsonOverride } from "../src/genJson.ts";

const memoryCacher =
  (store: Record<string, unknown>) => (_cacheId: string): Injector =>
    ((f: (...args: unknown[]) => Promise<unknown>) =>
    async (...args: unknown[]) => {
      const key = JSON.stringify(args);
      if (key in store) return store[key];
      const result = await f(...args);
      store[key] = result;
      return result;
    }) as Injector;

Deno.test(
  "dynamic history projection - dynamically summarizes settled past sessions in-memory without mutating raw history",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000;
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "detailed architectural decisions regarding database migration ".repeat(
        60,
      );

    const rawHistory: HistoryEvent[] = [];
    for (let session = 0; session < 2; session++) {
      const sessionTime = baseTime + session * 2 * hourMs;
      for (let turn = 0; turn < 5; turn++) {
        rawHistory.push({
          id: `p-${session}-${turn}`,
          type: "participant_utterance",
          isOwn: false,
          name: "user",
          text: `User request session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000,
        });
        rawHistory.push({
          id: `o-${session}-${turn}`,
          type: "own_utterance",
          isOwn: true,
          text:
            `Assistant reply session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000 + 500,
        });
      }
    }

    const now = Date.now();
    rawHistory.push({
      id: "p-active",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text: "Can you proceed with the next deployment step?",
      timestamp: now,
    });

    const initialHistorySnapshot = JSON.parse(JSON.stringify(rawHistory));

    let receivedHistoryByModel: HistoryEvent[] = [];
    let summaryCallCount = 0;

    const fakeGenJson =
      (_opts: unknown, _sys: string, _zod: unknown) =>
      (_userMsg: string): Promise<Record<string, string>> => {
        summaryCallCount++;
        return Promise.resolve({
          entities: "Database cluster",
          decisions: "Migrate to postgres",
          actions: "Schema updated",
          pendingItems: "Run deployment step",
          abandonedItems: "None",
          context: "Past migration discussion",
        });
      };

    const inMemoryCache: Record<string, unknown> = {};
    const mockCacher = memoryCacher(inMemoryCache);

    const fakeCallModel = (
      history: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      receivedHistoryByModel = history;
      return Promise.resolve([
        ownUtteranceTurn("I am proceeding with deployment step now."),
      ]);
    };

    await injectCacher(mockCacher)(async () => {
      await genJsonOverride.inject(() => fakeGenJson)(async () => {
        await injectCallModel(fakeCallModel)(async () => {
          await agentDeps(rawHistory)(runAgent)({
            maxIterations: 5,
            tools: [someTool],
            prompt: "You are a DevOps assistant.",
            timezoneIANA: "UTC",
          });
        })();
      })();
    })();

    // 1. Model received compacted history containing the summary
    const summaryEvents = receivedHistoryByModel.filter((e) =>
      e.type === "own_thought" &&
      typeof e.text === "string" &&
      e.text.includes("Past conversation history was compacted")
    );
    assertEquals(
      summaryEvents.length > 0,
      true,
      "Expected model context to contain the dynamically projected session summary",
    );

    // 2. Active message is preserved in full fidelity at the end
    const lastUserMsg = receivedHistoryByModel.find((e) => e.id === "p-active");
    assertNotEquals(
      lastUserMsg,
      undefined,
      "Expected active turn user message to be present in model context",
    );

    // 3. Raw history array is immutable: past events were not deleted or modified
    assertEquals(
      rawHistory.find((e) => e.id === "p-0-0")?.type,
      "participant_utterance",
      "Raw event p-0-0 must remain untouched in immutable event log",
    );
    assertEquals(
      rawHistory.length,
      initialHistorySnapshot.length + 1, // only the new assistant response appended
      "Raw history should only have the newly emitted response appended",
    );

    const summaryCallCountAfterFirstRun = summaryCallCount;
    assertEquals(
      summaryCallCountAfterFirstRun,
      2,
      "Expected each of the 2 settled sessions to be summarized once",
    );

    // 4. Intermediate artifact memoization: Running projection again hits cache without recomputing
    const projectedSecondTime = await injectCacher(mockCacher)(async () => {
      return await genJsonOverride.inject(() => fakeGenJson)(async () => {
        return await projectHistoryToModelContext({
          rawHistory,
          timezoneIANA: "UTC",
        });
      })();
    })();

    assertEquals(
      summaryCallCount,
      summaryCallCountAfterFirstRun,
      "Expected summary generation to hit memoized cache without any additional LLM calls on subsequent projection",
    );
    assertEquals(
      projectedSecondTime.some((e) =>
        e.type === "own_thought" &&
        typeof e.text === "string" &&
        e.text.includes("Past conversation history was compacted")
      ),
      true,
    );
  },
);

Deno.test(
  "dynamic history projection - in-flight tool outputs fold in memory into scratchpad without mutating raw history",
  async () => {
    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStore.get(id)),
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
    };

    const longToolResult = "Server log line: error occurred\n".repeat(600);

    const rawHistory: HistoryEvent[] = [
      {
        id: "p1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Please inspect logs",
        timestamp: Date.now() - 5000,
      },
      {
        id: "call-1",
        type: "tool_call",
        name: "run_logs",
        parameters: { command: "tail logs" },
        isOwn: true,
        timestamp: Date.now() - 4000,
      },
      {
        id: "res-1",
        type: "tool_result",
        toolCallId: "call-1",
        result: longToolResult,
        isOwn: true,
        timestamp: Date.now() - 3500,
      },
      {
        id: "call-2",
        type: "tool_call",
        name: "run_logs",
        parameters: { command: "tail logs 2" },
        isOwn: true,
        timestamp: Date.now() - 3000,
      },
      {
        id: "res-2",
        type: "tool_result",
        toolCallId: "call-2",
        result: longToolResult,
        isOwn: true,
        timestamp: Date.now() - 2500,
      },
      {
        id: "call-3",
        type: "tool_call",
        name: "run_logs",
        parameters: { command: "tail logs 3" },
        isOwn: true,
        timestamp: Date.now() - 2000,
      },
      {
        id: "res-3",
        type: "tool_result",
        toolCallId: "call-3",
        result: "Recent quick output: OK",
        isOwn: true,
        timestamp: Date.now() - 1500,
      },
    ];

    const projected = await projectHistoryToModelContext({
      rawHistory,
      scratchPad,
      generateTLDR: () => Promise.resolve("Inspected logs successfully."),
    });

    // Older tool result res-1 (turn distance 2) should be folded in projected context
    const projectedRes1 = projected.find((e) => e.id === "res-1");
    assertEquals(
      projectedRes1 && "result" in projectedRes1 &&
        typeof projectedRes1.result === "string" &&
        projectedRes1.result.includes(
          "Memory TLDR: Inspected logs successfully.",
        ),
      true,
      "Expected older tool result to be folded into TLDR in projected model context",
    );

    // Scratchpad store received full raw output
    assertEquals(
      scratchStore.get("res-1"),
      longToolResult,
      "Expected full output to be saved in scratchpad store",
    );

    // Raw history event res-1 remains untouched in the original raw history
    const rawRes1 = rawHistory.find((e) => e.id === "res-1");
    assertEquals(
      rawRes1 && "result" in rawRes1 && rawRes1.result === longToolResult,
      true,
      "Raw history event must not be mutated in-place",
    );
  },
);

Deno.test(
  "dynamic history projection - sanitizes window boundaries by dropping orphaned tool results",
  () => {
    const rawEvents: HistoryEvent[] = [
      {
        id: "orphaned-res",
        type: "tool_result",
        toolCallId: "lost-call-before-window",
        result: "Previous tool output that lost its call",
        isOwn: true,
        timestamp: 1000,
      },
      {
        id: "p1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Hello",
        timestamp: 2000,
      },
      {
        id: "call-1",
        type: "tool_call",
        name: "some_tool",
        parameters: {},
        isOwn: true,
        timestamp: 3000,
      },
      {
        id: "res-1",
        type: "tool_result",
        toolCallId: "call-1",
        result: "Tool output",
        isOwn: true,
        timestamp: 4000,
      },
    ];

    const sanitized = sanitizeWindowBoundary(rawEvents);
    assertEquals(sanitized.length, 3);
    assertEquals(sanitized[0].id, "p1");
  },
);

Deno.test(
  "dynamic history projection - recovers when tool output causes content policy block during session summarization",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000;
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "detailed architectural decisions regarding database migration ".repeat(
        600,
      );

    const rawHistory: HistoryEvent[] = [
      {
        id: "p-past-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: `Please check the database: ${fillerText}`,
        timestamp: baseTime,
      },
      {
        id: "call-past-1",
        type: "tool_call",
        name: "some_tool",
        parameters: { query: "lookup" },
        isOwn: true,
        timestamp: baseTime + 1000,
      },
      {
        id: "res-past-1",
        type: "tool_result",
        toolCallId: "call-past-1",
        result: "raw external scrape containing prohibited trigger content",
        isOwn: true,
        timestamp: baseTime + 2000,
      },
      {
        id: "o-past-1",
        type: "own_utterance",
        isOwn: true,
        text: `Found the details in the database: ${fillerText}`,
        timestamp: baseTime + 3000,
      },
      {
        id: "p-active",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Can you proceed with the next step?",
        timestamp: baseTime + 2 * hourMs,
      },
    ];

    let receivedHistoryByModel: HistoryEvent[] = [];
    const fakeGenJson =
      (_opts: unknown, _sys: string, _zod: unknown) =>
      (userMsg: string): Promise<Record<string, string>> => {
        if (userMsg.includes("prohibited trigger content")) {
          throw new Error(
            "Gemini request blocked with reason: PROHIBITED_CONTENT",
          );
        }
        return Promise.resolve({
          entities: "Database service",
          decisions: "Proceed to next step",
          actions: "Queried details",
          pendingItems: "Next step",
          abandonedItems: "None",
          context: "Lookup complete",
        });
      };

    const inMemoryCache: Record<string, unknown> = {};
    const mockCacher = memoryCacher(inMemoryCache);
    const fakeCallModel = (
      history: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      receivedHistoryByModel = history;
      return Promise.resolve([
        ownUtteranceTurn("Proceeding now."),
      ]);
    };

    await injectCacher(mockCacher)(async () => {
      await genJsonOverride.inject(() => fakeGenJson)(async () => {
        await injectCallModel(fakeCallModel)(async () => {
          await agentDeps(rawHistory)(runAgent)({
            maxIterations: 5,
            tools: [someTool],
            prompt: "You are a helpful assistant.",
            timezoneIANA: "UTC",
          });
        })();
      })();
    })();

    const summaryEvents = receivedHistoryByModel.filter((e) =>
      e.type === "own_thought" &&
      typeof e.text === "string" &&
      e.text.includes("Past conversation history was compacted")
    );
    assertEquals(
      summaryEvents.length > 0,
      true,
      "Expected model context to contain the dynamically projected session summary after recovering from tool-result content block",
    );
  },
);

Deno.test(
  "dynamic history projection - falls back to safe summary when entire session is blocked by content policy",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000;
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "detailed architectural decisions regarding database migration ".repeat(
        600,
      );

    const rawHistory: HistoryEvent[] = [
      {
        id: "p-past-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: `Extreme prohibited message: ${fillerText}`,
        timestamp: baseTime,
      },
      {
        id: "o-past-1",
        type: "own_utterance",
        isOwn: true,
        text: `Extreme prohibited assistant reply: ${fillerText}`,
        timestamp: baseTime + 1000,
      },
      {
        id: "p-active",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Can you proceed with the next step?",
        timestamp: baseTime + 2 * hourMs,
      },
    ];

    let receivedHistoryByModel: HistoryEvent[] = [];
    const fakeGenJson =
      (_opts: unknown, _sys: string, _zod: unknown) =>
      (_userMsg: string): Promise<Record<string, string>> => {
        throw new Error(
          "400 The request was rejected because it was considered high risk",
        );
      };

    const inMemoryCache: Record<string, unknown> = {};
    const mockCacher = memoryCacher(inMemoryCache);
    const fakeCallModel = (
      history: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      receivedHistoryByModel = history;
      return Promise.resolve([
        ownUtteranceTurn("Proceeding now."),
      ]);
    };

    await injectCacher(mockCacher)(async () => {
      await genJsonOverride.inject(() => fakeGenJson)(async () => {
        await injectCallModel(fakeCallModel)(async () => {
          await agentDeps(rawHistory)(runAgent)({
            maxIterations: 5,
            tools: [someTool],
            prompt: "You are a helpful assistant.",
            timezoneIANA: "UTC",
          });
        })();
      })();
    })();

    const summaryEvents = receivedHistoryByModel.filter((e) =>
      e.type === "own_thought" &&
      typeof e.text === "string" &&
      e.text.includes("Omitted due to content safety policy")
    );
    assertEquals(
      summaryEvents.length > 0,
      true,
      "Expected model context to contain safe fallback summary when entire session is blocked",
    );
  },
);

Deno.test(
  "dynamic history projection - unread fresh tool result and deferred tool result are not compacted before the model sees them",
  async () => {
    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStore.get(id)),
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
      threshold: 20_000,
    };

    const pageWithGoogleDocLink = "Navigation header\n".repeat(1050) +
      "Schedule link: https://docs.google.com/document/d/1pJPgZydMZS2rcfgrCgthYjAo4_qOjlJP8bn1qvK_hhs/edit?usp=sharing\n";

    const now = Date.now();
    const rawHistoryWithFreshReadUrl: HistoryEvent[] = [
      {
        id: "p1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Find the schedule",
        timestamp: now - 3000,
      },
      {
        id: "call-read-url",
        type: "tool_call",
        name: "run_command",
        parameters: {
          command: "web/read_url",
          params: { url: "https://example.com/festival" },
        },
        isOwn: true,
        timestamp: now - 2000,
      },
      {
        id: "res-read-url",
        type: "tool_result",
        toolCallId: "call-read-url",
        result: pageWithGoogleDocLink,
        isOwn: true,
        timestamp: now - 500,
      },
    ];

    const projectedFresh = await projectHistoryToModelContext({
      rawHistory: rawHistoryWithFreshReadUrl,
      scratchPad,
    });
    const freshToolRes = projectedFresh.find((e) => e.id === "res-read-url");
    assertEquals(
      freshToolRes?.type === "tool_result" &&
        !freshToolRes.result.includes("Because time has passed") &&
        freshToolRes.result.includes(
          "https://docs.google.com/document/d/1pJPgZydMZS2rcfgrCgthYjAo4_qOjlJP8bn1qvK_hhs/edit?usp=sharing",
        ),
      true,
      "Expected unread fresh 19k tool result to remain intact in model context so the model does not waste a turn calling read_scratch_file",
    );

    const deferredScheduleText = "Header line\n" +
      "Workshop schedule details\n".repeat(220);
    const rawHistoryWithDeferredResult: HistoryEvent[] = [
      {
        id: "p1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "What is the schedule today?",
        timestamp: now - 90_000,
      },
      {
        id: "call-thread",
        type: "tool_call",
        name: "run_in_new_thread",
        parameters: { prompt: "Find schedule" },
        isOwn: true,
        timestamp: now - 85_000,
      },
      {
        id: "ack-1",
        type: "own_utterance",
        isOwn: true,
        text: "Checking the schedule now...",
        timestamp: now - 84_000,
      },
      {
        id: "p2",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Well?",
        timestamp: now - 50_000,
      },
      {
        id: "ack-2",
        type: "own_utterance",
        isOwn: true,
        text: "Still scanning the festival pages, almost done!",
        timestamp: now - 45_000,
      },
      {
        id: "call-unlearn",
        type: "tool_call",
        name: "unlearn_skill",
        parameters: { skillName: "web" },
        isOwn: true,
        timestamp: now - 44_000,
      },
      {
        id: "res-unlearn",
        type: "tool_result",
        toolCallId: "call-unlearn",
        result: "Unlearned web",
        isOwn: true,
        timestamp: now - 43_900,
      },
      {
        id: "res-thread",
        type: "tool_result",
        toolCallId: "call-thread",
        result: deferredScheduleText,
        isOwn: true,
        timestamp: now - 200,
      },
    ];

    const projectedDeferred = await projectHistoryToModelContext({
      rawHistory: rawHistoryWithDeferredResult,
      scratchPad,
    });
    const deferredToolRes = projectedDeferred.find((e) =>
      e.id === "res-thread"
    );
    assertEquals(
      deferredToolRes?.type === "tool_result" &&
        !deferredToolRes.result.includes("Because time has passed") &&
        deferredToolRes.result.includes("Workshop schedule details"),
      true,
      "Expected unread fresh deferred tool result to remain intact even when its originating tool_call was several turns earlier",
    );
    assertEquals(
      projectedDeferred.at(-1)?.id,
      "res-thread",
      "Expected late-resolving deferred tool_result to stay at the end of projected history after intervening conversation turns",
    );
    assertEquals(
      projectedDeferred.at(-2)?.id,
      "call-thread",
      "Expected originating deferred tool_call to be paired immediately before its late-resolving tool_result at the tail",
    );
  },
);

Deno.test(
  "dynamic history projection - skill lifecycle and utterance events do not inflate turnDistance for tool result compaction",
  async () => {
    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStore.get(id)),
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
      threshold: 20_000,
    };

    const now = Date.now();
    const apiPayload = "Market listing details line\n".repeat(200);
    const rawHistory: HistoryEvent[] = [
      {
        id: "p1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Fetch the active market listings",
        timestamp: now - 120_000,
      },
      {
        id: "call-fetch",
        type: "tool_call",
        name: "fetch_markets",
        parameters: { category: "tech" },
        isOwn: true,
        timestamp: now - 115_000,
      },
      {
        id: "res-fetch",
        type: "tool_result",
        toolCallId: "call-fetch",
        result: apiPayload,
        isOwn: true,
        timestamp: now - 110_000,
      },
      {
        id: "reply-1",
        type: "own_utterance",
        isOwn: true,
        text: "Here is the summary of the markets.",
        timestamp: now - 105_000,
      },
      {
        id: "unlearn-1",
        type: "tool_call",
        name: "unlearn_skill",
        parameters: { skillName: "web" },
        isOwn: true,
        timestamp: now - 104_000,
      },
      {
        id: "unlearn-1-res",
        type: "tool_result",
        toolCallId: "unlearn-1",
        result: "Unlearned web",
        isOwn: true,
        timestamp: now - 103_900,
      },
      {
        id: "unlearn-2",
        type: "tool_call",
        name: "unlearn_skill",
        parameters: { skillName: "browser" },
        isOwn: true,
        timestamp: now - 103_800,
      },
      {
        id: "unlearn-2-res",
        type: "tool_result",
        toolCallId: "unlearn-2",
        result: "Unlearned browser",
        isOwn: true,
        timestamp: now - 103_700,
      },
      {
        id: "unlearn-3",
        type: "tool_call",
        name: "unlearn_skill",
        parameters: { skillName: "automation" },
        isOwn: true,
        timestamp: now - 103_600,
      },
      {
        id: "unlearn-3-res",
        type: "tool_result",
        toolCallId: "unlearn-3",
        result: "Unlearned automation",
        isOwn: true,
        timestamp: now - 103_500,
      },
      {
        id: "p2",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Give me the direct links from what you found",
        timestamp: now - 5_000,
      },
    ];

    const projected = await projectHistoryToModelContext({
      rawHistory,
      scratchPad,
    });
    const fetchRes = projected.find((e) => e.id === "res-fetch");
    assertEquals(
      fetchRes?.type === "tool_result" &&
        !fetchRes.result.includes("Because time has passed") &&
        fetchRes.result.includes("Market listing details line"),
      true,
      "Expected 5.6k tool result from 2 minutes ago to remain uncompacted when only skill lifecycle and utterance events followed it",
    );
  },
);

Deno.test(
  "dynamic history projection - multi-step tool calls within the same active turn do not prematurely compact prior file reads",
  async () => {
    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => Promise.resolve(scratchStore.get(id)),
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
      threshold: 20_000,
    };

    const now = Date.now();
    const primaryManifestContent = [
      "# -*- coding: utf-8 -*-",
      "outbound = [",
      ...Array.from(
        { length: 40 },
        (_, i) => `    "Passenger Outbound ${i + 1}",`,
      ),
      '    "Alice Carter",',
      "]",
      "return_trip = [",
      ...Array.from(
        { length: 40 },
        (_, i) => `    "Passenger Return ${i + 1}",`,
      ),
      '    "Alice Carter",',
      "]",
    ].join("\n");

    const secondaryManifestContent = [
      "# -*- coding: utf-8 -*-",
      "stations = {",
      ...Array.from(
        { length: 50 },
        (_, i) => `    "Passenger ${i + 1}": "Central Station",`,
      ),
      "}",
    ].join("\n");

    const rawHistory: HistoryEvent[] = [
      {
        id: "p-user",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text:
          "Keep Alice Carter on outbound only, and put Diana Ross on return_trip instead of Alice Carter.",
        timestamp: now - 25_000,
      },
      {
        id: "call-read-primary",
        type: "tool_call",
        name: "read_file",
        parameters: { path: "/workspace/manifest_primary.py" },
        isOwn: true,
        timestamp: now - 20_000,
      },
      {
        id: "res-read-primary",
        type: "tool_result",
        toolCallId: "call-read-primary",
        result: primaryManifestContent,
        isOwn: true,
        timestamp: now - 19_500,
      },
      {
        id: "call-read-secondary",
        type: "tool_call",
        name: "read_file",
        parameters: { path: "/workspace/manifest_secondary.py" },
        isOwn: true,
        timestamp: now - 15_000,
      },
      {
        id: "res-read-secondary",
        type: "tool_result",
        toolCallId: "call-read-secondary",
        result: secondaryManifestContent,
        isOwn: true,
        timestamp: now - 14_500,
      },
      {
        id: "call-grep-station",
        type: "tool_call",
        name: "grep_file",
        parameters: { pattern: "Diana Ross", path: "/workspace" },
        isOwn: true,
        timestamp: now - 10_000,
      },
      {
        id: "res-grep-station",
        type: "tool_result",
        toolCallId: "call-grep-station",
        result: "No matches found",
        isOwn: true,
        timestamp: now - 9_500,
      },
      {
        id: "call-check-counts",
        type: "tool_call",
        name: "check_counts",
        parameters: { file: "/workspace/manifest_primary.py" },
        isOwn: true,
        timestamp: now - 5_000,
      },
      {
        id: "res-check-counts",
        type: "tool_result",
        toolCallId: "call-check-counts",
        result: "outbound=41, return_trip=41",
        isOwn: true,
        timestamp: now - 4_500,
      },
    ];

    const projected = await projectHistoryToModelContext({
      rawHistory,
      scratchPad,
    });
    const primaryRes = projected.find((e) => e.id === "res-read-primary");
    assertEquals(
      primaryRes?.type === "tool_result" &&
        !primaryRes.result.includes("Because time has passed") &&
        primaryRes.result.includes('"Alice Carter"'),
      true,
      "Expected active-turn file read (20s old, 2.2k chars) to remain uncompacted across subsequent tool steps in the same turn",
    );

    const historyAfterThreeReplies: HistoryEvent[] = [
      ...rawHistory,
      {
        id: "reply-1",
        type: "own_utterance",
        isOwn: true,
        text: "Updated the primary manifest.",
        timestamp: now - 4_000,
      },
      {
        id: "p-followup-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "Thanks!",
        timestamp: now - 3_500,
      },
      {
        id: "reply-2",
        type: "own_utterance",
        isOwn: true,
        text: "You are welcome!",
        timestamp: now - 3_000,
      },
      {
        id: "p-followup-2",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: "All good.",
        timestamp: now - 2_500,
      },
      {
        id: "reply-3",
        type: "own_utterance",
        isOwn: true,
        text: "Have a great trip!",
        timestamp: now - 2_000,
      },
    ];

    const projectedAfterReplies = await projectHistoryToModelContext({
      rawHistory: historyAfterThreeReplies,
      scratchPad,
    });
    const compactedPrimaryRes = projectedAfterReplies.find(
      (e) => e.id === "res-read-primary",
    );
    assertEquals(
      compactedPrimaryRes?.type === "tool_result" &&
        compactedPrimaryRes.result.includes("Because time has passed"),
      true,
      "Expected file read to be compacted once 3 completed conversational turns have elapsed",
    );
  },
);

runForAllProviders(
  "agent edits already-read file directly without re-reading after 3 intermediate tool calls in the same turn",
  async (runAgentWithProvider) => {
    const primaryFileContent = [
      "# -*- coding: utf-8 -*-",
      "# Primary roster script",
      "outbound_roster = [",
      ...Array.from(
        { length: 35 },
        (_, i) =>
          `    "Outbound Member ${
            String(i + 1).padStart(2, "0")
          } - Standard Boarding",`,
      ),
      '    "Alice Carter - Gate 42B",',
      "]",
      "",
      "return_roster = [",
      ...Array.from(
        { length: 35 },
        (_, i) =>
          `    "Return Member ${
            String(i + 1).padStart(2, "0")
          } - Standard Boarding",`,
      ),
      '    "Martin Vance - Gate 19A",',
      '    "Alice Carter - Gate 42B",',
      '    "Elena Rostova - Gate 07C",',
      "]",
    ].join("\n");

    const secondaryFileContent = [
      "# -*- coding: utf-8 -*-",
      "# Replacement member lookup",
      "replacement_candidates = {",
      ...Array.from(
        { length: 40 },
        (_, i) =>
          `    "Candidate ${String(i + 1).padStart(2, "0")}": "Gate 00",`,
      ),
      '    "Diana Ross": "Gate 88D",',
      "}",
    ].join("\n");

    const readCalls: string[] = [];
    const scratchReadCalls: string[] = [];
    const editCalls: {
      path: string;
      old_string: string;
      new_string: string;
    }[] = [];

    const readFileTool = tool({
      name: "read_file",
      description: "Read the contents of a file at the given path.",
      parameters: z.object({
        path: z.string().describe("File path to read."),
      }),
      handler: ({ path }) => {
        readCalls.push(path);
        return Promise.resolve(
          path.includes("primary") ? primaryFileContent : secondaryFileContent,
        );
      },
    });

    const editFileTool = tool({
      name: "edit_file",
      description:
        "Replace an exact string in a file with a new string. Requires exact lines from the file.",
      parameters: z.object({
        path: z.string().describe("File path to edit."),
        old_string: z.string().describe("Exact existing text to replace."),
        new_string: z.string().describe("Replacement text."),
      }),
      handler: ({ path, old_string, new_string }) => {
        editCalls.push({ path, old_string, new_string });
        return Promise.resolve(`Successfully updated ${path}.`);
      },
    });

    const checkStatusTool = tool({
      name: "check_status",
      description: "Check roster validation status for a leg.",
      parameters: z.object({
        scope: z.string().describe("Either 'outbound' or 'return'."),
      }),
      handler: ({ scope }) => Promise.resolve(`Status for ${scope}: valid`),
    });

    const scratchStore = new Map<string, string>();
    const scratchPad: ToolOutputScratchPad = {
      get: (id: string) => {
        scratchReadCalls.push(id);
        return Promise.resolve(scratchStore.get(id));
      },
      set: (id: string, content: string) => {
        scratchStore.set(id, content);
        return Promise.resolve();
      },
      threshold: 20_000,
    };

    const fixedNow = 1_775_000_000_000;
    const history: HistoryEvent[] = [
      {
        id: "u-1",
        type: "participant_utterance",
        isOwn: false,
        name: "user",
        text: [
          "Execute these steps strictly one tool call per turn (never call multiple tools in the same turn):",
          "Turn 1: Call read_file on /app/roster_primary.py.",
          "Turn 2: Call read_file on /app/roster_secondary.py to look up Diana Ross's gate code.",
          "Turn 3: Call check_status with scope 'outbound'.",
          "Turn 4: Call check_status with scope 'return'.",
          "Turn 5: Call edit_file on /app/roster_primary.py to replace the exact return_roster entry for Alice Carter (keeping Martin Vance and Elena Rostova around it as context in old_string) with Diana Ross and her gate code from roster_secondary.py, then reply briefly.",
        ].join("\n"),
        timestamp: fixedNow,
      },
    ];

    await overrideTime(() => fixedNow)(
      agentDeps(history)(async () => {
        await runAgentWithProvider({
          maxIterations: 8,
          tools: [readFileTool, editFileTool, checkStatusTool],
          prompt:
            "You are a precise coding assistant. Follow the user's step-by-step sequence strictly one tool call per turn.",
          timezoneIANA: "UTC",
          toolOutputScratchPad: scratchPad,
        });
      }),
    )();

    const primaryReadCount =
      readCalls.filter((p) => p.includes("primary")).length;
    assertEquals(
      primaryReadCount,
      1,
      `Expected /app/roster_primary.py to be read only once in the turn, got ${primaryReadCount} reads (${
        JSON.stringify(readCalls)
      })`,
    );
    assertEquals(
      scratchReadCalls.length,
      0,
      `Expected 0 read_scratch_file calls for active-turn file reads, got ${
        JSON.stringify(scratchReadCalls)
      }`,
    );
    assertEquals(
      editCalls.length >= 1,
      true,
      "Expected agent to call edit_file on /app/roster_primary.py",
    );
    assertEquals(
      editCalls[0].new_string.includes("Diana Ross - Gate 88D"),
      true,
      `Expected edit_file to use exact gate code from secondary file and exact context from primary file, got: ${
        JSON.stringify(editCalls[0])
      }`,
    );
  },
);
