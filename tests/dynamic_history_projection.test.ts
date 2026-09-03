import { assertEquals, assertNotEquals } from "@std/assert";
import type { Injector } from "@uri/inject";
import {
  injectCacher,
  projectHistoryToModelContext,
  runAgent,
  sanitizeWindowBoundary,
  type ToolOutputScratchPad,
} from "../mod.ts";
import { agentDeps, someTool } from "../test_helpers.ts";
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
          skillsToReLearn: "None",
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
