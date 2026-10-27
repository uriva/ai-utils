import { assert, assertEquals } from "@std/assert";
import { pipe } from "gamla";
import { runAgent } from "../mod.ts";
import { agentDeps, someTool } from "../test_helpers.ts";
import {
  type HistoryEvent,
  injectCallModel,
  ownUtteranceTurn,
} from "../src/agent.ts";
import { genJsonOverride } from "../src/genJson.ts";

Deno.test(
  "runAgent - triggers compaction for settled past sessions even when total history is well below 100k",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000; // 24 hours ago
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "detailed discussion about previous task parameters and logs ".repeat(50); // ~500 tokens per message

    // Create 3 distinct past sessions separated by 1-hour gaps, totaling ~20k tokens
    const history: HistoryEvent[] = [];
    for (let session = 0; session < 3; session++) {
      const sessionTime = baseTime + session * 2 * hourMs;
      for (let turn = 0; turn < 6; turn++) {
        history.push({
          id: `p-${session}-${turn}`,
          type: "participant_utterance",
          isOwn: false,
          name: "user",
          text: `User message session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000,
        });
        history.push({
          id: `o-${session}-${turn}`,
          type: "own_utterance",
          isOwn: true,
          text:
            `Assistant response session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000 + 500,
        });
      }
    }

    // New active turn starting now
    const now = Date.now();
    history.push({
      id: "p-active",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text: "What is the summary of our previous decisions?",
      timestamp: now,
    });

    let receivedHistory: HistoryEvent[] = [];

    const fakeCallModel = (
      passedHistory: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      receivedHistory = passedHistory;
      return Promise.resolve([
        ownUtteranceTurn("Here is the summary based on past records."),
      ]);
    };

    const fakeGenJson =
      (_opts: unknown, _sys: string, _zod: unknown) =>
      (_userMsg: string): Promise<Record<string, string>> => {
        return Promise.resolve({
          entities: "Past task parameters",
          decisions: "Agreed on parameters",
          actions: "Logs processed",
          pendingItems: "None",
          abandonedItems: "None",
          context: "Past session details",
        });
      };

    await pipe(
      injectCallModel(fakeCallModel),
      genJsonOverride.inject(() => fakeGenJson),
    )(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 5,
        tools: [someTool],
        prompt: "You are a helpful assistant.",
        timezoneIANA: "UTC",
      });
    })();

    assertEquals(
      receivedHistory.length,
      4,
      "Expected 37 raw events from 3 settled sessions to be dynamically compacted into 3 summaries + 1 active message",
    );
    assertEquals(
      receivedHistory.filter((e) =>
        e.type === "own_thought" &&
        typeof e.text === "string" &&
        e.text.includes("Past conversation history was compacted")
      ).length,
      3,
      "Expected all 3 past settled sessions to have structured summaries in model context",
    );
    assertEquals(
      history.length,
      38, // 37 input events + 1 emitted response
      "Expected raw history to remain immutable with all original events preserved",
    );
  },
);

Deno.test(
  "runAgent - respects historyCompactionTokenThreshold if specified and does not compact settled sessions when below threshold",
  async () => {
    const baseTime = Date.now() - 24 * 60 * 60 * 1000; // 24 hours ago
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "detailed discussion about previous task parameters and logs ".repeat(50); // ~500 tokens per message

    // Create 3 distinct past sessions totaling ~20k tokens
    const history: HistoryEvent[] = [];
    for (let session = 0; session < 3; session++) {
      const sessionTime = baseTime + session * 2 * hourMs;
      for (let turn = 0; turn < 6; turn++) {
        history.push({
          id: `p-${session}-${turn}`,
          type: "participant_utterance",
          isOwn: false,
          name: "user",
          text: `User message session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000,
        });
        history.push({
          id: `o-${session}-${turn}`,
          type: "own_utterance",
          isOwn: true,
          text:
            `Assistant response session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000 + 500,
        });
      }
    }

    history.push({
      id: "p-active",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text: "What is the status?",
      timestamp: Date.now(),
    });

    let receivedHistory: HistoryEvent[] = [];

    const fakeCallModel = (
      passedHistory: HistoryEvent[],
    ): Promise<HistoryEvent[]> => {
      receivedHistory = passedHistory;
      return Promise.resolve([
        ownUtteranceTurn("Status is normal."),
      ]);
    };

    await pipe(
      injectCallModel(fakeCallModel),
    )(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 5,
        tools: [someTool],
        prompt: "You are a helpful assistant.",
        timezoneIANA: "UTC",
        historyCompactionTokenThreshold: 50_000, // Explicitly higher than the ~20k tokens in history
      });
    })();

    assertEquals(
      receivedHistory.length,
      37,
      "Expected all 37 raw events to be preserved in model context because history is under 50k threshold",
    );
  },
);

Deno.test(
  "runAgent - when compactHistory callback is configured, settled session summarization runs in background without blocking callModel",
  async () => {
    const delay = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));
    const baseTime = Date.now() - 24 * 60 * 60 * 1000;
    const hourMs = 60 * 60 * 1000;
    const fillerText =
      "uncached settled session log entry for background compaction test "
        .repeat(50);

    const history: HistoryEvent[] = [];
    for (let session = 0; session < 3; session++) {
      const sessionTime = baseTime + session * 2 * hourMs;
      for (let turn = 0; turn < 6; turn++) {
        history.push({
          id: `bg-p-${session}-${turn}`,
          type: "participant_utterance",
          isOwn: false,
          name: "user",
          text:
            `User message bg session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000,
        });
        history.push({
          id: `bg-o-${session}-${turn}`,
          type: "own_utterance",
          isOwn: true,
          text:
            `Assistant response bg session ${session} turn ${turn}: ${fillerText}`,
          timestamp: sessionTime + turn * 1000 + 500,
        });
      }
    }

    history.push({
      id: "bg-p-active",
      type: "participant_utterance",
      isOwn: false,
      name: "user",
      text: "Quick ping!",
      timestamp: Date.now(),
    });

    const SUMMARY_LATENCY_MS = 300;
    let callModelStartMs = 0;
    let compactHistoryCalledWith: HistoryEvent[] | undefined;

    const fakeGenJson =
      (_opts: unknown, _sys: string, _zod: unknown) =>
      async (_userMsg: string): Promise<Record<string, string>> => {
        await delay(SUMMARY_LATENCY_MS);
        return {
          entities: "Past task parameters",
          decisions: "Agreed on parameters",
          actions: "Logs processed",
          pendingItems: "None",
          abandonedItems: "None",
          context: "Past session details",
        };
      };

    const fakeCallModel = (): Promise<HistoryEvent[]> => {
      callModelStartMs = performance.now();
      return Promise.resolve([ownUtteranceTurn("Pong!")]);
    };

    const t0 = performance.now();
    await pipe(
      injectCallModel(fakeCallModel),
      genJsonOverride.inject(() => fakeGenJson),
    )(async () => {
      await agentDeps(history)(runAgent)({
        maxIterations: 5,
        tools: [someTool],
        prompt: "You are a helpful assistant.",
        timezoneIANA: "UTC",
        historyCompactionTokenThreshold: 10_000,
        compactHistory: (h) => {
          compactHistoryCalledWith = h;
          return Promise.resolve();
        },
      });
    })();

    const delayUntilCallModelMs = callModelStartMs - t0;
    assert(
      delayUntilCallModelMs < 100,
      `Expected callModel to start without blocking on 300ms settled-session summarization (< 100ms), but took ${
        Math.round(delayUntilCallModelMs)
      }ms`,
    );

    await delay(50);
    assert(
      compactHistoryCalledWith !== undefined,
      "Expected scheduleHistoryCompaction to invoke compactHistory in the background",
    );
  },
);
