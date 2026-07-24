import { assertEquals } from "@std/assert";
import {
  participantUtteranceTurn,
  scheduleHistoryCompaction,
} from "../src/agent.ts";
import { noopRewriteHistory } from "../test_helpers.ts";

// A compaction failure swallowed with console.error lets history grow
// unbounded while looking healthy in logs, silently multiplying token spend
// on every model call. Compaction failures must instead surface as
// unhandled rejections so they are impossible to miss.
Deno.test("scheduleHistoryCompaction surfaces compactHistory failure as an unhandled rejection", async () => {
  const compactionError = new Error("compaction exploded");
  const unhandled: unknown[] = [];
  const listener = (e: PromiseRejectionEvent) => {
    e.preventDefault();
    unhandled.push(e.reason);
  };
  globalThis.addEventListener("unhandledrejection", listener);
  scheduleHistoryCompaction(
    {
      prompt: "prompt",
      tools: [],
      maxIterations: 1,
      timezoneIANA: "UTC",
      rewriteHistory: noopRewriteHistory,
      compactHistory: () => Promise.reject(compactionError),
      historyCompactionTokenThreshold: 1,
    },
    [participantUtteranceTurn({ name: "user", text: "hello" })],
  );
  await new Promise((resolve) => setTimeout(resolve, 100));
  globalThis.removeEventListener("unhandledrejection", listener);
  assertEquals(unhandled, [compactionError]);
});

Deno.test("scheduleHistoryCompaction does not run compaction under the threshold", async () => {
  let compacted = false;
  const unhandled: unknown[] = [];
  const listener = (e: PromiseRejectionEvent) => {
    e.preventDefault();
    unhandled.push(e.reason);
  };
  globalThis.addEventListener("unhandledrejection", listener);
  scheduleHistoryCompaction(
    {
      prompt: "prompt",
      tools: [],
      maxIterations: 1,
      timezoneIANA: "UTC",
      rewriteHistory: noopRewriteHistory,
      compactHistory: () => {
        compacted = true;
        return Promise.resolve();
      },
      historyCompactionTokenThreshold: 1000000,
    },
    [participantUtteranceTurn({ name: "user", text: "hello" })],
  );
  await new Promise((resolve) => setTimeout(resolve, 100));
  globalThis.removeEventListener("unhandledrejection", listener);
  assertEquals(compacted, false);
  assertEquals(unhandled, []);
});
