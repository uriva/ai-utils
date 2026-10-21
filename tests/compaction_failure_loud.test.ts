import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import {
  estimateAgentInputTokens,
  injectTokenCounter,
  participantUtteranceTurn,
  scheduleHistoryCompaction,
  tool,
} from "../src/agent.ts";

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

Deno.test("scheduleHistoryCompaction skips full token counting when active turn spec and history are under thresholds despite large unlearned skills", async () => {
  let tokenCounterCalls = 0;
  await injectTokenCounter((events) => {
    tokenCounterCalls++;
    return Promise.resolve(events.length * 10);
  })(async () => {
    scheduleHistoryCompaction(
      {
        prompt: "short prompt",
        tools: [],
        skills: [
          {
            name: "huge_unlearned_skill",
            description: "desc",
            instructions: "word ".repeat(24_000),
            tools: [],
          },
        ],
        maxIterations: 1,
        timezoneIANA: "UTC",
        compactHistory: () => Promise.resolve(),
        historyCompactionTokenThreshold: 100_000,
      },
      [participantUtteranceTurn({ name: "user", text: "hello" })],
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
  })();
  assertEquals(tokenCounterCalls, 0);
});

Deno.test("estimateAgentInputTokens memoizes tool, skill, and prompt token counts across repeated calls", async () => {
  const tools = Array.from({ length: 10 }, (_, i) =>
    tool({
      name: `tool_${i}`,
      description: `Detailed tool description ${i} `.repeat(100),
      parameters: z.object({
        query: z.string(),
        limit: z.number().optional(),
        tags: z.array(z.string()).optional(),
      }),
      handler: () => Promise.resolve("ok"),
    }));
  const skills = Array.from({ length: 5 }, (_, i) => ({
    name: `skill_${i}`,
    description: `Skill ${i}`,
    instructions: `Skill instructions ${i} `.repeat(400),
    tools: tools.slice(0, 4),
  }));
  const spec = {
    prompt: "System prompt text ".repeat(400),
    tools,
    skills,
    maxIterations: 10,
    timezoneIANA: "UTC",
  };
  const firstTokens = await estimateAgentInputTokens(spec, []);
  const t0 = performance.now();
  const secondTokens = await estimateAgentInputTokens(spec, []);
  const elapsedMs = performance.now() - t0;
  assertEquals(secondTokens, firstTokens);
  assert(
    elapsedMs < 10,
    `Expected memoized estimateAgentInputTokens to take <10ms, took ${elapsedMs}ms`,
  );
});
