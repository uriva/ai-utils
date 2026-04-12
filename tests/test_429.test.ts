import { assert, assertEquals } from "@std/assert";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import { agentDeps, injectSecrets } from "../test_helpers.ts";
import { runAgent } from "../mod.ts";

const make429Response = () =>
  new Response(
    JSON.stringify({
      error: {
        code: 429,
        message: "RESOURCE_EXHAUSTED",
        status: "RESOURCE_EXHAUSTED",
      },
    }),
    {
      status: 429,
      statusText: "Too Many Requests",
      headers: { "Content-Type": "application/json" },
    },
  );

const makeSuccessResponse = () =>
  new Response(
    JSON.stringify({
      candidates: [
        {
          content: { parts: [{ text: "Got it!" }] },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
      },
    }),
    {
      status: 200,
      statusText: "OK",
      headers: { "Content-Type": "application/json" },
    },
  );

const simpleHistory: HistoryEvent[] = [
  participantUtteranceTurn({ name: "user", text: "Hello" }),
];

Deno.test(
  "agent retries on 429 and eventually succeeds",
  injectSecrets(async () => {
    let fetchCallCount = 0;
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (_input, _init) => {
      fetchCallCount++;
      if (fetchCallCount === 1) return Promise.resolve(make429Response());
      return Promise.resolve(makeSuccessResponse());
    };

    try {
      const history: HistoryEvent[] = [...simpleHistory];

      await agentDeps(history)(runAgent)({
        maxIterations: 4,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "You are a helpful assistant.",
        rewriteHistory: () => Promise.resolve(),
        timezoneIANA: "UTC",
        disableStreaming: true,
      });

      assert(
        fetchCallCount >= 2,
        `Fetch should be called at least twice, got ${fetchCallCount}`,
      );
      assert(
        history.some((e) => e.type === "own_utterance"),
        "Should emit a response after retry",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }),
);

Deno.test(
  "agent falls back to alternate model when 429 persists",
  injectSecrets(async () => {
    let fetchCallCount = 0;
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (_input, _init) => {
      fetchCallCount++;
      return Promise.resolve(make429Response());
    };

    try {
      const history: HistoryEvent[] = [...simpleHistory];

      await agentDeps(history)(runAgent)({
        maxIterations: 4,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "You are a helpful assistant.",
        rewriteHistory: () => Promise.resolve(),
        timezoneIANA: "UTC",
        disableStreaming: true,
      });

      assertEquals(
        true,
        false,
        "Should have thrown",
      );
    } catch (error) {
      assert(error instanceof Error, "Should throw an Error");
      assert(
        fetchCallCount > 4,
        `Should have retried multiple times, got ${fetchCallCount} calls`,
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }),
);
