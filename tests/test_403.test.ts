import { assert, assertEquals, assertRejects } from "@std/assert";
import { pipe } from "gamla";
import { injectGeminiErrorLogger, runAgent } from "../mod.ts";
import { type HistoryEvent, participantUtteranceTurn } from "../src/agent.ts";
import { agentDeps, injectSecrets } from "../test_helpers.ts";

const make403Response = (fileId: string) =>
  new Response(
    JSON.stringify({
      error: {
        code: 403,
        message:
          `You do not have permission to access the File ${fileId} or it may not exist.`,
        status: "PERMISSION_DENIED",
      },
    }),
    {
      status: 403,
      statusText: "Forbidden",
      headers: { "Content-Type": "application/json" },
    },
  );

const makeSuccessResponse = () =>
  new Response(
    JSON.stringify({
      candidates: [
        {
          content: { parts: [{ text: "Okay, I've ignored that file." }] },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 10,
        totalTokenCount: 20,
      },
    }),
    {
      status: 200,
      statusText: "OK",
      headers: { "Content-Type": "application/json" },
    },
  );

const historyWithFileAttachment = (fileId: string): HistoryEvent[] => [
  participantUtteranceTurn({
    name: "user",
    text: "Here is a file",
    attachments: [
      {
        kind: "file",
        fileUri:
          `https://generativelanguage.googleapis.com/v1beta/files/${fileId}`,
        mimeType: "image/jpeg",
      },
    ],
  }),
];

Deno.test(
  "agent gracefully handles 403 on missing file reference by rewriting history",
  injectSecrets(async () => {
    let rewriteCalled = false;
    let fetchCallCount = 0;

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (_input, _init) => {
      fetchCallCount++;
      if (fetchCallCount === 1) {
        return Promise.resolve(make403Response("gjwmj5neimif"));
      }
      return Promise.resolve(makeSuccessResponse());
    };

    try {
      const history = historyWithFileAttachment("gjwmj5neimif");

      await agentDeps(history)(runAgent)({
        maxIterations: 4,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "You are a helpful assistant.",
        rewriteHistory: (_replacements) => {
          rewriteCalled = true;
          return Promise.resolve();
        },
        timezoneIANA: "UTC",
        disableStreaming: true,
      });

      assert(rewriteCalled, "rewriteHistory was not called");
      assertEquals(
        fetchCallCount,
        2,
        "Fetch should have been called twice (1 error, 1 success retry)",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }),
);

Deno.test(
  "nuclear path: persistent 403 after stripping all files throws instead of crashing",
  injectSecrets(async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (_input, _init) =>
      Promise.resolve(make403Response("nonexistent"));

    try {
      const history = historyWithFileAttachment("someotherfileid");

      await assertRejects(
        () =>
          agentDeps(history)(runAgent)({
            maxIterations: 4,
            onMaxIterationsReached: () => {},
            tools: [],
            prompt: "You are a helpful assistant.",
            rewriteHistory: () => Promise.resolve(),
            timezoneIANA: "UTC",
            disableStreaming: true,
          }),
        Error,
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  }),
);

Deno.test(
  "geminiError logger does not fire when 403 is recovered by stripping files",
  pipe(
    injectGeminiErrorLogger(
      () => {
        throw new Error(
          "geminiError.access should not be called for recoverable 403s",
        );
      },
    ),
    injectSecrets,
  )(async () => {
    let fetchCallCount = 0;

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (_input, _init) => {
      fetchCallCount++;
      if (fetchCallCount === 1) {
        return Promise.resolve(make403Response("gjwmj5neimif"));
      }
      return Promise.resolve(makeSuccessResponse());
    };

    try {
      const history = historyWithFileAttachment("gjwmj5neimif");

      await agentDeps(history)(runAgent)({
        maxIterations: 4,
        onMaxIterationsReached: () => {},
        tools: [],
        prompt: "You are a helpful assistant.",
        rewriteHistory: () => Promise.resolve(),
        timezoneIANA: "UTC",
        disableStreaming: true,
      });

      assertEquals(fetchCallCount, 2);
    } finally {
      globalThis.fetch = originalFetch;
    }
  }),
);
