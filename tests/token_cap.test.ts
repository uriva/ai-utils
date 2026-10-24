import { assert, assertEquals } from "@std/assert";
import {
  capEventsToTokenBudget,
  getOversizedMediaText,
  injectGeminiSdkExchange,
} from "../src/geminiAgent.ts";
import {
  injectRawUploadBlobToGemini,
  maxGeminiTextAttachmentBytes,
} from "../src/gemini.ts";
import {
  estimateTokensLocal,
  type HistoryEvent,
  injectAccessHistory,
  injectMetadataStore,
  injectOutputEvent,
  injectTextTokenCounter,
  injectTokenCounter,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  injectDecisionModel,
  injectRouteThinkingLevelTimeoutMs,
  runAgent,
} from "../mod.ts";
import { map, pipe, sum } from "gamla";

const makeUtterance = (id: string, text: string, timestamp: number) => ({
  type: "participant_utterance" as const,
  isOwn: false,
  id,
  timestamp,
  name: "user",
  text,
});

const makeOwnUtterance = (id: string, text: string, timestamp: number) => ({
  type: "own_utterance" as const,
  isOwn: true,
  id,
  timestamp,
  text,
  modelMetadata: {
    type: "gemini" as const,
    responseId: `resp-${id}`,
    thoughtSignature: "",
  },
});

const makeToolCall = (id: string, timestamp: number) => ({
  type: "tool_call" as const,
  isOwn: true,
  id,
  timestamp,
  name: "someTool",
  parameters: { foo: "bar" },
  modelMetadata: {
    type: "gemini" as const,
    responseId: `resp-${id}`,
    thoughtSignature: "sig",
  },
});

const makeToolResult = (id: string, toolCallId: string, timestamp: number) => ({
  type: "tool_result" as const,
  isOwn: true,
  id,
  timestamp,
  name: "someTool",
  result: "ok",
  toolCallId,
});

// ~650 chars => ~211 tokens each
const longText = "x ".repeat(650);

Deno.test(
  "capEventsToTokenBudget drops oldest events when total exceeds budget",
  async () => {
    // deno-lint-ignore no-explicit-any
    const events: any[] = Array.from(
      { length: 100 },
      (_, i) => makeUtterance(`msg${i}`, longText, i * 1000),
    );

    const totalBefore = sum(map(estimateTokensLocal)(events));
    assert(
      totalBefore > 10000,
      `Total tokens ${totalBefore} should exceed 10000`,
    );

    // deno-lint-ignore no-explicit-any
    const capped = await capEventsToTokenBudget(10000)(events as any);
    const totalAfter = sum(map(estimateTokensLocal)(capped));

    assert(
      totalAfter <= 10000,
      `Capped tokens ${totalAfter} should be <= 10000`,
    );
    assert(capped.length < events.length, "Should have fewer events after cap");
    // Should keep the newest events
    assertEquals(
      capped[capped.length - 1].id,
      "msg99",
      "Should keep the newest event",
    );
  },
);

Deno.test(
  "capEventsToTokenBudget cleans up orphaned tool results after dropping",
  async () => {
    // Create a history where old tool_call + tool_result pairs get split
    // when oldest events are dropped
    // deno-lint-ignore no-explicit-any
    const events: any[] = [
      // Old tool call (will be dropped)
      makeToolCall("tc0", 100),
      // Old tool result (should also be cleaned up since tc0 is dropped)
      makeToolResult("tr0", "tc0", 200),
      // Many filler events to push over budget
      ...Array.from(
        { length: 50 },
        (_, i) => makeUtterance(`filler${i}`, longText, 1000 + i * 100),
      ),
      // Recent tool call + result (should be kept)
      makeToolCall("tc1", 10000),
      makeToolResult("tr1", "tc1", 10100),
      makeOwnUtterance("final", "done", 10200),
    ];

    // Use a budget that keeps ~half the filler + the recent events
    const totalTokens = sum(map(estimateTokensLocal)(events));
    const budget = Math.floor(totalTokens / 2);

    // deno-lint-ignore no-explicit-any
    const capped = await capEventsToTokenBudget(budget)(events as any);

    // The old tool_result for tc0 should NOT be in the output
    // (either dropped with tc0 or cleaned as orphan)
    const orphanedResults = capped.filter(
      // deno-lint-ignore no-explicit-any
      (e: any) => e.type === "tool_result" && e.toolCallId === "tc0",
    );
    assertEquals(
      orphanedResults.length,
      0,
      "Orphaned tool_result for dropped tool_call should be removed",
    );

    // The recent tc1/tr1 pair should still be intact
    // deno-lint-ignore no-explicit-any
    const recentCall = capped.find((e: any) => e.id === "tc1");
    // deno-lint-ignore no-explicit-any
    const recentResult = capped.find((e: any) => e.id === "tr1");
    assert(recentCall, "Recent tool_call should be kept");
    assert(recentResult, "Recent tool_result should be kept");
  },
);

Deno.test(
  "capEventsToTokenBudget is a no-op when events are within budget",
  async () => {
    // deno-lint-ignore no-explicit-any
    const events: any[] = [
      makeUtterance("msg0", "Hello", 100),
      makeOwnUtterance("msg1", "Hi there", 200),
    ];

    const totalTokens = sum(map(estimateTokensLocal)(events));
    const capped = await capEventsToTokenBudget(totalTokens + 1000)(
      // deno-lint-ignore no-explicit-any
      events as any,
    );

    assertEquals(capped.length, events.length, "No events should be dropped");
    assertEquals(capped[0].id, "msg0");
    assertEquals(capped[1].id, "msg1");
  },
);

Deno.test(
  "capEventsToTokenBudget also cleans orphaned tool_calls (no matching result)",
  async () => {
    // deno-lint-ignore no-explicit-any
    const events: any[] = [
      // Old tool result that matches old call
      makeToolResult("tr0", "tc0", 50),
      // Old tool call whose result was already dropped
      makeToolCall("tc0", 100),
      // Filler
      ...Array.from(
        { length: 30 },
        (_, i) => makeUtterance(`f${i}`, longText, 500 + i * 100),
      ),
      makeOwnUtterance("final", "end", 10000),
    ];

    const totalTokens = sum(map(estimateTokensLocal)(events));
    const budget = Math.floor(totalTokens / 3);

    // deno-lint-ignore no-explicit-any
    const capped = await capEventsToTokenBudget(budget)(events as any);

    // After capping, if tr0 was dropped but tc0 somehow survived,
    // the orphan filter should handle it. Or both are dropped.
    // Either way, no orphaned tool_results should remain.
    const allToolResults = capped.filter(
      // deno-lint-ignore no-explicit-any
      (e: any) => e.type === "tool_result",
    );
    const allToolCallIds = new Set(
      // deno-lint-ignore no-explicit-any
      capped.filter((e: any) => e.type === "tool_call").map((e: any) => e.id),
    );
    for (const tr of allToolResults) {
      // deno-lint-ignore no-explicit-any
      const toolCallId = (tr as any).toolCallId;
      assert(
        allToolCallIds.has(toolCallId),
        `tool_result ${tr.id} references missing tool_call ${toolCallId}`,
      );
    }
  },
);

Deno.test(
  "runAgent skips BPE token counting on hot path when input size is well below token budget",
  async () => {
    let tokenCounterCalls = 0;
    let textTokenCounterCalls = 0;
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Quick ping" }),
    ];

    await pipe(
      injectTokenCounter(() => {
        tokenCounterCalls++;
        return Promise.resolve(10);
      }),
      injectTextTokenCounter(() => {
        textTokenCounterCalls++;
        return Promise.resolve(10);
      }),
      injectGeminiSdkExchange(() =>
        Promise.resolve({
          parts: [{ text: "Pong" }],
          finishReason: "STOP",
        })
      ),
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      }),
    )(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "Reply briefly.",
      timezoneIANA: "UTC",
    });

    assertEquals(tokenCounterCalls, 0);
    assertEquals(textTokenCounterCalls, 0);
  },
);

Deno.test(
  "runAgent avoids redundant metadata mget calls and duplicate scratchpad writes across consecutive turns",
  async () => {
    const mgetCalls: string[][] = [];
    const scratchSetCalls: string[] = [];
    const now = Date.now();
    const thirtyMinsAgo = now - 30 * 60 * 1000;
    const rawHistory: HistoryEvent[] = [
      {
        ...makeUtterance("u-1", "Run the report", thirtyMinsAgo),
        isOwn: false,
      },
      { ...makeToolCall("tc-old", thirtyMinsAgo + 100), isOwn: true },
      {
        ...makeToolResult("tr-old", "tc-old", thirtyMinsAgo + 200),
        isOwn: true,
        result: "x".repeat(3000),
      },
      {
        ...makeOwnUtterance("bot-1", "Done", thirtyMinsAgo + 300),
        isOwn: true,
      },
      {
        type: "tool_call" as const,
        isOwn: true,
        id: "auto-unlearn-123",
        timestamp: thirtyMinsAgo + 400,
        name: "unlearn_skill",
        parameters: { skillName: "demo" },
      },
      {
        ...makeToolResult(
          "auto-unlearn-123-result",
          "auto-unlearn-123",
          thirtyMinsAgo + 450,
        ),
        isOwn: true,
        result: "Unlearned",
      },
      {
        ...makeUtterance("u-2", "First follow-up", now - 5000),
        isOwn: false,
      },
    ];

    const runTurn = pipe(
      injectMetadataStore(() => ({
        get: () => Promise.resolve(null),
        set: () => Promise.resolve(),
        mget: (ids: string[]) => {
          mgetCalls.push(ids);
          return Promise.resolve(ids.map(() => null));
        },
      })),
      injectDecisionModel(() => Promise.resolve({})),
      injectGeminiSdkExchange(() =>
        Promise.resolve({
          parts: [{ text: "Quick reply", thoughtSignature: "sig-reply" }],
          finishReason: "STOP",
        })
      ),
      injectAccessHistory(() => Promise.resolve([...rawHistory])),
      injectOutputEvent((event) => {
        rawHistory.push(event);
        return Promise.resolve();
      }),
    )(runAgent);

    const scratchPad = {
      get: () => Promise.resolve(undefined),
      set: (id: string) => {
        scratchSetCalls.push(id);
        return Promise.resolve();
      },
    };

    await runTurn({
      maxIterations: 1,
      tools: [],
      prompt: "Reply briefly.",
      timezoneIANA: "UTC",
      toolOutputScratchPad: scratchPad,
    });

    rawHistory.push({
      ...makeUtterance("u-3", "Second follow-up", now),
      isOwn: false,
    });

    await runTurn({
      maxIterations: 1,
      tools: [],
      prompt: "Reply briefly.",
      timezoneIANA: "UTC",
      toolOutputScratchPad: scratchPad,
    });

    assertEquals(
      mgetCalls.flat(),
      [],
      `Expected zero metadataStore.mget lookups when all model events carry inline metadata and remaining events are user/tool_result/auto-unlearn, got: ${
        JSON.stringify(mgetCalls)
      }`,
    );
    assertEquals(
      scratchSetCalls,
      ["tr-old"],
      `Expected old tool result tr-old to be written to scratchpad only once across consecutive turns, got: ${
        JSON.stringify(scratchSetCalls)
      }`,
    );
  },
);

Deno.test(
  "runAgent bounds routeThinkingLevel latency when decision model stalls",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Hello there" }),
    ];
    let slowTimer: ReturnType<typeof setTimeout> | undefined;
    const t0 = performance.now();

    try {
      await pipe(
        injectRouteThinkingLevelTimeoutMs(() => 50),
        injectDecisionModel(
          () =>
            new Promise((resolve) => {
              slowTimer = setTimeout(() => resolve({}), 2000);
            }),
        ),
        injectGeminiSdkExchange(() =>
          Promise.resolve({
            parts: [{ text: "Fast reply" }],
            finishReason: "STOP",
          })
        ),
        injectAccessHistory(() => Promise.resolve(history)),
        injectOutputEvent((event) => {
          history.push(event);
          return Promise.resolve();
        }),
      )(runAgent)({
        maxIterations: 5,
        tools: [],
        prompt: "Reply briefly.",
        timezoneIANA: "UTC",
      });
    } finally {
      if (slowTimer !== undefined) clearTimeout(slowTimer);
    }

    const elapsedMs = performance.now() - t0;
    assert(
      elapsedMs < 500,
      `Expected runAgent not to block on stalled decision model, but took ${
        Math.round(elapsedMs)
      }ms`,
    );
  },
);

Deno.test(
  "runAgent recovers from Gemini token limit exceeded caused by oversized file attachment",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please improve the design of this site [Attached file: text/html - Download URL: https://example.com/site.html]",
        attachments: [
          {
            kind: "file",
            mimeType: "text/html",
            fileUri:
              "https://generativelanguage.googleapis.com/v1beta/files/hugehtml123",
          },
        ],
      }),
    ];
    let callCount = 0;

    await pipe(
      injectDecisionModel(() => Promise.resolve({})),
      injectGeminiSdkExchange((_signal, { req }) => {
        callCount++;
        const contents = Array.isArray(req.contents) ? req.contents : [];
        const hasFileData = contents.some((c) =>
          typeof c === "object" && c !== null && "parts" in c &&
          Array.isArray(c.parts) &&
          c.parts.some((p) => "fileData" in p && p.fileData)
        );
        if (hasFileData) {
          const err = new Error(
            "The input token count exceeds the maximum number of tokens allowed (1048576).",
          );
          Object.assign(err, { status: 400 });
          return Promise.reject(err);
        }
        return Promise.resolve({
          parts: [{
            text: "I can fetch the HTML from the download URL and update it.",
            thoughtSignature: "sig-ok",
          }],
          finishReason: "STOP",
        });
      }),
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      }),
    )(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "Reply briefly.",
      timezoneIANA: "UTC",
    });

    assertEquals(callCount, 2);
    const utterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(utterances.length, 1);
  },
);

Deno.test(
  "runAgent degrades oversized inline text attachment without uploading it to Gemini",
  async () => {
    const oversizedHtml = "<html><body>" +
      "A".repeat(maxGeminiTextAttachmentBytes + 1024) + "</body></html>";
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          "Please improve this page [Attached file: text/html - Download URL: https://example.com/page.html]",
        attachments: [
          {
            kind: "inline",
            mimeType: "text/html",
            dataBase64: btoa(oversizedHtml),
          },
        ],
      }),
    ];
    let uploadCalled = false;
    let receivedFileData = false;

    await pipe(
      injectDecisionModel(() => Promise.resolve({})),
      injectRawUploadBlobToGemini(() => () => {
        uploadCalled = true;
        return Promise.resolve({
          geminiUri:
            "https://generativelanguage.googleapis.com/v1beta/files/unexpected",
          mimeType: "text/html",
        });
      }),
      injectGeminiSdkExchange((_signal, { req }) => {
        const contents = Array.isArray(req.contents) ? req.contents : [];
        receivedFileData = contents.some((c) =>
          typeof c === "object" && c !== null && "parts" in c &&
          Array.isArray(c.parts) &&
          c.parts.some((p) => "fileData" in p && p.fileData)
        );
        return Promise.resolve({
          parts: [{
            text: "I will inspect the HTML via the download link.",
            thoughtSignature: "sig-ok",
          }],
          finishReason: "STOP",
        });
      }),
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      }),
    )(runAgent)({
      maxIterations: 1,
      tools: [],
      prompt: "Reply briefly.",
      timezoneIANA: "UTC",
    });

    assertEquals(uploadCalled, false);
    assertEquals(receivedFileData, false);
    assertEquals(
      history.filter((e) => e.type === "own_utterance").length,
      1,
    );
    assert(
      getOversizedMediaText([
        {
          kind: "file",
          mimeType: "text/html",
          fileUri: "https://example.com/page.html",
        },
      ]).includes("text/html"),
    );
  },
);
