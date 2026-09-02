import { assert } from "@std/assert";
import { injectCallModel, injectGeminiToken, runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { genJsonOverride } from "../src/genJson.ts";
import { agentDeps, noopRewriteHistory, someTool } from "../test_helpers.ts";

// checkProgress is a periodic auditor model call. It must not bill full-model
// tokens over the entire history: it runs on the mini model and only sees a
// bounded recent slice of the plain-text history.

const hugeFiller = "x".repeat(50_000);

Deno.test("progress audit runs on the mini model over a bounded recent-history slice", async () => {
  const captured: { opts: unknown; userMsgLength: number }[] = [];
  const fakeGenJson =
    (opts: unknown, _sys: string, _zod: unknown) =>
    (userMsg: string): Promise<Record<string, string>> => {
      captured.push({ opts, userMsgLength: userMsg.length });
      return Promise.resolve({ shouldContinue: "true" });
    };

  const mockHistory: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: `Please help me. ${hugeFiller} END-OF-REQUEST`,
    }),
  ];

  let callCount = 0;
  const fakeCallModel = (): Promise<HistoryEvent[]> => {
    callCount++;
    if (callCount === 1) {
      return Promise.resolve([
        {
          type: "tool_call" as const,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          name: "dummy_tool",
          parameters: {},
          isOwn: true as const,
        },
      ]);
    }
    return Promise.resolve([ownUtteranceTurn("done")]);
  };

  await injectGeminiToken("fake-token")(async () => {
    await genJsonOverride.inject(() => fakeGenJson)(async () => {
      await injectCallModel(fakeCallModel)(() =>
        agentDeps(mockHistory)(runAgent)({
          maxIterations: 1,
          tools: [someTool],
          prompt: "You are an assistant.",
          rewriteHistory: noopRewriteHistory,
          timezoneIANA: "UTC",
        })
      )();
    })();
  })();

  assert(
    captured.length >= 1,
    "expected the progress audit to fire at least once",
  );
  for (const call of captured) {
    assert(
      (call.opts as { tier?: string }).tier === "flash",
      "progress audit must use the flash model tier, not the full-size one",
    );
    assert(
      call.userMsgLength < 35_000,
      `progress audit saw ${call.userMsgLength} chars of history; it must send only a bounded recent slice`,
    );
  }
});
