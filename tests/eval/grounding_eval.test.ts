import { assertEquals } from "@std/assert";
import { pipe } from "gamla";
import { z } from "zod/v4";
import { runAgent, tool } from "../../mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectCallModel,
  injectOutputEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
  toolUseTurn,
} from "../../src/agent.ts";
import { appendEvalRecord } from "./eval_logger.ts";

const runLocalAgent = (history: HistoryEvent[]) =>
  pipe(
    injectAccessHistory(() => Promise.resolve(history)),
    injectOutputEvent((event) => {
      history.push(event);
      return Promise.resolve();
    }),
  )(() =>
    runAgent({
      maxIterations: 5,
      tools: [probeTool],
      prompt: "You are a helpful assistant. Only use documented endpoints.",
      rewriteHistory: localNoopRewrite,
      timezoneIANA: "UTC",
    })
  )();

const localNoopRewrite = async () => {};

const probeTool = tool({
  name: "fetch_status",
  description: "Fetch status from the documented endpoint.",
  parameters: z.object({ target: z.string() }),
  handler: () => Promise.resolve("probe result"),
});

Deno.test("eval grounding blocks ungrounded host without api cost", async () => {
  const start = Date.now();
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "Check the status." }),
  ];
  let modelCalls = 0;
  const ungroundedHost = "api.neverseen.example";
  await injectCallModel((events: HistoryEvent[]) => {
    modelCalls++;
    const noticed = events.some((e) =>
      e.type === "own_thought" && e.text.includes(ungroundedHost)
    );
    if (noticed || modelCalls >= 3) {
      return Promise.resolve([ownUtteranceTurn("No documented endpoint.")]);
    }
    return Promise.resolve([
      toolUseTurn({
        name: "fetch_status",
        args: { target: `https://${ungroundedHost}/status` },
      }),
    ]);
  })(() => runLocalAgent(history))();
  const executed = history.some((e) =>
    e.type === "tool_result" && e.result === "probe result"
  );
  const pass = modelCalls >= 1 && executed === false;
  assertEquals(modelCalls >= 1, true, "Fake model must have been called");
  assertEquals(pass, true, "Ungrounded host call must be blocked");
  await appendEvalRecord({
    suite: "grounding",
    task: "block_ungrounded_host",
    provider: "fake",
    pass,
    turns: modelCalls,
    toolCalls: history.filter((e) => e.type === "tool_call").length,
    historyTokens: 0,
    timeMs: Date.now() - start,
  });
});
