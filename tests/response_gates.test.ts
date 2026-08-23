import { assert, assertRejects } from "@std/assert";
import { runAgent } from "../mod.ts";
import { agentDeps, noopRewriteHistory } from "../test_helpers.ts";
import {
  forcedStopUtterance,
  type HistoryEvent,
  injectCallModel,
  ownThoughtTurn,
  ownUtteranceTurn,
  participantUtteranceTurn,
  stopThoughtDefault,
} from "../src/agent.ts";
import { genJsonOverride } from "../src/genJson.ts";
import { injectGeminiToken } from "../src/gemini.ts";

const run = (
  history: HistoryEvent[],
  fakeCallModel: (events: HistoryEvent[]) => Promise<HistoryEvent[]>,
  specOverrides: Partial<Parameters<typeof runAgent>[0]> = {},
) =>
  injectCallModel(fakeCallModel)(async () => {
    await agentDeps(history)(runAgent)({
      maxIterations: 20,
      tools: [],
      prompt: "You are a helpful assistant.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
      ...specOverrides,
    });
  })();

const baseHistory = (): HistoryEvent[] => [
  participantUtteranceTurn({ name: "user", text: "hi" }),
];

const truncatedUtterance = (text: string): HistoryEvent => ({
  type: "own_utterance",
  id: crypto.randomUUID(),
  timestamp: Date.now(),
  isOwn: true,
  text,
  truncated: true,
});

Deno.test("emoji flood gate: floods are retried until a clean response arrives", async () => {
  const history = baseHistory();
  let calls = 0;
  await run(history, () => {
    calls++;
    if (calls === 1) {
      return Promise.resolve([ownUtteranceTurn("😀".repeat(150))]);
    }
    return Promise.resolve([ownUtteranceTurn("all done")]);
  });
  const replies = history.filter((e) => e.type === "own_utterance");
  assert(replies.length === 1, "only the clean reply must be emitted");
  assert(
    replies[0].type === "own_utterance" && replies[0].text === "all done",
    "the clean reply must win",
  );
});

Deno.test("emoji flood gate: persistent flooding aborts the run", async () => {
  await assertRejects(
    () =>
      run(
        baseHistory(),
        () => Promise.resolve([ownUtteranceTurn("😀".repeat(150))]),
      ),
    Error,
    "emoji flood",
  );
});

Deno.test("repetition flood gate: persistent repetition aborts the run", async () => {
  await assertRejects(
    () =>
      run(
        baseHistory(),
        () => Promise.resolve([ownUtteranceTurn("abc".repeat(31))]),
      ),
    Error,
    "repetition flood",
  );
});

Deno.test("truncation gate: retries with a correctional thought referencing the token budget", async () => {
  const history = baseHistory();
  const inputs: HistoryEvent[][] = [];
  let calls = 0;
  await run(history, (events) => {
    inputs.push(events);
    calls++;
    if (calls === 1) {
      return Promise.resolve([
        truncatedUtterance("partial answer that got cut"),
      ]);
    }
    return Promise.resolve([ownUtteranceTurn("complete answer")]);
  });
  assert(calls === 2, "the truncated response must trigger exactly one retry");
  assert(
    inputs[1].some((e) =>
      e.type === "own_thought" &&
      e.text.includes("hit the output token budget")
    ),
    "the retry input must carry the truncation correctional thought",
  );
  const replies = history.filter((e) => e.type === "own_utterance");
  assert(replies.length === 1, "only the completed answer must be emitted");
});

Deno.test("truncation gate: after exhausting retries the truncated response is accepted with its flag stripped", async () => {
  const history = baseHistory();
  let calls = 0;
  await run(history, () => {
    calls++;
    return Promise.resolve([truncatedUtterance(`attempt ${calls}`)]);
  });
  assert(calls === 3, "two retries then acceptance on the third attempt");
  const replies = history.filter((e) => e.type === "own_utterance");
  assert(
    replies.length === 1 && replies[0].type === "own_utterance" &&
      replies[0].text === "attempt 3",
    "the final truncated attempt must be emitted",
  );
  assert(
    replies[0].type === "own_utterance" && !replies[0].truncated,
    "the emitted event must not keep the internal truncated flag",
  );
});

const runWithProgressAudit = (
  history: HistoryEvent[],
  auditResults: string[],
  fakeCallModel: (events: HistoryEvent[]) => Promise<HistoryEvent[]>,
) =>
  injectGeminiToken("fake-token")(async () => {
    let audits = 0;
    await genJsonOverride.inject(
      () =>
      (_opts: unknown, _sys: string, _zod: unknown) =>
      (): Promise<{ shouldContinue: boolean }> => {
        audits++;
        return Promise.resolve({
          shouldContinue: (auditResults[audits - 1] ?? "true") === "true",
        });
      },
    )(() => run(history, fakeCallModel, { maxIterations: 1 }))();
  })();

Deno.test("progress check gate: a soft stop injects the default stop thought into the next model call", async () => {
  const history = baseHistory();
  const inputs: HistoryEvent[][] = [];
  await runWithProgressAudit(history, ["false"], (events) => {
    inputs.push(events);
    return Promise.resolve(
      events.some((e) => e.type === "own_thought")
        ? [ownUtteranceTurn("wrapping up now")]
        : [ownThoughtTurn("thinking")],
    );
  });
  assert(
    inputs[0].some((e) =>
      e.type === "own_thought" && e.text.startsWith(stopThoughtDefault)
    ),
    "the next model call must see the stop-thought correction",
  );
});

Deno.test("progress check gate: two stop verdicts escalate to a forced user-facing stop utterance", async () => {
  const history = baseHistory();
  let calls = 0;
  await runWithProgressAudit(history, ["false", "false"], () => {
    calls++;
    return Promise.resolve([ownThoughtTurn("thinking")]);
  });
  assert(calls === 1, "the run must end before another model call");
  const stops = history.filter((e) =>
    e.type === "own_utterance" && e.text === forcedStopUtterance
  );
  assert(
    stops.length === 1,
    "exactly one forced stop utterance must be emitted",
  );
});
