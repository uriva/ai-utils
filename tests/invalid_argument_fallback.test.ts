import { assert, assertEquals, assertRejects } from "@std/assert";
import { runAgent, z } from "../mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { injectGeminiSdkExchange } from "../src/geminiAgent.ts";
import {
  geminiFallbackVersion,
  geminiGenJsonFromConvo,
  injectGeminiGenerateContent,
} from "../src/gemini.ts";
import { pipe } from "gamla";

// Gemini-specific: the alternate-model fallback wiring lives in geminiAgent.ts,
// so this test cannot run on other providers. Reproduces a production incident
// where Gemini's primary serving path intermittently rejected valid requests
// with 400 INVALID_ARGUMENT (13 fleet-wide occurrences in one day vs 1 the day
// before; the exact same requests replayed successfully afterwards). The engine
// treated the 400 as non-retryable, never tried the fallback model, and surfaced
// an immediate user-facing error. The exchange is scripted via
// injectGeminiSdkExchange, so no API call is made.

export const invalidArgumentErrorMessage = JSON.stringify({
  error: {
    code: 400,
    message: "Request contains an invalid argument.",
    status: "INVALID_ARGUMENT",
  },
});

const invalidArgumentError = () =>
  Object.assign(new Error(invalidArgumentErrorMessage), { status: 400 });

const recoveredText = "Recovered on the fallback model.";

const recoveredExchange = {
  parts: [{ text: recoveredText }],
  finishReason: "STOP",
};

const baseSpec = {
  maxIterations: 5,
  tools: [],
  prompt: "You are a helpful assistant.",
  rewriteHistory: () => Promise.resolve(),
  timezoneIANA: "UTC",
};

Deno.test("400 INVALID_ARGUMENT on the primary model falls back to the alternate model", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "hi" }),
  ];
  const attemptedModels: string[] = [];
  const scriptedExchange = (
    _signal: AbortSignal,
    args: { req: { model?: string } },
  ) => {
    attemptedModels.push(args.req.model ?? "");
    return args.req.model === geminiFallbackVersion
      ? Promise.resolve(recoveredExchange)
      : Promise.reject(invalidArgumentError());
  };

  await pipe(
    injectGeminiSdkExchange(scriptedExchange),
    injectAccessHistory(() => Promise.resolve(history)),
    injectOutputEvent((event) => {
      history.push(event);
      return Promise.resolve();
    }),
  )(runAgent)(baseSpec);

  assert(
    attemptedModels.includes(geminiFallbackVersion),
    "expected the run to attempt the fallback model after a 400 INVALID_ARGUMENT",
  );
  assert(
    history.some((e) => e.type === "own_utterance" && e.text === recoveredText),
    "expected the run to complete with the fallback model's reply",
  );
});

Deno.test("400 INVALID_ARGUMENT on both models propagates the error", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({ name: "user", text: "hi" }),
  ];
  const scriptedExchange = () => Promise.reject(invalidArgumentError());

  await assertRejects(
    () =>
      pipe(
        injectGeminiSdkExchange(scriptedExchange),
        injectAccessHistory(() => Promise.resolve(history)),
        injectOutputEvent(() => Promise.resolve()),
      )(runAgent)(baseSpec),
    Error,
    "invalid argument",
  );
});

Deno.test("geminiGenJsonFromConvo falls back to alternate model on 400 INVALID_ARGUMENT", async () => {
  const attemptedModels: string[] = [];
  const fakeGenerateContent = (req: { model?: string }) => {
    attemptedModels.push(req.model ?? "");
    if (req.model === geminiFallbackVersion) {
      return Promise.resolve(
        {
          candidates: [{ finishReason: "STOP" }],
          text: '{"ok": true}',
        } as unknown as ReturnType<
          Parameters<typeof injectGeminiGenerateContent>[0]
        > extends Promise<infer R> ? R : never,
      );
    }
    return Promise.reject(invalidArgumentError());
  };

  const result = await pipe(
    injectGeminiGenerateContent(fakeGenerateContent),
  )(() =>
    geminiGenJsonFromConvo(
      { mini: true },
      [{ role: "user", content: "hello" }],
      z.object({ ok: z.boolean() }),
    )
  )();

  assert(attemptedModels.includes(geminiFallbackVersion));
  assertEquals(result, { ok: true });
});
