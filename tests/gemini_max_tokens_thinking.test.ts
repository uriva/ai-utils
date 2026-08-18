import { assertEquals } from "@std/assert";
import { runAgent } from "../mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  participantUtteranceTurn,
  thinkingTokenExhaustionWarningText,
} from "../src/agent.ts";
import { injectGeminiSdkExchange } from "../src/geminiAgent.ts";
import { pipe } from "gamla";

// Gemini-specific: when Gemini finishes with MAX_TOKENS while generating thoughts
// (0 non-thought text and 0 function calls), it must NOT silently yield do_nothing.
// Instead it produces thinkingTokenExhaustionWarningText with truncated=true, which
// triggers the truncation retry and surfaces a visible warning if retries exhaust.

const thoughtOnlyMaxTokensExchange = {
  parts: [
    {
      text: "Drafting a very large implementation in internal reasoning...",
      thought: true,
    },
  ],
  finishReason: "MAX_TOKENS",
};

const recoveredExchange = {
  parts: [{ text: "Here is the concise implementation." }],
  finishReason: "STOP",
};

Deno.test(
  "Gemini MAX_TOKENS with thought-only parts retries and recovers",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please build this plugin.",
      }),
    ];

    let exchangeCount = 0;
    const scriptedExchange = () => {
      const exchange = exchangeCount === 0
        ? thoughtOnlyMaxTokensExchange
        : recoveredExchange;
      exchangeCount++;
      return Promise.resolve(exchange);
    };

    await pipe(
      injectGeminiSdkExchange(scriptedExchange),
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      }),
    )(runAgent)({
      maxIterations: 5,
      tools: [],
      prompt: "You are a coding assistant.",
      rewriteHistory: () => Promise.resolve(),
      timezoneIANA: "UTC",
    });

    assertEquals(
      exchangeCount,
      2,
      "expected initial MAX_TOKENS turn followed by retry",
    );
    const utterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(utterances.length, 1);
    if (utterances[0].type !== "own_utterance") throw new Error("unreachable");
    assertEquals(utterances[0].text, "Here is the concise implementation.");
    const doNothings = history.filter((e) => e.type === "do_nothing");
    assertEquals(
      doNothings.length,
      0,
      "must not emit do_nothing when truncated by MAX_TOKENS",
    );
  },
);

Deno.test(
  "Gemini MAX_TOKENS with thought-only parts emits visible warning when retries exhaust",
  async () => {
    const history: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please build this plugin.",
      }),
    ];

    let exchangeCount = 0;
    const scriptedExchange = () => {
      exchangeCount++;
      return Promise.resolve(thoughtOnlyMaxTokensExchange);
    };

    await pipe(
      injectGeminiSdkExchange(scriptedExchange),
      injectAccessHistory(() => Promise.resolve(history)),
      injectOutputEvent((event) => {
        history.push(event);
        return Promise.resolve();
      }),
    )(runAgent)({
      maxIterations: 5,
      tools: [],
      prompt: "You are a coding assistant.",
      rewriteHistory: () => Promise.resolve(),
      timezoneIANA: "UTC",
    });

    const utterances = history.filter((e) => e.type === "own_utterance");
    assertEquals(utterances.length, 1, "expected visible fallback utterance");
    if (utterances[0].type !== "own_utterance") throw new Error("unreachable");
    assertEquals(utterances[0].text, thinkingTokenExhaustionWarningText);
    const doNothings = history.filter((e) => e.type === "do_nothing");
    assertEquals(
      doNothings.length,
      0,
      "must not emit do_nothing when retries exhaust on MAX_TOKENS",
    );
  },
);
