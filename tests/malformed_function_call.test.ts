import { assert, assertEquals } from "@std/assert";
import { z } from "zod/v4";
import { runAgent, tool } from "../mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  participantUtteranceTurn,
} from "../src/agent.ts";
import { injectGeminiSdkExchange } from "../src/geminiAgent.ts";
import { pipe } from "gamla";

// Gemini-specific: MALFORMED_FUNCTION_CALL is a Gemini-only finish reason, so
// this test cannot run on other providers. Reproduces a production incident
// where Gemini streamed an empty-args functionCall shell plus the raw arg-JSON
// as text fragments, and the platform shipped the fragments to the user as
// utterances and executed the empty call. The exchange is scripted via
// injectGeminiSdkExchange, so no API call is made.

const updateUserField = tool({
  name: "update_user_field",
  description: "Update a field on the user profile",
  parameters: z.object({
    userId: z.string(),
    weight: z.number().optional(),
    goal: z.string().optional(),
  }),
  handler: () => Promise.resolve(JSON.stringify({ success: true })),
});

// What Gemini actually returned in the incident: a functionCall shell with
// empty args, and the intended args JSON scattered across text parts.
const malformedExchange = {
  parts: [
    { text: "I should save the user's goal now.", thought: true },
    { functionCall: { name: "update_user_field", args: {} } },
    { text: ",userId:" },
    { text: "ירידה במשקל" },
    { text: "}" },
    { text: "972506207421" },
  ],
  finishReason: "MALFORMED_FUNCTION_CALL",
};

const validToolCallExchange = {
  parts: [{
    thoughtSignature: "sig",
    functionCall: {
      id: "call-1",
      name: "update_user_field",
      args: { userId: "972506207421", goal: "ירידה במשקל" },
    },
  }],
  finishReason: "STOP",
};

const finalTextExchange = {
  parts: [{ text: "Goal saved!" }],
  finishReason: "STOP",
};

const exchanges = [
  malformedExchange,
  validToolCallExchange,
  finalTextExchange,
];

Deno.test("MALFORMED_FUNCTION_CALL with garbage parts is retried, not leaked to the user", async () => {
  const history: HistoryEvent[] = [
    participantUtteranceTurn({
      name: "user",
      text: "my goal is a healthy lifestyle",
    }),
  ];
  let exchangeCount = 0;
  const scriptedExchange = () => {
    const exchange = exchanges[exchangeCount] ?? finalTextExchange;
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
    tools: [updateUserField],
    prompt: "You are a nutrition assistant.",
    rewriteHistory: () => Promise.resolve(),
    timezoneIANA: "UTC",
  });

  const leaked = history.filter((e) =>
    e.type === "own_utterance" &&
    ["userId", "972506207421", "ירידה במשקל", "}"].some((fragment) =>
      e.text.includes(fragment)
    )
  );
  assertEquals(
    leaked,
    [],
    "arg-JSON fragments of a malformed function call must not reach the user",
  );

  const badCalls = history.filter((e) =>
    e.type === "tool_call" &&
    (typeof e.parameters !== "object" || e.parameters === null ||
      !("userId" in e.parameters))
  );
  assertEquals(
    badCalls,
    [],
    "the empty-args functionCall shell of a malformed call must not execute",
  );

  assert(
    history.some((e) =>
      e.type === "tool_call" && e.name === "update_user_field" &&
      typeof e.parameters === "object" && e.parameters !== null &&
      "userId" in e.parameters && e.parameters.userId === "972506207421"
    ),
    "expected the retried call to update the user field with the real args",
  );

  assertEquals(
    exchangeCount,
    3,
    "malformed exchange → retry with valid tool call → final text",
  );
});
