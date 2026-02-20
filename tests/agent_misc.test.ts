import { assert, assertEquals } from "@std/assert";
import { each, pipe } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z } from "zod/v4";
import {
  geminiGenJsonFromConvo,
  openAiGenJsonFromConvo,
  runAgent,
} from "../mod.ts";
import {
  type HistoryEvent,
  overrideTime,
  participantUtteranceTurn,
} from "../src/agent.ts";
import {
  agentDeps,
  injectSecrets,
  llmTest,
  noopRewriteHistory,
} from "../test_helpers.ts";

Deno.test(
  "returns valid result for hello schema",
  injectSecrets(async () => {
    const schema = z.object({ hello: z.string() });
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: "Say hello as JSON." },
      { role: "user", content: "hello" },
    ];
    await each((service) =>
      each(async (mini) => {
        const result = await service({ mini }, messages, schema);
        assertEquals(result, { hello: result.hello });
      })([true, false])
    )([openAiGenJsonFromConvo, geminiGenJsonFromConvo]);
  }),
);

llmTest(
  "agent repeats back order of four speakers",
  injectSecrets(async () => {
    const mockHistory = [
      { name: "Alice", text: "Hi everyone" },
      { name: "Bob", text: "Yo" },
      { name: "Carol", text: "Howdy" },
      { name: "Dave", text: "Hello" },
      {
        name: "Alice",
        text:
          "List the speakers in the order they first spoke. Reply ONLY with: Alice,Bob,Carol,Dave",
      },
    ].map(participantUtteranceTurn);

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are an AI that strictly follows formatting instructions. When asked to list speakers, reply exactly as instructed without extra text.",
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "UTC",
    });

    const answer = mockHistory.find((
      e,
    ): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
      e.type === "own_utterance" && "text" in e && typeof e.text === "string" &&
      e.text.trim().startsWith("Alice")
    );
    assert(answer, "AI should respond with an own_utterance");
    const normalized = answer.text.replace(/\s/g, "");
    assertEquals(normalized, "Alice,Bob,Carol,Dave");
  }),
);

Deno.test(
  "agent knows the time from message timestamps",
  pipe(
    injectSecrets,
    overrideTime(() => new Date("2026-02-05T21:34:00Z").getTime()),
  )(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "What time is it right now?",
      }),
    ];

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are a helpful assistant. When asked about the time, look at the timestamp shown in brackets before the user's message.",
      lightModel: true,
      rewriteHistory: noopRewriteHistory,
      timezoneIANA: "America/New_York",
    });

    const response = mockHistory.find((
      e,
    ): e is Extract<HistoryEvent, { type: "own_utterance" }> =>
      e.type === "own_utterance" && !!e.text
    );
    assert(response, "AI should respond with an own_utterance");
    assert(
      response.text.includes("4:34") || response.text.includes("4:34 PM"),
      `AI should mention the time 4:34 PM (got: ${response.text})`,
    );
  }),
);
