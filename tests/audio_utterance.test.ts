import { assert, assertEquals } from "@std/assert";
import {
  type AudioSessionEvent,
  createDuplexPair,
  type HistoryEvent,
  runAgent,
  tool,
} from "../mod.ts";
import { spokenReplyOnly, transcriptOf } from "../src/audioTransportAgent.ts";
import { injectSecrets } from "../test_helpers.ts";
import { z } from "zod/v4";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const waitForCondition = (
  predicate: () => boolean,
  timeoutMs: number,
) =>
  new Promise<void>((resolve) => {
    const deadline = setTimeout(() => resolve(), timeoutMs);
    const poll = setInterval(() => {
      if (predicate()) {
        clearTimeout(deadline);
        clearInterval(poll);
        resolve();
      }
    }, 500);
  });

Deno.test({
  name: "audio agent emits own_utterance events via real Gemini session",
  ignore: !Deno.env.get("GEMINI_API_KEY"),
  sanitizeOps: false,
  sanitizeResources: false,
  fn: injectSecrets(async () => {
    const { left: testEndpoint, right: agentEndpoint } = createDuplexPair();
    const outputEvents: HistoryEvent[] = [];

    const agentTask = runAgent({
      prompt:
        "You are a friendly assistant. Respond with a short greeting back. Keep responses under 10 words.",
      tools: [],
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio" as const,
        endpoint: agentEndpoint,
        voiceName: "Zephyr",
        participantName: "User",
      },
      onOutputEvent: (event) => {
        outputEvents.push(event);
        return Promise.resolve();
      },
      rewriteHistory: async () => {},
    });

    await testEndpoint.sendData({
      type: "text",
      text: "Hello!",
      from: "tester",
    });

    await waitForCondition(
      () => outputEvents.some((e) => e.type === "own_utterance"),
      30_000,
    );

    await testEndpoint.sendData({ type: "close", from: "tester" });
    await agentTask;

    const utterances = outputEvents.filter((e) => e.type === "own_utterance");
    assert(
      utterances.length > 0,
      `Expected at least one own_utterance event, got: ${
        outputEvents.map((e) => e.type).join(", ")
      }`,
    );
    assert(
      utterances[0].type === "own_utterance" && utterances[0].text.length > 0,
      "Utterance text should be non-empty",
    );
  }),
});

const fastTool = tool({
  name: "get_weather",
  description: "Get the current weather for a city. Returns immediately.",
  parameters: z.object({ city: z.string() }),
  handler: ({ city }) => Promise.resolve(`Sunny, 22°C in ${city}`),
});

const slowTool = tool({
  name: "book_flight",
  description:
    "Book a flight to a destination. Takes a few seconds to process.",
  parameters: z.object({
    destination: z.string(),
  }),
  handler: async ({ destination }) => {
    await delay(3000);
    return `Flight booked to ${destination}, confirmation #AB123`;
  },
});

Deno.test({
  name: "audio agent uses fast and slow tools via real Gemini session",
  ignore: !Deno.env.get("GEMINI_API_KEY"),
  sanitizeOps: false,
  sanitizeResources: false,
  fn: injectSecrets(async () => {
    const { left: testEndpoint, right: agentEndpoint } = createDuplexPair();
    const outputEvents: HistoryEvent[] = [];

    const agentTask = runAgent({
      prompt:
        "You are a travel assistant. When the user asks you to plan a trip, you MUST call BOTH the get_weather tool AND the book_flight tool. Always call both tools.",
      tools: [fastTool, slowTool],
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio" as const,
        endpoint: agentEndpoint,
        voiceName: "Zephyr",
        participantName: "User",
      },
      onOutputEvent: (event) => {
        outputEvents.push(event);
        return Promise.resolve();
      },
      rewriteHistory: async () => {},
    });

    await testEndpoint.sendData({
      type: "text",
      text:
        "Plan a trip to Paris for me. Check the weather there and book a flight.",
      from: "tester",
    });

    const hasToolCallFor = (name: string) =>
      outputEvents.some((e) => e.type === "tool_call" && e.name === name);

    await waitForCondition(
      () => hasToolCallFor("get_weather") && hasToolCallFor("book_flight"),
      60_000,
    );

    await testEndpoint.sendData({ type: "close", from: "tester" });
    await agentTask;

    assert(
      hasToolCallFor("get_weather"),
      `Expected get_weather tool_call event, got: ${
        outputEvents.map((e) =>
          e.type === "tool_call" ? `tool_call:${e.name}` : e.type
        ).join(", ")
      }`,
    );
    assert(
      hasToolCallFor("book_flight"),
      `Expected book_flight tool_call event, got: ${
        outputEvents.map((e) =>
          e.type === "tool_call" ? `tool_call:${e.name}` : e.type
        ).join(", ")
      }`,
    );
  }),
});

Deno.test("spokenReplyOnly strips reasoning preamble", () => {
  assertEquals(spokenReplyOnly("Hello, how are you?"), "Hello, how are you?");
  assertEquals(
    spokenReplyOnly(
      "**Internal Reasoning**\n\nI am triggering the immediate use of tool X",
    ),
    "",
  );
  assertEquals(
    spokenReplyOnly("Now, I will call the function to do something"),
    "",
  );
  assertEquals(
    spokenReplyOnly("The relay code is 1234, triggering the immediate use of"),
    "The relay code is 1234, triggering the immediate use of",
  );
});

Deno.test(
  "transcriptOf returns last accumulated text, not joined duplicates",
  () => {
    const events: AudioSessionEvent[] = [
      { type: "input_transcript", text: "Hey", finished: false },
      { type: "input_transcript", text: "Hey there", finished: false },
      {
        type: "input_transcript",
        text: "Hey there. How are you?",
        finished: true,
      },
      { type: "turn_complete" },
    ];
    assertEquals(
      transcriptOf(events, "input_transcript"),
      "Hey there. How are you?",
    );
  },
);

Deno.test({
  name:
    "two audio bots exchange speech without duplicated participant_edit_message",
  ignore: !Deno.env.get("GEMINI_API_KEY"),
  sanitizeOps: false,
  sanitizeResources: false,
  fn: injectSecrets(async () => {
    const { left: aliceEndpoint, right: bobEndpoint } = createDuplexPair();
    const aliceEvents: HistoryEvent[] = [];
    const bobEvents: HistoryEvent[] = [];

    const aliceTask = runAgent({
      prompt:
        "You are Alice. Greet Bob briefly (under 10 words). Then say goodbye and stop talking.",
      tools: [],
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio" as const,
        endpoint: aliceEndpoint,
        voiceName: "Zephyr",
        participantName: "Bob",
      },
      onOutputEvent: (event) => {
        aliceEvents.push(event);
        return Promise.resolve();
      },
      rewriteHistory: async () => {},
    });

    const bobTask = runAgent({
      prompt:
        "You are Bob. Greet Alice briefly (under 10 words). Then say goodbye and stop talking.",
      tools: [],
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      timezoneIANA: "UTC",
      transport: {
        kind: "audio" as const,
        endpoint: bobEndpoint,
        voiceName: "Orus",
        participantName: "Alice",
      },
      onOutputEvent: (event) => {
        bobEvents.push(event);
        return Promise.resolve();
      },
      rewriteHistory: async () => {},
    });

    await aliceEndpoint.sendData({
      type: "text",
      text: "Start with a short greeting.",
      from: "tester",
    });

    await waitForCondition(
      () =>
        aliceEvents.some((e) => e.type === "participant_edit_message") ||
        bobEvents.some((e) => e.type === "participant_edit_message"),
      60_000,
    );

    await Promise.all([
      aliceEndpoint.sendData({ type: "close", from: "tester" }),
      bobEndpoint.sendData({ type: "close", from: "tester" }),
    ]);
    await Promise.all([aliceTask, bobTask]);

    const allEditMessages = [
      ...aliceEvents.filter((e) => e.type === "participant_edit_message"),
      ...bobEvents.filter((e) => e.type === "participant_edit_message"),
    ];

    for (const event of allEditMessages) {
      if (event.type !== "participant_edit_message") continue;
      const words = event.text.split(/\s+/);
      const uniqueThreeGrams = new Set<string>();
      let duplicateCount = 0;
      for (let i = 0; i <= words.length - 3; i++) {
        const gram = words.slice(i, i + 3).join(" ");
        if (uniqueThreeGrams.has(gram)) duplicateCount++;
        uniqueThreeGrams.add(gram);
      }
      assert(
        duplicateCount <= 1,
        `participant_edit_message has duplicated text (${duplicateCount} repeated 3-grams): "${event.text}"`,
      );
    }
  }),
});
