import { assert, assertEquals } from "@std/assert";
import { createDuplexPair, type HistoryEvent, runAgent, tool } from "../mod.ts";
import { spokenReplyOnly } from "../src/audioTransportAgent.ts";
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
