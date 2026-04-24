import { assert, assertEquals } from "@std/assert";
import {
  type AudioSessionEvent,
  createDuplexPair,
  type HistoryEvent,
  runAgent,
  tool,
} from "../mod.ts";
import {
  makeSessionEventHandler,
  spokenReplyOnly,
  transcriptOf,
} from "../src/audioTransportAgent.ts";
import { injectSecrets, withRetries } from "../test_helpers.ts";
import { z } from "zod/v4";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const canRunLiveGemini = Deno.env.get("TEST_PROVIDER") === "google" &&
  !!Deno.env.get("GEMINI_API_KEY");

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
  ignore: !canRunLiveGemini,
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
    const fullText = utterances
      .map((e) => e.type === "own_utterance" ? e.text : "")
      .join(" ")
      .trim();
    assert(
      fullText.split(/\s+/).length >= 2,
      `Expected utterance to contain at least 2 words, got: "${fullText}"`,
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
  ignore: !canRunLiveGemini,
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

const runTwoBotExchange = async (): Promise<
  { aliceEvents: HistoryEvent[]; bobEvents: HistoryEvent[] } | "retry"
> => {
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

  const aliceSpoke = () => aliceEvents.some((e) => e.type === "own_utterance");
  const bobSpoke = () => bobEvents.some((e) => e.type === "own_utterance");
  const someBotThought = () =>
    [...aliceEvents, ...bobEvents].some((event) =>
      event.type === "own_thought"
    );

  await bobEndpoint.sendData({
    type: "text",
    text: "Say a short greeting out loud now.",
    from: "tester",
  });
  await waitForCondition(aliceSpoke, 15_000);

  await new Promise((r) => setTimeout(r, 2000));
  await aliceEndpoint.sendData({
    type: "text",
    text: "Say a short greeting out loud now.",
    from: "tester",
  });
  await waitForCondition(bobSpoke, 15_000);

  await Promise.all([
    aliceEndpoint.sendData({ type: "close", from: "tester" }),
    bobEndpoint.sendData({ type: "close", from: "tester" }),
  ]);
  await Promise.all([aliceTask, bobTask]);

  if ((!aliceSpoke() || !bobSpoke()) && someBotThought()) return "retry";
  if (!aliceSpoke() || !bobSpoke()) return { aliceEvents, bobEvents };
  return { aliceEvents, bobEvents };
};

Deno.test({
  name:
    "two audio bots exchange speech without duplicated participant_edit_message",
  ignore: !canRunLiveGemini,
  sanitizeOps: false,
  sanitizeResources: false,
  fn: injectSecrets(async () => {
    let result: Awaited<ReturnType<typeof runTwoBotExchange>> = "retry";
    for (let attempt = 0; attempt < 3 && result === "retry"; attempt++) {
      result = await runTwoBotExchange();
    }
    assert(result !== "retry", "Bots never produced stable speech exchange");
    const { aliceEvents, bobEvents } = result;
    if (
      !aliceEvents.some((e) => e.type === "own_utterance") ||
      !bobEvents.some((e) => e.type === "own_utterance")
    ) {
      console.log("Alice events:", aliceEvents);
      console.log("Bob events:", bobEvents);
    }
    assert(
      aliceEvents.some((e) => e.type === "own_utterance") &&
        bobEvents.some((e) => e.type === "own_utterance"),
      `Both bots failed to produce own_utterance. Alice: ${
        aliceEvents.map((e) => e.type).join(", ")
      }. Bob: ${bobEvents.map((e) => e.type).join(", ")}`,
    );

    const allEditMessages = [
      ...aliceEvents.filter((e) => e.type === "participant_edit_message"),
      ...bobEvents.filter((e) => e.type === "participant_edit_message"),
    ];

    for (const event of allEditMessages) {
      if (event.type !== "participant_edit_message") continue;
      assert(
        event.text.trim().split(/\s+/).length >= 2,
        `Expected participant_edit_message to contain at least 2 words, got: "${event.text}"`,
      );
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

Deno.test(
  "makeSessionEventHandler: tool_call after interrupted passes wasInterrupted=false",
  () => {
    const turnOutputCalls: { wasInterrupted: boolean }[] = [];
    const handler = makeSessionEventHandler({
      onAudio: () => {},
      onUtterance: () => {},
      onFlush: () => {},
      onTurnOutput: (_sessionOutput, wasInterrupted) => {
        turnOutputCalls.push({ wasInterrupted });
      },
    }).handle;

    handler({ type: "output_transcript", text: "Hello", finished: false });
    handler({ type: "interrupted" });
    handler({
      type: "tool_call",
      id: "tc1",
      name: "get_weather",
      args: { city: "Paris" },
    });

    assertEquals(turnOutputCalls.length, 2);
    assertEquals(turnOutputCalls[0].wasInterrupted, true);
    assertEquals(
      turnOutputCalls[1].wasInterrupted,
      false,
      "tool_call after interrupted should pass wasInterrupted=false so tool_call events are emitted to view-chat",
    );
  },
);

Deno.test(
  "makeSessionEventHandler emits latest utterance before tool_call boundary",
  () => {
    const utterances: string[] = [];
    const handler = makeSessionEventHandler({
      onAudio: () => {},
      onUtterance: (text) => {
        utterances.push(text);
      },
      onFlush: () => {},
      onTurnOutput: () => {},
    }).handle;

    handler({
      type: "output_transcript",
      text: "ALPHA TANGO",
      finished: false,
    });
    handler({
      type: "tool_call",
      id: "tc1",
      name: "hangUp",
      args: {},
    });

    assertEquals(utterances, ["ALPHA TANGO"]);
  },
);

Deno.test(
  "makeSessionEventHandler flushPending recovers transcript arriving after tool_call",
  () => {
    const utterances: string[] = [];
    const { handle, flushPending } = makeSessionEventHandler({
      onAudio: () => {},
      onUtterance: (text) => {
        utterances.push(text);
      },
      onFlush: () => {},
      onTurnOutput: () => {},
    });

    handle({
      type: "tool_call",
      id: "tc1",
      name: "hangUp",
      args: {},
    });
    handle({
      type: "output_transcript",
      text: "ALPHA TANGO",
      finished: false,
    });

    assertEquals(utterances, []);
    flushPending();
    assertEquals(utterances, ["ALPHA TANGO"]);
  },
);

const exampleSkill = {
  name: "secret_skill",
  description: "A skill for getting secrets",
  instructions: "Use this skill to get the secret word",
  tools: [
    tool({
      name: "get_secret",
      description: "Gets the secret word",
      parameters: z.object({}),
      handler: () => Promise.resolve("Bananarama"),
    }),
  ],
};

Deno.test({
  name: "audio agent uses skills via real Gemini session",
  ignore: !canRunLiveGemini,
  sanitizeOps: false,
  sanitizeResources: false,
  fn: withRetries(
    3,
    injectSecrets(async () => {
      const { left: testEndpoint, right: agentEndpoint } = createDuplexPair();
      const outputEvents: HistoryEvent[] = [];

      const agentTask = runAgent({
        prompt:
          "You are a math assistant. When asked to add numbers, use the secret_skill to add them. You must use the tool.",
        tools: [],
        skills: [exampleSkill],
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
        text: "Please get the secret word using your secret_skill.",
        from: "tester",
      });

      const hasRunCommandCall = () =>
        outputEvents.some((e) =>
          e.type === "tool_call" && e.name === "run_command" &&
          (e.parameters as Record<string, unknown>)?.command ===
            "secret_skill/get_secret"
        );

      await waitForCondition(
        hasRunCommandCall,
        60_000,
      );

      await testEndpoint.sendData({ type: "close", from: "tester" });
      await agentTask;

      assert(
        hasRunCommandCall(),
        `Expected run_command tool_call event for secret_skill/get_secret, got: ${
          outputEvents.map((e) =>
            e.type === "tool_call" ? `tool_call:${e.name}` : e.type
          ).join(", ")
        }`,
      );
    }),
  ),
});

Deno.test({
  name:
    "audio agent speaks the result of a skill tool when not explicitly told to",
  ignore: !canRunLiveGemini,
  sanitizeOps: false,
  sanitizeResources: false,
  fn: withRetries(
    3,
    injectSecrets(async () => {
      const { left: testEndpoint, right: agentEndpoint } = createDuplexPair();
      const outputEvents: HistoryEvent[] = [];

      const agentTask = runAgent({
        prompt:
          "You are a helpful voice assistant. Use the secret_skill to find the secret word. When you use the skill, use the run_command tool to execute secret_skill/get_secret.",
        tools: [],
        skills: [exampleSkill],
        maxIterations: 5,
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
        text: "What is the secret word?",
        from: "tester",
      });

      await waitForCondition(
        () => {
          const hasToolCall = outputEvents.some((e) =>
            e.type === "tool_call" && e.name === "run_command" &&
            (e.parameters as Record<string, unknown>)?.command ===
              "secret_skill/get_secret"
          );
          const hasUtteranceWithAnswer = outputEvents.some((e) =>
            e.type === "own_utterance" && e.text.includes("Bananarama")
          );
          return hasToolCall && hasUtteranceWithAnswer;
        },
        45_000,
      );

      await testEndpoint.sendData({ type: "close", from: "tester" });
      await agentTask;

      const utterances = outputEvents.filter((e) => e.type === "own_utterance");
      const spokenText = utterances.map((e) => (e as { text: string }).text)
        .join(
          " ",
        );

      assert(
        spokenText.includes("Bananarama"),
        `Expected agent to speak the answer "Bananarama", but it only spoke: "${spokenText}"`,
      );
    }),
  ),
});
