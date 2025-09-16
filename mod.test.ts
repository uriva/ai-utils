import { assert, assertEquals } from "@std/assert";
import { each, pipe, sleep } from "gamla";
import type { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { z } from "zod/v4";
import {
  geminiGenJsonFromConvo,
  injectCacher,
  injectGeminiToken,
  injectOpenAiToken,
  openAiGenJsonFromConvo,
  runAgent,
} from "./mod.ts";
import {
  type HistoryEvent,
  injectAccessHistory,
  injectOutputEvent,
  ownUtteranceTurn,
  participantUtteranceTurn,
  toolResultTurn,
  type ToolReturn,
  toolUseTurn,
} from "./src/agent.ts";
import type { FnToSameFn } from "./src/utils.ts";

const injectSecrets = pipe(
  injectCacher(() => (f) => f),
  injectOpenAiToken(
    Deno.env.get("OPENAI_API_KEY") ?? "",
  ),
  injectGeminiToken(Deno.env.get("GEMINI_API_KEY") ?? ""),
);

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
        console.log(result);
        assertEquals(result, { hello: result.hello });
      })([true, false])
    )([openAiGenJsonFromConvo, geminiGenJsonFromConvo]);
  }),
);

Deno.test(
  "agent can run run when history starts with only a model message",
  injectSecrets(async () => {
    await agentDeps([ownUtteranceTurn("Priming without user turn")])(runAgent)({
      maxIterations: 1,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You are a helper.",
      lightModel: true,
    });
  }),
);

const agentDeps = (inMemoryHistory: HistoryEvent[]): FnToSameFn =>
  pipe(
    injectAccessHistory(() => Promise.resolve(inMemoryHistory)),
    injectOutputEvent((event) => {
      inMemoryHistory.push(event);
      return Promise.resolve();
    }),
  );

const toolResult = "43212e8e-4c29-4a3c-aba2-723e668b5537";

const someTool = {
  name: "doSomethingUnique",
  description: "Returns a unique string so we know the tool was called.",
  parameters: z.object({}),
  handler: () => Promise.resolve(toolResult),
};

Deno.test(
  "runBot calls the tool and replies with its output",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text:
        `Please call the doSomethingUnique tool now and only reply with its output.`,
    })];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: `You are an AI assistant.`,
    });
    assert(
      mockHistory.some((event) => (event.type === "own_utterance" &&
        event.text.includes(toolResult))
      ),
      `AI did not reply with tool output. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test(
  "agent can start an empty conversation",
  injectSecrets(async () => {
    await agentDeps([])(runAgent)({
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
    });
  }),
);

Deno.test(
  "conversation can start with a tool call",
  injectSecrets(async () => {
    await agentDeps([
      toolUseTurn({ name: someTool.name, args: {} }),
      toolResultTurn({ name: someTool.name, result: toolResult }),
    ])(
      runAgent,
    )({
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt: `You are the neighborhood friendly spiderman.`,
      maxIterations: 5,
    });
  }),
);

Deno.test(
  "ai returns text event before calling actions",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text:
          `Please call the doSomethingUnique tool and explain what you're doing.`,
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 5,
      onMaxIterationsReached: () => {},
      tools: [someTool],
      prompt:
        `You are an AI assistant. Always explain what you're doing before using tools.`,
    });
    const firstTextIndex = mockHistory.findIndex((event) =>
      event.type === "own_utterance" && event.text
    );
    const firstToolIndex = mockHistory.findIndex((event) =>
      event.type === "tool_call"
    );
    assert(firstTextIndex >= 0, "AI should produce text output");
    assert(firstToolIndex >= 0, "AI should call the tool");
    assert(
      firstTextIndex < firstToolIndex,
      "Text should come before tool call",
    );
  }),
);

Deno.test(
  "ai handles new history items while waiting for function calls",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: `Please call the slowTool.`,
    })];

    const slowTool = {
      name: "slowTool",
      description: "A tool that takes time to execute",
      parameters: z.object({}),
      handler: async () => {
        // Add a new event to history while tool is running
        await sleep(10);
        mockHistory.push(participantUtteranceTurn({
          name: "user",
          text: "While you're working, here's additional context.",
        }));
        return "slow result";
      },
    };

    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 10,
      onMaxIterationsReached: () => {},
      tools: [slowTool],
      prompt: `You are an AI assistant. Always acknowledge new messages.`,
    });

    // Check that the tool was called
    const toolCall = mockHistory.find((e) =>
      e.type === "tool_call" && e.name === "slowTool"
    );
    assert(toolCall, "slowTool should be called");

    // Check that the additional message was added during tool execution
    const addedContext = mockHistory.find((e) =>
      e.type === "participant_utterance" &&
      e.text?.includes("additional context")
    );
    assert(addedContext, "Additional message should be in history");

    // Since the agent processes history in iterations, the additional message
    // should be processed in a subsequent iteration
    const responseAfterContext = mockHistory.find((event) =>
      event.type === "own_utterance" &&
      event.text &&
      event.timestamp > addedContext.timestamp
    );
    assert(
      responseAfterContext,
      "AI should respond to additional message in next iteration",
    );
  }),
);

Deno.test(
  "maxIterationsReached aborts the loop",
  injectSecrets(async () => {
    let callbackCalled = false;
    const mockHistory: HistoryEvent[] = [participantUtteranceTurn({
      name: "user",
      text: `Please keep talking and calling tools continuously.`,
    })];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3, // Very low limit to ensure we hit it
      onMaxIterationsReached: () => {
        callbackCalled = true;
      },
      tools: [{
        name: "continueTalking",
        description: "A tool that keeps the conversation going",
        parameters: z.object({}),
        handler: async () => {
          // Add another user message to keep the conversation going
          await sleep(5); // Small delay to satisfy async requirement
          mockHistory.push(participantUtteranceTurn({
            name: "user",
            text: "Keep going, call the tool again!",
          }));
          return "continue";
        },
      }],
      prompt:
        `You are a chatty AI. Always call the continueTalking tool in every response and keep the conversation going.`,
    });
    assert(callbackCalled, "onMaxIterationsReached callback should be called");
  }),
);

Deno.test(
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
  "agent triggers do nothing event when model should say nothing",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({ name: "user", text: "Say nothing." }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 2,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt:
        "You are an AI assistant. If the user says 'Say nothing.', do not reply with any text.",
      lightModel: true,
    });
    assertEquals(mockHistory[mockHistory.length - 1].type, "do_nothing");
  }),
);

const toBase64 = (u8: Uint8Array): string => {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < u8.length; i += chunk) {
    binary += String.fromCharCode(...u8.subarray(i, i + chunk));
  }
  return btoa(binary);
};

const bytes = await Deno.readFile("./dog.jpg");

const b64 = toBase64(bytes);

const mediaTool = {
  name: "returnMedia",
  description: "Returns media via attachments",
  parameters: z.object({}),
  handler: () => {
    const ret: ToolReturn = {
      result: "image attached",
      attachments: [
        { kind: "inline", mimeType: "image/jpeg", dataBase64: b64 },
      ],
    };
    return Promise.resolve(ret);
  },
};

Deno.test(
  "tool result attachments are forwarded to model",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please call returnMedia and then describe the image.",
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [mediaTool],
      prompt: "You can see images returned by tools.",
      lightModel: true,
    });
    assert(
      mockHistory.some((e) =>
        e.type === "own_utterance" && e.text.toLowerCase().includes("dog")
      ),
      `AI did not describe the image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);

Deno.test(
  "user attachments are forwarded to model",
  injectSecrets(async () => {
    const mockHistory: HistoryEvent[] = [
      participantUtteranceTurn({
        name: "user",
        text: "Please describe the attached image.",
        attachments: [
          { kind: "inline", mimeType: "image/jpeg", dataBase64: b64 },
        ],
      }),
    ];
    await agentDeps(mockHistory)(runAgent)({
      maxIterations: 3,
      onMaxIterationsReached: () => {},
      tools: [],
      prompt: "You can see images attached by the user.",
      lightModel: true,
    });
    assert(
      mockHistory.some((e) =>
        e.type === "own_utterance" && e.text.toLowerCase().includes("dog")
      ),
      `AI did not describe the image as a dog. History: ${
        JSON.stringify(mockHistory, null, 2)
      }`,
    );
  }),
);
